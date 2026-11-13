"""Postgres + pgvector :class:`Memory` backend.

Schema (created by :meth:`init_schema`):

* ``memory_blocks(namespace, name, content, pinned_order, updated_at)``
* ``episodes(id, namespace, session_id, occurred_at, input, output,
  embedding vector(N))`` with HNSW cosine index on ``embedding``

The ``vector(N)`` column dimension is fixed at table-creation time and
must match the configured embedder's ``dimensions``. Switching
embedders later requires migrating the table.

The table names above are the defaults. ``table_prefix="myapp_"``
and / or ``schema="agent"`` (on the constructor and :meth:`connect`)
move every table — ``episodes``, ``memory_blocks``,
``episode_tool_transcripts`` and the fact store's ``facts`` — out of
the way of an application that already has tables with those names.
:meth:`init_schema` refuses to touch an existing ``episodes`` /
``memory_blocks`` table that isn't this backend's.

pgvector can HNSW-index at most 2000 dimensions. With a wider
embedder (``text-embedding-3-large`` is 3072) the vector index is
skipped with a warning and recall runs an exact scan.

Both ``asyncpg`` and ``pgvector`` are imported lazily inside
:meth:`connect` / :meth:`init_schema` so the module loads in
environments without those extras installed; the import only fires
when actually opening a connection.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from ..core.errors import ConfigError
from ..core.protocols import Embedder
from ..core.types import (
    Episode,
    EpisodeMatch,
    Fact,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
    Message,
    Role,
)
from ..vectorstore._pgvector import (
    HNSW_MAX_DIMENSIONS,
    hnsw_scan,
    index_reference,
    table_reference,
    validate_schema,
    validate_table_prefix,
)
from . import _extractions
from ._bounds import (
    DEFAULT_MAX_BLOCK_CHARS,
    resolve_max_block_chars,
    validate_limit,
)
from ._embedding_util import vector_to_list
from ._user_key import ANON_USER_ID, decode_user_id, encode_user_id
from .embedder import HashEmbedder, warn_hash_embedder_fallback
from .facts import count_facts, delete_facts

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "default"

# Columns a table must have to be one of ours. ``init_schema`` runs
# ``ALTER TABLE`` migrations, so it first makes sure an existing table
# of the same name really is this backend's (any loomflow version)
# and not the application's own ``episodes`` table.
_EPISODES_COLUMNS = frozenset(
    {"id", "namespace", "session_id", "occurred_at", "input", "output", "embedding"}
)
_BLOCKS_COLUMNS = frozenset({"namespace", "name", "content"})

# Postgres can't carry NULL inside a primary key column, so the
# anonymous bucket needs a non-NULL representation on the wire.
# Earlier versions used the empty string ``''`` for this — that
# silently conflicts with anyone who passes ``""`` as a real
# user_id and looks like a hack in the schema. M10 replaced it with
# a reserved sentinel; the encoding now lives in the shared
# ``memory._user_key`` module so every backend agrees on it. The
# old private names are kept as aliases for back-compat imports.
_ANON_USER_ID = ANON_USER_ID
_encode_user_id = encode_user_id
_decode_user_id = decode_user_id


class PostgresMemory:
    """Postgres-backed :class:`Memory`.

    ``pool`` is an ``asyncpg.Pool`` (or anything with the same API).
    Tests can pass a fake pool whose ``acquire()`` returns a fake
    connection.

    ``table_prefix`` / ``schema`` rename and relocate the tables (see
    the module docstring); both must be plain identifiers and default
    to the historical unprefixed names in the connection's search
    path. ``append_block`` keeps the newest ``max_block_chars``
    characters of a block (default 20,000; ``None`` for no limit).
    """

    def __init__(
        self,
        pool: Any,
        *,
        embedder: Embedder | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        fact_store: Any | None = None,
        table_prefix: str = "",
        schema: str | None = None,
        max_block_chars: int | None = DEFAULT_MAX_BLOCK_CHARS,
    ) -> None:
        self._pool = pool
        self._table_prefix = validate_table_prefix(table_prefix)
        self._schema = validate_schema(schema)
        self._blocks_table = table_reference(
            self._schema, self._table_prefix, "memory_blocks"
        )
        self._episodes_table = table_reference(
            self._schema, self._table_prefix, "episodes"
        )
        self._transcripts_table = table_reference(
            self._schema, self._table_prefix, "episode_tool_transcripts"
        )
        # Validate the longest derived index name up front too.
        self._index("episode_tool_transcripts_episode_idx")
        self._max_block_chars = resolve_max_block_chars(max_block_chars)
        if embedder is None:
            warn_hash_embedder_fallback("PostgresMemory")
            embedder = HashEmbedder()
        self._embedder: Embedder = embedder
        self._namespace = namespace
        # ``facts`` is left as ``None`` by default to avoid forcing the
        # facts schema on users who don't want it. Pass an explicit
        # :class:`PostgresFactStore` (or set ``with_facts=True`` on
        # :meth:`connect`) to enable it.
        self.facts: Any | None = fact_store

    def _index(self, name: str) -> str:
        return index_reference(self._table_prefix, name)

    # ---- factory ---------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        embedder: Embedder | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        min_size: int = 1,
        max_size: int = 10,
        with_facts: bool = False,
        init_schema: bool = False,
        table_prefix: str = "",
        schema: str | None = None,
        max_block_chars: int | None = DEFAULT_MAX_BLOCK_CHARS,
    ) -> PostgresMemory:
        """Open an asyncpg pool and register the pgvector codec.

        When ``with_facts=True`` a :class:`PostgresFactStore` rooted at
        the same pool is attached as ``self.facts`` so the agent loop's
        memory.facts integration just works.

        ``init_schema=True`` creates the ``vector`` extension and the
        tables if they don't exist (idempotent): the default when the
        store comes from a ``memory="postgres://..."`` URL.

        ``table_prefix`` / ``schema`` apply to the fact store too.
        """
        # Bad identifiers fail here, before any connection is opened.
        validate_table_prefix(table_prefix)
        validate_schema(schema)
        try:
            import asyncpg  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "asyncpg is not installed. "
                "Install with: pip install 'loomflow[postgres]'"
            ) from exc
        try:
            from pgvector.asyncpg import (  # type: ignore[import-not-found, import-untyped]
                register_vector,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pgvector is not installed. "
                "Install with: pip install pgvector"
            ) from exc

        async def _setup(conn: Any) -> None:
            await register_vector(conn)

        if init_schema:
            # The pool's codec setup needs the ``vector`` type to exist.
            bootstrap = await asyncpg.connect(dsn=dsn)
            try:
                await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await bootstrap.close()
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            init=_setup,
        )
        instance = cls(
            pool,
            embedder=embedder,
            namespace=namespace,
            table_prefix=table_prefix,
            schema=schema,
            max_block_chars=max_block_chars,
        )
        if with_facts:
            from .postgres_facts import PostgresFactStore

            instance.facts = PostgresFactStore(
                pool,
                embedder=embedder,
                table_prefix=table_prefix,
                schema=schema,
            )
        if init_schema:
            await instance.init_schema()
        return instance

    async def aclose(self) -> None:
        if self._pool is not None and hasattr(self._pool, "close"):
            await self._pool.close()

    # ---- schema ----------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def embedding_dimensions(self) -> int:
        return self._embedder.dimensions

    def schema_sql(self) -> list[str]:
        """Return the DDL needed to bootstrap this backend's schema.

        Exposed so tests can assert on the SQL without running it; also
        usable from migration scripts that want to apply the schema in
        their own transaction.
        """
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            (
                f"CREATE TABLE IF NOT EXISTS {self._blocks_table} ("
                "  namespace TEXT NOT NULL,"
                # ``user_id`` cannot be NULL inside a PK in Postgres,
                # so the anonymous bucket gets a reserved sentinel
                # on the wire (round-tripped to ``None`` in code).
                # The default also uses the sentinel so legacy
                # callers that don't set the column land in the
                # anonymous bucket — same behaviour as Python's
                # ``user_id=None`` default elsewhere.
                f"  user_id TEXT NOT NULL DEFAULT '{_ANON_USER_ID}',"
                "  name TEXT NOT NULL,"
                "  content TEXT NOT NULL,"
                "  pinned_order INT NOT NULL DEFAULT 0,"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
                "  PRIMARY KEY (namespace, user_id, name)"
                ");"
            ),
            # Idempotent ALTER for upgrades from pre-M9 schemas. The
            # default '' was used by 0.9.x; 0.10+ uses the sentinel.
            # Both ALTERs are no-ops if the column already exists at
            # the right shape.
            f"ALTER TABLE {self._blocks_table} ADD COLUMN IF NOT EXISTS "
            f"user_id TEXT NOT NULL DEFAULT '{_ANON_USER_ID}';",
            # Migrate 0.9.x-era empty-string rows to the sentinel.
            # No-op once already migrated.
            f"UPDATE {self._blocks_table} SET user_id = "
            f"'{_ANON_USER_ID}' WHERE user_id = '';",
            # Update the column default so freshly-inserted rows
            # without an explicit user_id (legacy callers) land in
            # the sentinel bucket too.
            f"ALTER TABLE {self._blocks_table} ALTER COLUMN user_id "
            f"SET DEFAULT '{_ANON_USER_ID}';",
            (
                f"CREATE TABLE IF NOT EXISTS {self._episodes_table} ("
                f"  id TEXT PRIMARY KEY,"
                f"  namespace TEXT NOT NULL,"
                f"  session_id TEXT NOT NULL,"
                f"  user_id TEXT,"
                f"  occurred_at TIMESTAMPTZ NOT NULL,"
                f"  input TEXT NOT NULL,"
                f"  output TEXT NOT NULL,"
                f"  embedding vector({self.embedding_dimensions}) NOT NULL"
                f");"
            ),
            # Idempotent ALTER for upgrades from pre-``user_id`` schemas.
            (
                f"ALTER TABLE {self._episodes_table} "
                "ADD COLUMN IF NOT EXISTS user_id TEXT;"
            ),
            (
                "CREATE INDEX IF NOT EXISTS "
                f"{self._index('episodes_namespace_idx')} "
                f"ON {self._episodes_table} (namespace, occurred_at DESC);"
            ),
            (
                "CREATE INDEX IF NOT EXISTS "
                f"{self._index('episodes_user_id_idx')} "
                f"ON {self._episodes_table} "
                "(namespace, user_id, occurred_at DESC);"
            ),
            # ``session_messages`` filters on (namespace, session_id)
            # and sorts by occurred_at — without this index it scans
            # the whole namespace partition.
            (
                "CREATE INDEX IF NOT EXISTS "
                f"{self._index('episodes_session_idx')} "
                f"ON {self._episodes_table} "
                "(namespace, session_id, occurred_at DESC);"
            ),
            # Tool-transcript sidecar table — see the matching
            # comment in sqlite.py for the design rationale. One
            # row per captured tool_call / tool_result message,
            # joined by ``episode_id``. ON DELETE CASCADE keeps
            # GDPR forget semantics intact. Idempotent across
            # restarts; pre-existing DBs add the table on next
            # init without touching existing episodes.
            (
                f"CREATE TABLE IF NOT EXISTS {self._transcripts_table} ("
                "  episode_id TEXT NOT NULL "
                f"    REFERENCES {self._episodes_table}(id) ON DELETE CASCADE,"
                "  sequence INTEGER NOT NULL,"
                "  message_json TEXT NOT NULL,"
                "  PRIMARY KEY (episode_id, sequence)"
                ");"
            ),
            (
                "CREATE INDEX IF NOT EXISTS "
                f"{self._index('episode_tool_transcripts_episode_idx')} "
                f"ON {self._transcripts_table} (episode_id, sequence);"
            ),
        ]
        if self._schema is not None:
            statements.insert(1, f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";')
        # pgvector's HNSW index takes at most 2000 dimensions; a wider
        # column simply gets no ANN index (``init_schema`` warns) and
        # recall is an exact scan.
        if self.embedding_dimensions <= HNSW_MAX_DIMENSIONS:
            statements.append(
                "CREATE INDEX IF NOT EXISTS "
                f"{self._index('episodes_embedding_idx')} "
                f"ON {self._episodes_table} "
                "USING hnsw (embedding vector_cosine_ops);"
            )
        return statements

    async def init_schema(self) -> None:
        """Apply :meth:`schema_sql` against the connected pool.

        When a :class:`PostgresFactStore` is attached as ``self.facts``,
        its schema is initialised in the same call.

        Raises :class:`ConfigError` — before changing anything — when
        a table named like one of ours already exists but isn't ours
        (an application's own ``episodes`` table, say): pass
        ``table_prefix=`` or ``schema=`` to keep the two apart.
        """
        if self.embedding_dimensions > HNSW_MAX_DIMENSIONS:
            logger.warning(
                "PostgresMemory: the embedder produces %d-dimensional "
                "vectors, above pgvector's %d-dimension limit for HNSW "
                "indexes; no vector index is created on %s and recall "
                "scans the table (exact, but slow when large). Use an "
                "embedder with at most %d dimensions (e.g. "
                "OpenAIEmbedder('text-embedding-3-large', "
                "dimensions=1536)) to get an index.",
                self.embedding_dimensions,
                HNSW_MAX_DIMENSIONS,
                self._episodes_table,
                HNSW_MAX_DIMENSIONS,
            )
        async with self._pool.acquire() as conn:
            await self._refuse_foreign_tables(conn)
            for stmt in self.schema_sql():
                await conn.execute(stmt)
        if self.facts is not None and hasattr(self.facts, "init_schema"):
            await self.facts.init_schema()

    async def _refuse_foreign_tables(self, conn: Any) -> None:
        """Raise if ``episodes`` / ``memory_blocks`` exist with a shape
        that isn't ours — the DDL that follows would ``ALTER`` them."""
        for bare, required in (
            ("episodes", _EPISODES_COLUMNS),
            ("memory_blocks", _BLOCKS_COLUMNS),
        ):
            name = self._table_prefix + bare
            try:
                rows = await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema::text = "
                    "COALESCE($1::text, current_schema()::text) "
                    "AND table_name::text = $2::text",
                    self._schema,
                    name,
                )
                columns = {str(row["column_name"]) for row in rows}
            except (KeyError, TypeError, IndexError, AttributeError):
                continue  # can't introspect (a stand-in connection)
            if columns and not required <= columns:
                missing = ", ".join(sorted(required - columns))
                raise ConfigError(
                    f"PostgresMemory: a table named {name!r} already "
                    "exists in this database but is not a loomflow "
                    f"memory table (it has no column(s): {missing}). "
                    "Refusing to alter it. Give loomflow its own tables "
                    "with PostgresMemory(..., table_prefix='loomflow_') "
                    "or schema='loomflow' (also accepted by "
                    "PostgresMemory.connect)."
                )

    # ---- working blocks --------------------------------------------------

    async def working(
        self, *, user_id: str | None = None
    ) -> list[MemoryBlock]:
        # Encode None → sentinel for the wire (Postgres PK can't
        # carry NULL); the row's user_id never appears in the
        # returned MemoryBlock so no decode needed here.
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name, content, updated_at, pinned_order "
                f"FROM {self._blocks_table} WHERE namespace = $1 "
                "AND user_id = $2 "
                "ORDER BY pinned_order ASC",
                self._namespace,
                _encode_user_id(user_id),
            )
        return [
            MemoryBlock(
                name=r["name"],
                content=r["content"],
                updated_at=r["updated_at"],
                pinned_order=r["pinned_order"],
            )
            for r in _rows(rows)
        ]

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {self._blocks_table}"
                "(namespace, user_id, name, content) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (namespace, user_id, name) DO UPDATE "
                "SET content = EXCLUDED.content, updated_at = NOW();",
                self._namespace,
                _encode_user_id(user_id),
                name,
                content,
            )

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        # Oldest-first trimming in SQL: RIGHT(text, n) keeps the
        # newest n characters of the appended block.
        if self._max_block_chars is None:
            new_content = "$4"
            merged = "existing.content || EXCLUDED.content"
            args: tuple[Any, ...] = ()
        else:
            new_content = "RIGHT($4, $5)"
            merged = "RIGHT(existing.content || EXCLUDED.content, $5)"
            args = (self._max_block_chars,)
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {self._blocks_table} AS existing"
                "(namespace, user_id, name, content) "
                f"VALUES ($1, $2, $3, {new_content}) "
                "ON CONFLICT (namespace, user_id, name) DO UPDATE "
                f"SET content = {merged}, "
                "    updated_at = NOW();",
                self._namespace,
                _encode_user_id(user_id),
                name,
                content,
                *args,
            )

    # ---- episodes --------------------------------------------------------

    async def remember(self, episode: Episode) -> str:
        if episode.embedding is None:
            text = "\n".join(p for p in (episode.input, episode.output) if p)
            # Embed before touching the pool — embedders may make
            # network calls and shouldn't hold a connection hostage.
            embedding = await self._embedder.embed(text)
            episode = episode.model_copy(update={"embedding": embedding})

        async with self._pool.acquire() as conn:
            # ``DO UPDATE`` (not ``DO NOTHING``) so re-``remember()``
            # with the same id refreshes the row — matching SQLite's
            # ``INSERT OR REPLACE`` semantics and staying consistent
            # with the transcript sidecar below, which already
            # replaces (DELETE + INSERT) on re-remember.
            insert_episode = (
                f"INSERT INTO {self._episodes_table}"
                "(id, namespace, session_id, user_id, "
                " occurred_at, input, output, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (id) DO UPDATE SET "
                "  session_id = EXCLUDED.session_id, "
                "  user_id = EXCLUDED.user_id, "
                "  occurred_at = EXCLUDED.occurred_at, "
                "  input = EXCLUDED.input, "
                "  output = EXCLUDED.output, "
                "  embedding = EXCLUDED.embedding;"
            )
            episode_args = (
                episode.id,
                self._namespace,
                episode.session_id,
                episode.user_id,
                episode.occurred_at,
                episode.input,
                episode.output,
                episode.embedding,
            )
            # Tool transcript sidecar — DELETE then INSERT to
            # stay consistent if remember() runs twice with the
            # same id. Skipped entirely when ``tool_transcript``
            # is None (default), so users without
            # ``Agent(persist_tool_transcripts=True)`` pay no
            # extra round-trip — the single INSERT is atomic on
            # its own. The multi-statement sidecar path runs in
            # one explicit transaction so a cancellation (e.g.
            # ``Agent(timeout=)`` firing mid-write) rolls back
            # cleanly instead of leaving an episode row without
            # its transcript (or a deleted transcript with no
            # replacement).
            if episode.tool_transcript is None:
                await conn.execute(insert_episode, *episode_args)
            else:
                async with conn.transaction():
                    await conn.execute(insert_episode, *episode_args)
                    await conn.execute(
                        f"DELETE FROM {self._transcripts_table} "
                        "WHERE episode_id = $1;",
                        episode.id,
                    )
                    if episode.tool_transcript:
                        rows = [
                            (episode.id, i, msg.model_dump_json())
                            for i, msg in enumerate(
                                episode.tool_transcript
                            )
                        ]
                        await conn.executemany(
                            f"INSERT INTO {self._transcripts_table} "
                            "(episode_id, sequence, message_json) "
                            "VALUES ($1, $2, $3);",
                            rows,
                        )
        return episode.id

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
    ) -> list[Episode]:
        validate_limit(limit)
        if not query.strip():
            return await self._recall_recent(limit, time_range, user_id)

        query_embedding = await self._embedder.embed(query)
        lo = time_range[0] if time_range is not None else None
        hi = time_range[1] if time_range is not None else None

        # Hard namespace partition by ``user_id``: NULL filter matches
        # only NULL rows; a string filter matches exactly. ``IS NOT
        # DISTINCT FROM`` makes ``NULL = NULL`` work correctly.
        # The namespace / user / time filter applies to an HNSW scan:
        # ``hnsw_scan`` keeps it from returning fewer rows than exist.
        async with self._pool.acquire() as conn, hnsw_scan(conn, fetch=limit, filtered=True):
            rows = await conn.fetch(
                "SELECT id, session_id, user_id, occurred_at, input, output, embedding "
                f"FROM {self._episodes_table} "
                "WHERE namespace = $1 "
                "  AND user_id IS NOT DISTINCT FROM $2 "
                "  AND ($3::timestamptz IS NULL OR occurred_at >= $3) "
                "  AND ($4::timestamptz IS NULL OR occurred_at <= $4) "
                "ORDER BY embedding <=> $5 "
                "LIMIT $6",
                self._namespace,
                user_id,
                lo,
                hi,
                query_embedding,
                limit,
            )
        return _rows_to_episodes(_rows(rows))

    async def recall_scored(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
        alpha: float = 0.5,
    ) -> list[EpisodeMatch]:
        """Native hybrid recall: BM25 lexical + cosine vector, fused
        via Reciprocal Rank Fusion. Mirrors
        :meth:`VectorMemory.recall_scored`.

        We over-fetch a candidate pool via the pgvector HNSW ANN
        (``max(limit * 8, 40)`` rows ordered by ``embedding <=>
        query``) — wide enough that the lexical (BM25) arm has real
        candidates to re-rank even when they aren't the nearest
        vectors — then compute cosine and BM25 in process so both
        component scores ride along on each :class:`EpisodeMatch`.
        Empty queries fall through to recency with neutral ``1.0``
        scores; a no-match query falls through to recency with
        ``0.0``.
        """
        from ._hybrid import hybrid_rank_episodes

        validate_limit(limit)
        if not query.strip():
            recent = await self._recall_recent(limit, time_range, user_id)
            return [EpisodeMatch(episode=e, score=1.0) for e in recent]

        query_embedding = await self._embedder.embed(query)
        lo = time_range[0] if time_range is not None else None
        hi = time_range[1] if time_range is not None else None
        fetch_limit = max(limit * 8, 40)

        async with self._pool.acquire() as conn, hnsw_scan(
            conn, fetch=fetch_limit, filtered=True
        ):
            rows = await conn.fetch(
                "SELECT id, session_id, user_id, occurred_at, input, output, embedding "
                f"FROM {self._episodes_table} "
                "WHERE namespace = $1 "
                "  AND user_id IS NOT DISTINCT FROM $2 "
                "  AND ($3::timestamptz IS NULL OR occurred_at >= $3) "
                "  AND ($4::timestamptz IS NULL OR occurred_at <= $4) "
                "ORDER BY embedding <=> $5 "
                "LIMIT $6",
                self._namespace,
                user_id,
                lo,
                hi,
                query_embedding,
                fetch_limit,
            )
        candidates = _rows_to_episodes(_rows(rows))
        candidates = [e for e in candidates if e.embedding is not None]
        if not candidates:
            return []

        return hybrid_rank_episodes(
            candidates,
            query=query,
            query_embedding=query_embedding,
            alpha=alpha,
            limit=limit,
        )

    async def _recall_recent(
        self,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        lo = time_range[0] if time_range is not None else None
        hi = time_range[1] if time_range is not None else None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, session_id, user_id, occurred_at, input, output, embedding "
                f"FROM {self._episodes_table} "
                "WHERE namespace = $1 "
                "  AND user_id IS NOT DISTINCT FROM $2 "
                "  AND ($3::timestamptz IS NULL OR occurred_at >= $3) "
                "  AND ($4::timestamptz IS NULL OR occurred_at <= $4) "
                "ORDER BY occurred_at DESC "
                "LIMIT $5",
                self._namespace,
                user_id,
                lo,
                hi,
                limit,
            )
        return _rows_to_episodes(_rows(rows))

    async def recall_facts(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
        user_id: str | None = None,
    ) -> list[Fact]:
        if self.facts is None:
            return []
        return list(
            await self.facts.recall_text(
                query, limit=limit, valid_at=valid_at, user_id=user_id
            )
        )

    async def session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        max_episodes = max(1, limit // 2)
        async with self._pool.acquire() as conn:
            # Pull the most-recent ``max_episodes`` for the session,
            # then return them oldest-first below.
            rows = await conn.fetch(
                "SELECT id, session_id, user_id, occurred_at, "
                "       input, output, embedding "
                f"FROM {self._episodes_table} "
                "WHERE namespace = $1 "
                "  AND session_id = $2 "
                "  AND user_id IS NOT DISTINCT FROM $3 "
                "ORDER BY occurred_at DESC "
                "LIMIT $4",
                self._namespace,
                session_id,
                user_id,
                max_episodes,
            )
            episodes = _rows_to_episodes(_rows(rows))
            episodes.sort(key=lambda e: e.occurred_at)
            # Bulk-fetch tool transcripts in one round-trip
            # (asyncpg supports ANY() with a list arg). Returns
            # an empty dict when no episodes have transcripts —
            # the typical case for users without
            # ``persist_tool_transcripts=True``.
            transcripts: dict[str, list[Message]] = {}
            if episodes:
                ep_ids = [ep.id for ep in episodes]
                ts_rows = await conn.fetch(
                    "SELECT episode_id, sequence, message_json "
                    f"FROM {self._transcripts_table} "
                    "WHERE episode_id = ANY($1::TEXT[]) "
                    "ORDER BY episode_id, sequence",
                    ep_ids,
                )
                for ts_row in ts_rows:
                    ep_id = ts_row["episode_id"]
                    transcripts.setdefault(ep_id, []).append(
                        Message.model_validate_json(
                            ts_row["message_json"]
                        )
                    )
        out: list[Message] = []
        for ep in episodes:
            if ep.input:
                out.append(Message(role=Role.USER, content=ep.input))
            # Splice transcript between USER and ASSISTANT so a
            # resumed worker sees its prior tool work.
            ep_transcript = transcripts.get(ep.id, [])
            if ep_transcript:
                out.extend(ep_transcript)
            if ep.output:
                out.append(Message(role=Role.ASSISTANT, content=ep.output))
        return out

    # ---- profile / forget / export (GDPR) -------------------------------

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        async with self._pool.acquire() as conn:
            count_row = await conn.fetchrow(
                "SELECT COUNT(*) AS c, MAX(occurred_at) AS last "
                f"FROM {self._episodes_table} WHERE namespace = $1 "
                "AND user_id IS NOT DISTINCT FROM $2",
                self._namespace,
                user_id,
            )
            sessions_rows = await conn.fetch(
                "SELECT DISTINCT session_id, MAX(occurred_at) AS m "
                f"FROM {self._episodes_table} WHERE namespace = $1 "
                "AND user_id IS NOT DISTINCT FROM $2 "
                "GROUP BY session_id ORDER BY m DESC LIMIT 10",
                self._namespace,
                user_id,
            )
        episode_count = int(count_row["c"]) if count_row else 0
        last_seen = count_row["last"] if count_row else None
        recent_sessions = [r["session_id"] for r in sessions_rows]

        sample_facts: list[Fact] = []
        fact_count = 0
        if self.facts is not None:
            sample_facts = list(
                await self.facts.query(user_id=user_id, limit=10)
            )
            fact_count = await count_facts(self.facts, user_id=user_id)

        return MemoryProfile(
            user_id=user_id,
            episode_count=episode_count,
            fact_count=fact_count,
            last_seen=last_seen,
            recent_sessions=recent_sessions,
            sample_facts=sample_facts,
        )

    async def forget(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
    ) -> int:
        # A background extraction of an erased episode would write
        # its facts back after this returns.
        await _extractions.stop_extractions(
            self, user_id=user_id, session_id=session_id, before=before
        )
        # Build episode-delete clause; ``IS NOT DISTINCT FROM`` makes
        # NULL=NULL behave as expected for the anonymous bucket.
        clauses = ["namespace = $1", "user_id IS NOT DISTINCT FROM $2"]
        params: list[Any] = [self._namespace, user_id]
        idx = 3
        if session_id is not None:
            clauses.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1
        if before is not None:
            clauses.append(f"occurred_at < ${idx}")
            params.append(before)
            idx += 1
        sql = f"DELETE FROM {self._episodes_table} WHERE " + " AND ".join(clauses)
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params)
            # asyncpg returns "DELETE <n>" — parse the count.
            deleted = _parse_delete_count(result)
            if session_id is None:
                # Working blocks are per user too (no session).
                block_sql = (
                    f"DELETE FROM {self._blocks_table} "
                    "WHERE namespace = $1 AND user_id = $2"
                )
                block_params: list[Any] = [
                    self._namespace,
                    _encode_user_id(user_id),
                ]
                if before is not None:
                    block_sql += " AND updated_at < $3"
                    block_params.append(before)
                deleted += _parse_delete_count(
                    await conn.execute(block_sql, *block_params)
                )

        # Facts are user-scoped, not session-scoped — erase them via
        # the FactStore's own delete surface (not raw SQL against its
        # table) unless the caller narrowed by session_id.
        if session_id is None and self.facts is not None:
            deleted += await delete_facts(
                self.facts, user_id=user_id, before=before
            )
        return deleted

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, session_id, user_id, occurred_at, input, output, embedding "
                f"FROM {self._episodes_table} WHERE namespace = $1 "
                "AND user_id IS NOT DISTINCT FROM $2 "
                "ORDER BY occurred_at ASC",
                self._namespace,
                user_id,
            )
        episodes = _rows_to_episodes(_rows(rows))
        facts: list[Fact] = []
        if self.facts is not None:
            facts = list(
                await self.facts.query(user_id=user_id, limit=100_000)
            )
        return MemoryExport(
            user_id=user_id,
            episodes=episodes,
            facts=sorted(facts, key=lambda f: f.recorded_at),
            blocks=await self.working(user_id=user_id),
        )

    async def consolidate(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows(result: Iterable[Any]) -> list[Any]:
    return list(result) if not isinstance(result, list) else result


def _parse_delete_count(asyncpg_result: str) -> int:
    """asyncpg's ``Connection.execute`` returns a status string like
    ``"DELETE 17"`` for DML; parse the count off the end. Returns
    ``0`` when the status doesn't carry a number."""
    parts = asyncpg_result.strip().split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def _rows_to_episodes(rows: list[Any]) -> list[Episode]:
    out: list[Episode] = []
    for r in rows:
        # pgvector returns a Vector / numpy array / text; coerce.
        emb_list = vector_to_list(r["embedding"])
        # ``user_id`` may not be present on rows fetched from a legacy
        # SELECT that doesn't list it. Treat absence as ``None``.
        try:
            user_id_val = r["user_id"]
        except (KeyError, IndexError):
            user_id_val = None
        out.append(
            Episode(
                id=r["id"],
                session_id=r["session_id"],
                user_id=user_id_val,
                occurred_at=r["occurred_at"],
                input=r["input"],
                output=r["output"],
                embedding=emb_list,
            )
        )
    return out
