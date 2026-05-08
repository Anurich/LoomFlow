"""Postgres + pgvector :class:`Memory` backend.

Schema (created by :meth:`init_schema`):

* ``memory_blocks(namespace, name, content, pinned_order, updated_at)``
* ``episodes(id, namespace, session_id, occurred_at, input, output,
  embedding vector(N))`` with HNSW cosine index on ``embedding``

The ``vector(N)`` column dimension is fixed at table-creation time and
must match the configured embedder's ``dimensions``. Switching
embedders later requires migrating the table.

Both ``asyncpg`` and ``pgvector`` are imported lazily inside
:meth:`connect` / :meth:`init_schema` so the module loads in
environments without those extras installed; the import only fires
when actually opening a connection.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import anyio

from ..core.protocols import Embedder
from ..core.types import Episode, Fact, MemoryBlock, Message, Role
from .embedder import HashEmbedder

DEFAULT_NAMESPACE = "default"


class PostgresMemory:
    """Postgres-backed :class:`Memory`.

    ``pool`` is an ``asyncpg.Pool`` (or anything with the same API).
    Tests can pass a fake pool whose ``acquire()`` returns a fake
    connection.
    """

    def __init__(
        self,
        pool: Any,
        *,
        embedder: Embedder | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        fact_store: Any | None = None,
    ) -> None:
        self._pool = pool
        self._embedder: Embedder = embedder if embedder is not None else HashEmbedder()
        self._namespace = namespace
        # ``facts`` is left as ``None`` by default to avoid forcing the
        # facts schema on users who don't want it. Pass an explicit
        # :class:`PostgresFactStore` (or set ``with_facts=True`` on
        # :meth:`connect`) to enable it.
        self.facts: Any | None = fact_store

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
    ) -> PostgresMemory:
        """Open an asyncpg pool and register the pgvector codec.

        When ``with_facts=True`` a :class:`PostgresFactStore` rooted at
        the same pool is attached as ``self.facts`` so the agent loop's
        memory.facts integration just works.
        """
        try:
            import asyncpg  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "asyncpg is not installed. "
                "Install with: pip install 'jeevesagent[postgres]'"
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

        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            init=_setup,
        )
        instance = cls(pool, embedder=embedder, namespace=namespace)
        if with_facts:
            from .postgres_facts import PostgresFactStore

            instance.facts = PostgresFactStore(pool, embedder=embedder)
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
        return [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            (
                "CREATE TABLE IF NOT EXISTS memory_blocks ("
                "  namespace TEXT NOT NULL,"
                "  name TEXT NOT NULL,"
                "  content TEXT NOT NULL,"
                "  pinned_order INT NOT NULL DEFAULT 0,"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
                "  PRIMARY KEY (namespace, name)"
                ");"
            ),
            (
                f"CREATE TABLE IF NOT EXISTS episodes ("
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
            "ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id TEXT;",
            (
                "CREATE INDEX IF NOT EXISTS episodes_namespace_idx "
                "ON episodes (namespace, occurred_at DESC);"
            ),
            (
                "CREATE INDEX IF NOT EXISTS episodes_user_id_idx "
                "ON episodes (namespace, user_id, occurred_at DESC);"
            ),
            (
                "CREATE INDEX IF NOT EXISTS episodes_embedding_idx "
                "ON episodes USING hnsw (embedding vector_cosine_ops);"
            ),
        ]

    async def init_schema(self) -> None:
        """Apply :meth:`schema_sql` against the connected pool.

        When a :class:`PostgresFactStore` is attached as ``self.facts``,
        its schema is initialised in the same call.
        """
        async with self._pool.acquire() as conn:
            for stmt in self.schema_sql():
                await conn.execute(stmt)
        if self.facts is not None and hasattr(self.facts, "init_schema"):
            await self.facts.init_schema()

    # ---- working blocks --------------------------------------------------

    async def working(self) -> list[MemoryBlock]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name, content, updated_at, pinned_order "
                "FROM memory_blocks WHERE namespace = $1 "
                "ORDER BY pinned_order ASC",
                self._namespace,
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

    async def update_block(self, name: str, content: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_blocks(namespace, name, content) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (namespace, name) DO UPDATE "
                "SET content = EXCLUDED.content, updated_at = NOW();",
                self._namespace,
                name,
                content,
            )

    async def append_block(self, name: str, content: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_blocks(namespace, name, content) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (namespace, name) DO UPDATE "
                "SET content = memory_blocks.content || EXCLUDED.content, "
                "    updated_at = NOW();",
                self._namespace,
                name,
                content,
            )

    # ---- episodes --------------------------------------------------------

    async def remember(self, episode: Episode) -> str:
        if episode.embedding is None:
            text = "\n".join(p for p in (episode.input, episode.output) if p)

            # Embed in parallel with the connection acquire to amortise
            # latency. (No-op for HashEmbedder; meaningful for OpenAI.)
            async with anyio.create_task_group() as tg:
                holder: list[list[float] | None] = [None]

                async def _do_embed() -> None:
                    holder[0] = await self._embedder.embed(text)

                tg.start_soon(_do_embed)
            assert holder[0] is not None
            episode = episode.model_copy(update={"embedding": holder[0]})

        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO episodes(id, namespace, session_id, user_id, "
                "                     occurred_at, input, output, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (id) DO NOTHING;",
                episode.id,
                self._namespace,
                episode.session_id,
                episode.user_id,
                episode.occurred_at,
                episode.input,
                episode.output,
                episode.embedding,
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
        if not query.strip():
            return await self._recall_recent(limit, time_range, user_id)

        query_embedding = await self._embedder.embed(query)
        lo = time_range[0] if time_range is not None else None
        hi = time_range[1] if time_range is not None else None

        # Hard namespace partition by ``user_id``: NULL filter matches
        # only NULL rows; a string filter matches exactly. ``IS NOT
        # DISTINCT FROM`` makes ``NULL = NULL`` work correctly.
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, session_id, user_id, occurred_at, input, output, embedding "
                "FROM episodes "
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
                "FROM episodes "
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
                "FROM episodes "
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
        out: list[Message] = []
        for ep in episodes:
            if ep.input:
                out.append(Message(role=Role.USER, content=ep.input))
            if ep.output:
                out.append(Message(role=Role.ASSISTANT, content=ep.output))
        return out

    async def consolidate(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows(result: Iterable[Any]) -> list[Any]:
    return list(result) if not isinstance(result, list) else result


def _rows_to_episodes(rows: list[Any]) -> list[Episode]:
    out: list[Episode] = []
    for r in rows:
        emb = r["embedding"]
        # asyncpg + pgvector returns numpy array or list; coerce.
        emb_list = list(emb) if emb is not None else None
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
