"""Postgres + pgvector vector store.

Production durable storage. Lazy import via ``asyncpg``; install
with ``pip install 'loomflow[vectorstore-postgres]'`` and ensure
the ``vector`` extension is enabled on your database
(``CREATE EXTENSION IF NOT EXISTS vector``).

Schema (auto-created via :meth:`init_schema`)::

    CREATE TABLE jeeves_vectors (
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        metadata    JSONB,
        embedding   vector(N) NOT NULL,
        content_tsv tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', content)) STORED
    );
    CREATE INDEX ON jeeves_vectors USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX ON jeeves_vectors USING gin (content_tsv);

``content_tsv`` powers :meth:`search_hybrid` (full-text lexical
ranking fused with the vector ranking via RRF). Tables created by
older loomflow versions upgrade in place — ``init_schema`` adds the
column and GIN index idempotently.

Filter language: full Mongo-style operators translated to JSONB
SQL. ``$eq`` / ``$ne`` / ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` /
``$in`` / ``$nin`` / ``$and`` / ``$or`` / ``$not`` / ``$exists``
are all supported.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

import anyio

from ..core.errors import ConfigError
from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._bm25 import fuse_weighted
from ._filter import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from ._mmr import rerank_tail
from ._pgvector import (
    HNSW_MAX_DIMENSIONS,
    hnsw_scan,
    index_name,
    qualified_name,
    split_table_name,
)
from ._util import (
    dedupe_last,
    embed_all,
    embedder_identity,
    resolve_ids,
    validate_k,
    validate_query_vector,
)
from .base import SearchResult, _chunks_from_texts

logger = logging.getLogger(__name__)

# Ordering operators → SQL. They act on the JSONB value extracted as
# text (``->>``), cast to numeric or compared as text depending on the
# operand's type (see ``_xlate_field``).
_SQL_BIN_OPS: dict[str, str] = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}

# Text-search config names are interpolated as SQL literals (generated
# columns can't take bind parameters), so allow only bare regconfig
# identifiers — same conservative posture as ``_safe_key``.
_FTS_LANG_RE = re.compile(r"[a-z_]+\Z")


class PostgresVectorStore:
    """Vector store backed by Postgres + ``pgvector``."""

    name = "postgres"

    def __init__(
        self,
        embedder: Embedder,
        *,
        dsn: str,
        table: str = "jeeves_vectors",
        dimension: int | None = None,
        pool_size: int = 10,
        fts_language: str = "english",
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        # The language is interpolated as a SQL literal (a GENERATED
        # column expression can't take a bind parameter), so it must
        # be a bare regconfig name — 'english', 'simple', 'german', …
        if not _FTS_LANG_RE.fullmatch(fts_language):
            raise ValueError(
                f"invalid fts_language: {fts_language!r} (expected a "
                "bare Postgres text-search config name like 'english' "
                "or 'simple')"
            )
        # The table name is interpolated into every statement, so it is
        # validated here and only ever emitted quoted.
        self._schema, self._table_name = split_table_name(table)
        self._table = qualified_name(self._schema, self._table_name)
        if dimension is not None and (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError(
                f"dimension must be a positive integer, got {dimension!r}"
            )
        _, embedder_dims = embedder_identity(embedder)
        if (
            dimension is not None
            and embedder_dims is not None
            and dimension != embedder_dims
        ):
            raise ValueError(
                f"dimension={dimension} does not match the embedder "
                f"({getattr(embedder, 'name', type(embedder).__name__)!r} "
                f"produces {embedder_dims}-dimensional vectors). Drop "
                "dimension= to use the embedder's, or pass an embedder "
                "with that output size."
            )
        self._embedder = embedder
        self._dsn = dsn
        self._dimension = dimension
        self._initialized = False
        self._pool_size = pool_size
        self._fts_language = fts_language
        self._pool_obj: Any = None
        self._pool_lock = anyio.Lock()

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # ---------------------------------------------------------------
    # Factory classmethods — explicit kwargs so IDEs autocomplete
    # ---------------------------------------------------------------

    @classmethod
    async def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        embedder: Embedder,
        ids: list[str] | None = None,
        dsn: str,
        table: str = "jeeves_vectors",
        dimension: int | None = None,
        fts_language: str = "english",
    ) -> PostgresVectorStore:
        """One-shot: construct a PostgresVectorStore + add ``chunks``.

        FACTORY — builds and returns a NEW store. To add to an
        EXISTING store call ``store.add(chunks)`` (or
        ``index_document(path, store)``); calling this in a loop
        creates throwaway stores and drops writes.
        """
        store = cls(
            embedder=embedder,
            dsn=dsn,
            table=table,
            dimension=dimension,
            fts_language=fts_language,
        )
        await store.add(chunks, ids=ids)
        return store

    @classmethod
    async def from_texts(
        cls,
        texts: list[str],
        *,
        embedder: Embedder,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        dsn: str,
        table: str = "jeeves_vectors",
        dimension: int | None = None,
        fts_language: str = "english",
    ) -> PostgresVectorStore:
        """One-shot: construct a PostgresVectorStore from raw text
        strings (each becomes a :class:`Chunk` with the matching
        metadata dict, or empty if ``metadatas`` is None)."""
        return await cls.from_chunks(
            _chunks_from_texts(texts, metadatas),
            embedder=embedder,
            ids=ids,
            dsn=dsn,
            table=table,
            dimension=dimension,
            fts_language=fts_language,
        )

    async def _pool(self) -> Any:
        """Lazily create + cache a connection POOL, then reuse it.

        Every op previously opened a brand-new asyncpg connection and
        closed it — 10 searches meant 10 TCP+auth handshakes, an
        order-of-magnitude slower than the in-process stores and a
        silent scaling footgun. A pooled store amortises that to ~one
        connection's cost. Created once on first use (under a lock so
        concurrent first calls don't each build a pool); ``aclose()``
        tears it down. A bad DSN now surfaces HERE, on the first
        operation, with a clear asyncpg error rather than per-call.
        """
        if self._pool_obj is not None:
            return self._pool_obj
        try:
            import asyncpg  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "asyncpg is not installed. "
                "Install with: pip install "
                "'loomflow[vectorstore-postgres]'."
            ) from exc
        async with self._pool_lock:
            if self._pool_obj is None:
                self._pool_obj = await asyncpg.create_pool(
                    self._dsn, min_size=1, max_size=self._pool_size
                )
        return self._pool_obj

    def _acquire(self) -> Any:
        """``async with self._acquire() as conn:`` — borrow a pooled
        connection (returned to the pool on exit, never closed)."""

        store = self

        class _Acquire:
            async def __aenter__(self) -> Any:
                self._pool = await store._pool()
                self._conn = await self._pool.acquire()
                return self._conn

            async def __aexit__(self, *exc: Any) -> None:
                await self._pool.release(self._conn)

        return _Acquire()

    async def aclose(self) -> None:
        """Close the connection pool. Call on shutdown; idempotent."""
        if self._pool_obj is not None:
            await self._pool_obj.close()
            self._pool_obj = None

    async def init_schema(self, dimension: int) -> None:
        """Create the table + HNSW and GIN indexes. Idempotent.

        Also upgrades tables created by older loomflow versions in
        place: ``ADD COLUMN IF NOT EXISTS content_tsv`` back-fills
        the full-text column (generated, so Postgres computes it for
        existing rows) that :meth:`search_hybrid` ranks on.

        pgvector can only HNSW-index a ``vector`` column of up to
        2000 dimensions. Above that (e.g. ``text-embedding-3-large``
        at 3072) the index is skipped with a warning: searches stay
        correct — an exact scan — but slow on large tables.
        """
        self._check_dimension(dimension)
        self._dimension = dimension
        tsv_expr = (
            f"tsvector GENERATED ALWAYS AS "
            f"(to_tsvector('{self._fts_language}', content)) STORED"
        )
        async with self._acquire() as conn:
            await conn.execute(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({int(dimension)}) NOT NULL,
                    content_tsv {tsv_expr}
                )
                """
            )
            # Pre-0.13 tables lack the column; the generated
            # expression keeps the fts_language the column was FIRST
            # created with (changing the kwarg later doesn't rewrite
            # an existing column — drop it manually to re-language).
            await conn.execute(
                f"""
                ALTER TABLE {self._table}
                ADD COLUMN IF NOT EXISTS content_tsv {tsv_expr}
                """
            )
            if dimension <= HNSW_MAX_DIMENSIONS:
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS
                        {index_name(self._table_name, "embedding_hnsw")}
                    ON {self._table}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            else:
                logger.warning(
                    "PostgresVectorStore: %s holds %d-dimensional vectors, "
                    "above pgvector's %d-dimension limit for HNSW indexes; "
                    "no vector index was created and searches scan the "
                    "whole table (exact, but slow on large tables). Use an "
                    "embedder with at most %d dimensions (e.g. "
                    "OpenAIEmbedder('text-embedding-3-large', "
                    "dimensions=1536)) to get an index.",
                    self._table,
                    dimension,
                    HNSW_MAX_DIMENSIONS,
                    HNSW_MAX_DIMENSIONS,
                )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {index_name(self._table_name, "content_tsv_gin")}
                ON {self._table}
                USING gin (content_tsv)
                """
            )
            self._initialized = True

    def _check_dimension(self, actual: int) -> None:
        """The constructor's ``dimension=`` is a promise about the
        embedder's output; a vector of another size breaks it."""
        if self._dimension is not None and actual != self._dimension:
            raise ValueError(
                f"embedder produced {actual}-dimensional vectors but "
                f"the store was configured for {self._dimension} "
                "(dimension=). Use an embedder with that output size, "
                "or a new table for the new embedder."
            )

    async def add(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks:
            return []
        assigned = resolve_ids(ids, len(chunks))
        # An id repeated in one call upserts onto itself: last wins.
        unique_ids, unique_chunks = dedupe_last(assigned, chunks)
        # Serialise metadata BEFORE embedding: a failure here shouldn't
        # cost an embedding call. JSONB round-trips lists/dicts
        # natively (so a MarkdownChunker ``headers`` list comes back a
        # list, unlike Chroma's scalar-only store). ``default=str`` is
        # the safety net so an exotic value can never crash the insert.
        metadatas = [
            json.dumps(c.metadata or {}, default=str) for c in unique_chunks
        ]
        vectors = await embed_all(
            self._embedder, [c.content for c in unique_chunks]
        )
        for vec in vectors:
            validate_query_vector(vec)
            self._check_dimension(len(vec))

        if not self._initialized:
            await self.init_schema(len(vectors[0]))

        rows = [
            (
                unique_ids[i],
                unique_chunks[i].content,
                metadatas[i],
                _vec_to_pg(vectors[i]),
            )
            for i in range(len(unique_chunks))
        ]
        async with self._acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._table} (id, content, metadata, embedding)
                VALUES ($1, $2, $3::jsonb, $4::vector)
                ON CONFLICT (id) DO UPDATE
                  SET content   = EXCLUDED.content,
                      metadata  = EXCLUDED.metadata,
                      embedding = EXCLUDED.embedding
                """,
                rows,
            )
        return assigned

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1::text[])",
                list(ids),
            )

    async def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        async with self._acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata
                FROM {self._table}
                WHERE id = ANY($1::text[])
                """,
                list(ids),
            )
        by_id: dict[str, Chunk] = {}
        for row in rows:
            md = row["metadata"]
            metadata = (
                json.loads(md) if isinstance(md, str) else (md or {})
            )
            by_id[row["id"]] = Chunk(
                content=row["content"], metadata=metadata
            )
        return [by_id[cid] for cid in ids if cid in by_id]

    async def search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
        validate_k(k)
        q_vec = await self._embedder.embed(query)
        return await self.search_by_vector(
            q_vec, k=k, filter=filter, diversity=diversity
        )

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
        validate_k(k)
        validate_query_vector(vector, dimensions=self._dimension)
        params: list[Any] = [_vec_to_pg(vector)]
        where_sql = ""
        if filter:
            where_sql, params = _build_where_sql(filter, params)

        # MMR rerank needs the raw vectors; a plain top-k search
        # doesn't — skip fetching the (wide) embedding column then.
        want_mmr = diversity is not None and diversity > 0
        # Wider candidate pool when MMR-reranking.
        n_fetch = max(k * 4, 20) if want_mmr else k
        params.append(n_fetch)
        embedding_col = "embedding," if want_mmr else ""

        sql = f"""
            SELECT id, content, metadata, {embedding_col}
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT ${len(params)}
        """

        async with self._acquire() as conn, hnsw_scan(
            conn, fetch=n_fetch, filtered=bool(filter)
        ):
            rows = await conn.fetch(sql, *params)

        candidates: list[SearchResult] = []
        cand_vecs: list[list[float]] = []
        for row in rows:
            md = row["metadata"]
            metadata = (
                json.loads(md) if isinstance(md, str) else (md or {})
            )
            candidates.append(
                SearchResult(
                    chunk=Chunk(
                        content=row["content"],
                        metadata=metadata,
                    ),
                    score=float(row["score"]),
                    id=row["id"],
                )
            )
            if want_mmr:
                # pgvector returns embedding as text "[1.0,2.0,...]"
                cand_vecs.append(_pg_to_vec(row["embedding"]))

        return rerank_tail(vector, candidates, cand_vecs, k, diversity)

    async def search_hybrid(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Hybrid lexical + vector search via RRF — same contract as
        :meth:`InMemoryVectorStore.search_hybrid`.

        ``alpha`` is in [0, 1]: 0 = pure lexical, 1 = pure vector,
        0.5 = even weighting. Two indexed queries run — HNSW top-N
        by cosine, and Postgres full-text search top-N ranked by
        ``ts_rank_cd`` over the stored GIN-indexed ``content_tsv``
        column — and their rankings are fused by weighted Reciprocal
        Rank Fusion.

        Because the lexical leg is real FTS over the WHOLE corpus
        (not BM25 over the vector candidates), an exact-term match —
        an error code, a model name — surfaces even when it isn't
        vector-close to the query. That's the case hybrid exists for.

        Requires the ``content_tsv`` column: tables created before
        loomflow 0.13 need one ``await store.init_schema(dimension)``
        to upgrade in place (idempotent; raises :class:`ConfigError`
        with that instruction if the column is missing).
        """
        validate_k(k)
        alpha = max(0.0, min(1.0, alpha))
        n_fetch = max(k * 4, 20)

        # --- vector ranking: HNSW top-N (skipped when alpha=0 —
        # also skips the embedding call) ---
        v_rows: list[Any] = []
        v_sql = ""
        v_params: list[Any] = []
        if alpha > 0:
            q_vec = await self._embedder.embed(query)
            validate_query_vector(q_vec, dimensions=self._dimension)
            v_params = [_vec_to_pg(q_vec)]
            v_where = ""
            if filter:
                v_where, v_params = _build_where_sql(filter, v_params)
            v_params.append(n_fetch)
            v_sql = f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM {self._table}
                {v_where}
                ORDER BY embedding <=> $1::vector
                LIMIT ${len(v_params)}
            """

        # --- lexical ranking: FTS top-N (skipped when alpha=1).
        # ``websearch_to_tsquery`` never raises on arbitrary user
        # input (unlike ``to_tsquery``); an all-stopword query just
        # matches nothing and fusion proceeds on the vector leg. ---
        l_sql = ""
        l_params: list[Any] = []
        if alpha < 1:
            l_params = []
            l_where = ""
            if filter:
                l_where, l_params = _build_where_sql(filter, l_params)
            l_params.append(self._fts_language)
            lang_ref = f"${len(l_params)}::regconfig"
            l_params.append(query)
            tsq = f"websearch_to_tsquery({lang_ref}, ${len(l_params)})"
            match_sql = f"content_tsv @@ {tsq}"
            l_where = (
                f"{l_where} AND {match_sql}"
                if l_where
                else f"WHERE {match_sql}"
            )
            l_params.append(n_fetch)
            l_sql = f"""
                SELECT id, content, metadata,
                       ts_rank_cd(content_tsv, {tsq}) AS score
                FROM {self._table}
                {l_where}
                ORDER BY score DESC
                LIMIT ${len(l_params)}
            """

        l_rows: list[Any] = []
        async with self._acquire() as conn:
            if v_sql:
                async with hnsw_scan(conn, fetch=n_fetch, filtered=bool(filter)):
                    v_rows = await conn.fetch(v_sql, *v_params)
            if l_sql:
                try:
                    l_rows = await conn.fetch(l_sql, *l_params)
                except Exception as exc:
                    # 42703 = undefined_column: a pre-0.13 table
                    # without content_tsv. Point at the one-call fix
                    # instead of leaking a bare asyncpg error.
                    if getattr(exc, "sqlstate", None) == "42703":
                        raise ConfigError(
                            f"hybrid search needs the "
                            f"{self._table}.content_tsv full-text "
                            "column (loomflow 0.13 schema). Upgrade "
                            "the table in place with: await "
                            "store.init_schema(dimension) — "
                            "idempotent; adds the generated tsvector "
                            "column + GIN index."
                        ) from exc
                    raise

        chunks_by_id: dict[str, Chunk] = {}

        def _ranking(rows: list[Any]) -> list[tuple[str, float]]:
            out: list[tuple[str, float]] = []
            for row in rows:
                md = row["metadata"]
                metadata = (
                    json.loads(md) if isinstance(md, str) else (md or {})
                )
                chunks_by_id.setdefault(
                    row["id"],
                    Chunk(content=row["content"], metadata=metadata),
                )
                out.append((row["id"], float(row["score"])))
            return out

        fused = fuse_weighted(_ranking(v_rows), _ranking(l_rows), alpha)
        return [
            SearchResult(chunk=chunks_by_id[cid], score=score, id=cid)
            for cid, score in fused[:k]
        ]

    async def count(self) -> int:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) AS n FROM {self._table}"
            )
            return int(row["n"]) if row else 0


# ---------------------------------------------------------------------------
# Filter translation: Mongo-style → JSONB SQL
# ---------------------------------------------------------------------------


def _build_where_sql(
    filter: Mapping[str, Any], params: list[Any]
) -> tuple[str, list[Any]]:
    expr, params = _xlate_node(filter, params)
    return f"WHERE {expr}", params


def _xlate_node(
    node: Mapping[str, Any], params: list[Any]
) -> tuple[str, list[Any]]:
    parts: list[str] = []
    for key, value in node.items():
        if key == "$and":
            if not isinstance(value, list):
                raise FilterError("$and expects a list of subfilters")
            sub_exprs = []
            for sub in value:
                expr, params = _xlate_node(sub, params)
                sub_exprs.append(f"({expr})")
            parts.append(" AND ".join(sub_exprs) if sub_exprs else "true")
        elif key == "$or":
            if not isinstance(value, list):
                raise FilterError("$or expects a list of subfilters")
            sub_exprs = []
            for sub in value:
                expr, params = _xlate_node(sub, params)
                sub_exprs.append(f"({expr})")
            parts.append(
                "(" + " OR ".join(sub_exprs) + ")" if sub_exprs else "false"
            )
        elif key == "$not":
            if not isinstance(value, Mapping):
                raise FilterError("$not expects a subfilter dict")
            expr, params = _xlate_node(value, params)
            # A comparison on a missing key is NULL, and NOT NULL is
            # NULL: COALESCE so "not matching" includes those rows, as
            # in every other store.
            parts.append(f"NOT COALESCE(({expr}), false)")
        elif key in LOGICAL_OPERATORS:
            raise FilterError(f"Unhandled logical operator: {key}")
        elif key.startswith("$"):
            raise FilterError(f"Unknown top-level operator: {key}")
        else:
            expr, params = _xlate_field(key, value, params)
            parts.append(expr)
    return (" AND ".join(parts) if parts else "true"), params


# Metadata keys are interpolated into the SQL text (JSONB ``->>``
# takes a literal), so they must be validated — a quote or backslash
# in a key would be an injection vector. Conservative allow-list.
_SAFE_KEY_RE = re.compile(r"[A-Za-z0-9_.\- ]+\Z")


def _safe_key(key: str) -> str:
    if not _SAFE_KEY_RE.fullmatch(key):
        raise FilterError(
            f"invalid metadata key for filtering: {key!r} "
            "(only letters, digits, '_', '.', '-' and spaces allowed)"
        )
    return key


def _is_numeric_operand(value: Any) -> bool:
    """True for int/float operands (bool excluded — it's an int
    subclass but is its own JSON type)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _jsonb_param(value: Any, params: list[Any]) -> str:
    """Bind ``value`` as a JSONB parameter; returns its placeholder."""
    params.append(json.dumps(value, default=str))
    return f"${len(params)}::jsonb"


def _eq_sql(key: str, expected: Any, params: list[Any]) -> str:
    """``$eq`` under the shared filter contract, never NULL.

    The comparison is JSONB-typed, so it is type-strict like the
    in-process stores: the number ``2020`` doesn't equal the string
    ``"2020"`` and ``1`` doesn't equal ``true``, while ``1 = 1.0``.
    ``@>`` (containment) is equality for scalars and membership when
    the stored value is an array. ``None`` matches a missing key or a
    JSON null — ``->>`` is NULL for both.
    """
    if expected is None:
        return f"((metadata->>'{key}') IS NULL)"
    placeholder = _jsonb_param(expected, params)
    op = "=" if isinstance(expected, list | tuple | Mapping) else "@>"
    return f"COALESCE((metadata->'{key}') {op} {placeholder}, false)"


def _in_sql(key: str, expected: Any, params: list[Any]) -> str:
    """``$in``: the stored value (or, for an array, any element of it)
    equals one of ``expected``. Never NULL."""
    values = [v for v in expected if v is not None]
    alternatives: list[str] = []
    if values:
        placeholder = _jsonb_param(values, params)
        alternatives.append(
            "EXISTS (SELECT 1 FROM jsonb_array_elements("
            f"{placeholder}) AS _in(v) "
            f"WHERE (metadata->'{key}') @> _in.v)"
        )
    if len(values) != len(expected):  # the list held a None
        alternatives.append(f"(metadata->>'{key}') IS NULL")
    if not alternatives:
        return "false"
    return "(" + " OR ".join(alternatives) + ")"


def _ordering_sql(
    key: str, op: str, expected: Any, params: list[Any]
) -> str:
    """``$gt`` / ``$gte`` / ``$lt`` / ``$lte``: number against number
    or string against string; anything else doesn't match. ``CASE``
    (unlike ``AND``) guarantees the type test runs before the cast, so
    a non-numeric value can never abort the query."""
    sql_op = _SQL_BIN_OPS[op]
    if _is_numeric_operand(expected):
        params.append(expected)
        return (
            f"(CASE WHEN jsonb_typeof(metadata->'{key}') = 'number' "
            f"THEN ((metadata->>'{key}'))::numeric {sql_op} ${len(params)} "
            "ELSE false END)"
        )
    if isinstance(expected, str):
        params.append(expected)
        # COLLATE "C" = code-point order, as Python compares strings.
        return (
            f"(CASE WHEN jsonb_typeof(metadata->'{key}') = 'string' "
            f"THEN (metadata->>'{key}') COLLATE \"C\" {sql_op} "
            f"${len(params)}::text ELSE false END)"
        )
    return "false"


def _xlate_field(
    key: str, condition: Any, params: list[Any]
) -> tuple[str, list[Any]]:
    """Translate one field constraint to a SQL boolean expression."""
    key = _safe_key(key)
    if isinstance(condition, Mapping) and condition and all(
        k.startswith("$") for k in condition
    ):
        sub_exprs: list[str] = []
        for op, expected in condition.items():
            if op == "$eq":
                sub_exprs.append(_eq_sql(key, expected, params))
            elif op == "$ne":
                # A missing key is "not equal to anything" (the shared
                # contract) — ``_eq_sql`` is never NULL, so NOT is safe.
                sub_exprs.append(f"(NOT {_eq_sql(key, expected, params)})")
            elif op in _SQL_BIN_OPS:
                sub_exprs.append(_ordering_sql(key, op, expected, params))
            elif op == "$in":
                if not isinstance(expected, list | tuple):
                    raise FilterError("$in expects a list")
                sub_exprs.append(_in_sql(key, expected, params))
            elif op == "$nin":
                if not isinstance(expected, list | tuple):
                    raise FilterError("$nin expects a list")
                sub_exprs.append(f"(NOT {_in_sql(key, expected, params)})")
            elif op == "$exists":
                if expected:
                    sub_exprs.append(
                        f"(metadata ? '{key}')"
                    )
                else:
                    sub_exprs.append(
                        f"(NOT (metadata ? '{key}'))"
                    )
            elif op not in COMPARISON_OPERATORS:
                raise FilterError(f"Unknown field operator: {op}")
        return " AND ".join(sub_exprs), params

    if isinstance(condition, list | tuple):
        return _in_sql(key, condition, params), params

    return _eq_sql(key, condition, params), params


def _vec_to_pg(vec: list[float]) -> str:
    """Serialize a Python float list to pgvector's wire format."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _pg_to_vec(s: Any) -> list[float]:
    """Inverse — parse pgvector's text representation."""
    if isinstance(s, list):
        return [float(x) for x in s]
    text = str(s).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(x) for x in text.split(",")]
