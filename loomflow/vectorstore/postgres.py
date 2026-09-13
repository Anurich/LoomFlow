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
from ._util import embed_all, resolve_ids
from .base import SearchResult, _chunks_from_texts

# Map Mongo-style ops to SQL operators that act on the JSONB extracted
# value. Note: we always extract via ``->>`` (text) and cast on demand
# so numeric comparisons work on integers stored as JSON numbers.
_SQL_BIN_OPS: dict[str, str] = {
    "$eq": "=",
    "$ne": "<>",
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
        self._embedder = embedder
        self._dsn = dsn
        self._table = table
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
        """
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
                    embedding vector({dimension}) NOT NULL,
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
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {self._table}_embedding_hnsw
                ON {self._table}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {self._table}_content_tsv_gin
                ON {self._table}
                USING gin (content_tsv)
                """
            )
            self._initialized = True

    async def add(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks:
            return []
        assigned = resolve_ids(ids, len(chunks))
        vectors = await embed_all(
            self._embedder, [c.content for c in chunks]
        )

        if not self._initialized:
            await self.init_schema(len(vectors[0]))

        rows = [
            (
                assigned[i],
                chunks[i].content,
                # JSONB round-trips lists/dicts natively (so a
                # MarkdownChunker ``headers`` list comes back a
                # list, unlike Chroma's scalar-only store).
                # ``default=str`` is the safety net so an exotic
                # value can never crash the insert.
                json.dumps(chunks[i].metadata or {}, default=str),
                _vec_to_pg(vectors[i]),
            )
            for i in range(len(chunks))
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

        async with self._acquire() as conn:
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
        alpha = max(0.0, min(1.0, alpha))
        n_fetch = max(k * 4, 20)

        # --- vector ranking: HNSW top-N (skipped when alpha=0 —
        # also skips the embedding call) ---
        v_rows: list[Any] = []
        v_sql = ""
        v_params: list[Any] = []
        if alpha > 0:
            q_vec = await self._embedder.embed(query)
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
                            f"'{self._table}.content_tsv' full-text "
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
            assert isinstance(value, list)
            sub_exprs = []
            for sub in value:
                expr, params = _xlate_node(sub, params)
                sub_exprs.append(f"({expr})")
            parts.append(" AND ".join(sub_exprs))
        elif key == "$or":
            assert isinstance(value, list)
            sub_exprs = []
            for sub in value:
                expr, params = _xlate_node(sub, params)
                sub_exprs.append(f"({expr})")
            parts.append("(" + " OR ".join(sub_exprs) + ")")
        elif key == "$not":
            assert isinstance(value, Mapping)
            expr, params = _xlate_node(value, params)
            parts.append(f"NOT ({expr})")
        elif key in LOGICAL_OPERATORS:
            raise FilterError(f"Unhandled logical operator: {key}")
        elif key.startswith("$"):
            raise FilterError(f"Unknown top-level operator: {key}")
        else:
            expr, params = _xlate_field(key, value, params)
            parts.append(expr)
    return " AND ".join(parts), params


# Metadata keys are interpolated into the SQL text (JSONB ``->>``
# takes a literal), so they must be validated — a quote or backslash
# in a key would be an injection vector. Conservative allow-list.
_SAFE_KEY_RE = re.compile(r"[A-Za-z0-9_.\- ]+\Z")

# SQL-side guard so casting a non-numeric metadata value to numeric
# can't blow up the whole query: only rows whose text looks like a
# JSON number get cast + compared; everything else simply doesn't
# match (same semantics as the in-memory store's typed comparison).
_NUMERIC_GUARD_SQL = r"'^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'"


def _safe_key(key: str) -> str:
    if not _SAFE_KEY_RE.fullmatch(key):
        raise FilterError(
            f"invalid metadata key for filtering: {key!r} "
            "(only letters, digits, '_', '.', '-' and spaces allowed)"
        )
    return key


def _is_numeric_operand(value: Any) -> bool:
    """True for int/float operands (bool excluded — it's an int
    subclass but compares as the JSON strings 'true'/'false')."""
    return isinstance(value, int | float) and not isinstance(value, bool)


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
            if op in _SQL_BIN_OPS:
                params.append(
                    expected
                    if _is_numeric_operand(expected)
                    else _pg_field_value(expected)
                )
                if _is_numeric_operand(expected):
                    # Numeric operand → compare numerically, not as
                    # text ("10" < "9" lexicographically!). Matches
                    # the module docstring's promised semantics.
                    sub_exprs.append(
                        f"((metadata->>'{key}') ~ {_NUMERIC_GUARD_SQL} "
                        f"AND ((metadata->>'{key}'))::numeric "
                        f"{_SQL_BIN_OPS[op]} ${len(params)})"
                    )
                else:
                    sub_exprs.append(
                        f"(metadata->>'{key}') "
                        f"{_SQL_BIN_OPS[op]} "
                        f"${len(params)}"
                    )
            elif op == "$in":
                if not isinstance(expected, list | tuple):
                    raise FilterError("$in expects a list")
                params.append(
                    [_pg_field_value(v) for v in expected]
                )
                sub_exprs.append(
                    f"(metadata->>'{key}') = ANY(${len(params)}::text[])"
                )
            elif op == "$nin":
                if not isinstance(expected, list | tuple):
                    raise FilterError("$nin expects a list")
                params.append(
                    [_pg_field_value(v) for v in expected]
                )
                sub_exprs.append(
                    f"((metadata->>'{key}') IS NULL OR "
                    f"(metadata->>'{key}') <> ALL(${len(params)}::text[]))"
                )
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
        params.append([_pg_field_value(v) for v in condition])
        return (
            f"(metadata->>'{key}') = ANY(${len(params)}::text[])",
            params,
        )

    if _is_numeric_operand(condition):
        params.append(condition)
        return (
            f"((metadata->>'{key}') ~ {_NUMERIC_GUARD_SQL} "
            f"AND ((metadata->>'{key}'))::numeric = ${len(params)})",
            params,
        )
    params.append(_pg_field_value(condition))
    return f"(metadata->>'{key}') = ${len(params)}", params


def _pg_field_value(v: Any) -> str:
    """JSONB ``->>`` always returns text; cast values to strings so
    the parameter binding matches."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


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
