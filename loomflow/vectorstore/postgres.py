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
        embedding   vector(N) NOT NULL
    );
    CREATE INDEX ON jeeves_vectors USING hnsw (embedding vector_cosine_ops);

Filter language: full Mongo-style operators translated to JSONB
SQL. ``$eq`` / ``$ne`` / ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` /
``$in`` / ``$nin`` / ``$and`` / ``$or`` / ``$not`` / ``$exists``
are all supported.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.ids import new_id
from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._filter import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from ._mmr import mmr_select
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
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        self._embedder = embedder
        self._dsn = dsn
        self._table = table
        self._dimension = dimension
        self._initialized = False

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
    ) -> PostgresVectorStore:
        """One-shot: construct a PostgresVectorStore + add ``chunks``."""
        store = cls(
            embedder=embedder,
            dsn=dsn,
            table=table,
            dimension=dimension,
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
        )

    async def _connect(self) -> Any:
        try:
            import asyncpg  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "asyncpg is not installed. "
                "Install with: pip install "
                "'loomflow[vectorstore-postgres]'."
            ) from exc
        return await asyncpg.connect(self._dsn)

    async def init_schema(self, dimension: int) -> None:
        """Create the table + HNSW index. Idempotent."""
        self._dimension = dimension
        conn = await self._connect()
        try:
            await conn.execute(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({dimension}) NOT NULL
                )
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
            self._initialized = True
        finally:
            await conn.close()

    async def add(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks:
            return []
        if ids is not None and len(ids) != len(chunks):
            raise ValueError(
                f"ids length ({len(ids)}) must match chunks "
                f"length ({len(chunks)})"
            )
        try:
            vectors = await self._embedder.embed_batch(
                [c.content for c in chunks]
            )
        except (AttributeError, NotImplementedError):
            vectors = [
                await self._embedder.embed(c.content) for c in chunks
            ]

        if not self._initialized:
            await self.init_schema(len(vectors[0]))

        assigned = (
            list(ids)
            if ids is not None
            else [new_id("vec") for _ in chunks]
        )

        conn = await self._connect()
        try:
            rows = [
                (
                    assigned[i],
                    chunks[i].content,
                    json.dumps(chunks[i].metadata or {}),
                    _vec_to_pg(vectors[i]),
                )
                for i in range(len(chunks))
            ]
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
        finally:
            await conn.close()
        return assigned

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        conn = await self._connect()
        try:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1::text[])",
                list(ids),
            )
        finally:
            await conn.close()

    async def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata
                FROM {self._table}
                WHERE id = ANY($1::text[])
                """,
                list(ids),
            )
        finally:
            await conn.close()
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

        # Wider candidate pool when MMR-reranking.
        n_fetch = max(k * 4, 20) if diversity else k
        params.append(n_fetch)

        sql = f"""
            SELECT id, content, metadata, embedding,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT ${len(params)}
        """

        conn = await self._connect()
        try:
            rows = await conn.fetch(sql, *params)
        finally:
            await conn.close()

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
            # pgvector returns embedding as text "[1.0,2.0,...]"
            emb = row["embedding"]
            cand_vecs.append(_pg_to_vec(emb))

        if diversity is None or diversity <= 0:
            return candidates[:k]

        chosen = mmr_select(vector, cand_vecs, k, diversity=diversity)
        return [candidates[i] for i in chosen]

    async def count(self) -> int:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) AS n FROM {self._table}"
            )
            return int(row["n"]) if row else 0
        finally:
            await conn.close()


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


def _xlate_field(
    key: str, condition: Any, params: list[Any]
) -> tuple[str, list[Any]]:
    """Translate one field constraint to a SQL boolean expression."""
    if isinstance(condition, Mapping) and condition and all(
        k.startswith("$") for k in condition
    ):
        sub_exprs: list[str] = []
        for op, expected in condition.items():
            if op in _SQL_BIN_OPS:
                params.append(_pg_field_value(expected))
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
