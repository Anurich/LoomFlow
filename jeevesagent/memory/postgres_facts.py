"""Postgres + pgvector :class:`FactStore`.

Schema (created by :meth:`init_schema`):

* ``facts(id, subject, predicate, object, confidence, valid_from,
  valid_until, recorded_at, sources, embedding vector(N))`` with
  optional HNSW index on ``embedding`` (only when an embedder is
  configured at construction time — the dimension is fixed in the
  column type).

The ``vector(N)`` dimension is locked at table-creation time. Switching
embedders later requires migrating the table.

Lazy ``asyncpg`` + ``pgvector.asyncpg`` imports inside :meth:`connect`
mirror the pattern in :mod:`memory.postgres`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from ..core.protocols import Embedder
from ..core.types import Fact


class PostgresFactStore:
    """Postgres-backed bi-temporal fact store."""

    def __init__(
        self,
        pool: Any,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self._pool = pool
        self._embedder = embedder

    # ---- factory ---------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        embedder: Embedder | None = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> PostgresFactStore:
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
        return cls(pool, embedder=embedder)

    async def aclose(self) -> None:
        if self._pool is not None and hasattr(self._pool, "close"):
            await self._pool.close()

    # ---- schema ----------------------------------------------------------

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    def schema_sql(self) -> list[str]:
        """Return the DDL for this fact store's schema.

        Exposed so tests can assert on the SQL strings, and so
        migration scripts can apply the schema in their own
        transaction.
        """
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            (
                f"CREATE TABLE IF NOT EXISTS facts ("
                f"  id TEXT PRIMARY KEY,"
                f"  subject TEXT NOT NULL,"
                f"  predicate TEXT NOT NULL,"
                f"  object TEXT NOT NULL,"
                f"  confidence REAL NOT NULL DEFAULT 1.0,"
                f"  valid_from TIMESTAMPTZ NOT NULL,"
                f"  valid_until TIMESTAMPTZ,"
                f"  recorded_at TIMESTAMPTZ NOT NULL,"
                f"  sources TEXT[] NOT NULL DEFAULT '{{}}',"
                f"  embedding vector({self._dimensions()}) "
                f");"
            ),
            (
                "CREATE INDEX IF NOT EXISTS facts_subject_idx "
                "ON facts (subject);"
            ),
            (
                "CREATE INDEX IF NOT EXISTS facts_subject_predicate_idx "
                "ON facts (subject, predicate);"
            ),
        ]
        if self._embedder is not None:
            statements.append(
                "CREATE INDEX IF NOT EXISTS facts_embedding_idx "
                "ON facts USING hnsw (embedding vector_cosine_ops) "
                "WHERE embedding IS NOT NULL;"
            )
        return statements

    def _dimensions(self) -> int:
        return self._embedder.dimensions if self._embedder is not None else 1

    async def init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            for stmt in self.schema_sql():
                await conn.execute(stmt)

    # ---- mutation --------------------------------------------------------

    async def append(self, fact: Fact) -> str:
        embedding: list[float] | None = None
        if self._embedder is not None:
            triple = f"{fact.subject} {fact.predicate} {fact.object}"
            embedding = await self._embedder.embed(triple)

        async with self._pool.acquire() as conn:
            # Supersession: close off any currently-valid predecessors
            # for the same (subject, predicate) where the object differs.
            await conn.execute(
                "UPDATE facts SET valid_until = $1 "
                "WHERE subject = $2 AND predicate = $3 "
                "AND object != $4 AND valid_until IS NULL",
                fact.valid_from,
                fact.subject,
                fact.predicate,
                fact.object,
            )
            await conn.execute(
                "INSERT INTO facts "
                "(id, subject, predicate, object, confidence, "
                " valid_from, valid_until, recorded_at, sources, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) "
                "ON CONFLICT (id) DO NOTHING;",
                fact.id,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.confidence,
                fact.valid_from,
                fact.valid_until,
                fact.recorded_at,
                list(fact.sources),
                embedding,
            )
        return fact.id

    async def append_many(self, facts: Iterable[Fact]) -> list[str]:
        return [await self.append(f) for f in facts]

    # ---- queries ---------------------------------------------------------

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
    ) -> list[Fact]:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        idx = 1
        if subject is not None:
            clauses.append(f"subject = ${idx}")
            params.append(subject)
            idx += 1
        if predicate is not None:
            clauses.append(f"predicate = ${idx}")
            params.append(predicate)
            idx += 1
        if object_ is not None:
            clauses.append(f"object = ${idx}")
            params.append(object_)
            idx += 1
        if valid_at is not None:
            clauses.append(
                f"valid_from <= ${idx} "
                f"AND (valid_until IS NULL OR ${idx} < valid_until)"
            )
            params.append(valid_at)
            idx += 1

        sql = (
            "SELECT id, subject, predicate, object, confidence, "
            "valid_from, valid_until, recorded_at, sources, embedding "
            "FROM facts "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY recorded_at DESC LIMIT ${idx}"
        )
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_fact(r) for r in rows]

    async def recall_text(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
    ) -> list[Fact]:
        if self._embedder is not None:
            return await self._recall_embedding(query, limit, valid_at)
        return await self._recall_ilike(query, limit, valid_at)

    async def _recall_embedding(
        self,
        query: str,
        limit: int,
        valid_at: datetime | None,
    ) -> list[Fact]:
        assert self._embedder is not None
        query_embedding = await self._embedder.embed(query)

        # Cosine distance via pgvector's ``<=>`` operator. NULL
        # embeddings are excluded so the index stays useful.
        clauses = ["embedding IS NOT NULL"]
        params: list[Any] = []
        idx = 1
        if valid_at is not None:
            clauses.append(
                f"valid_from <= ${idx} "
                f"AND (valid_until IS NULL OR ${idx} < valid_until)"
            )
            params.append(valid_at)
            idx += 1
        params.append(query_embedding)
        embed_idx = idx
        idx += 1
        params.append(limit)

        sql = (
            "SELECT id, subject, predicate, object, confidence, "
            "valid_from, valid_until, recorded_at, sources, embedding "
            "FROM facts "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY embedding <=> ${embed_idx} "
            f"LIMIT ${idx}"
        )

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_fact(r) for r in rows]

    async def _recall_ilike(
        self,
        query: str,
        limit: int,
        valid_at: datetime | None,
    ) -> list[Fact]:
        clauses = ["1=1"]
        params: list[Any] = []
        idx = 1
        if valid_at is not None:
            clauses.append(
                f"valid_from <= ${idx} "
                f"AND (valid_until IS NULL OR ${idx} < valid_until)"
            )
            params.append(valid_at)
            idx += 1

        terms = [t for t in query.split() if t.strip()]
        if terms:
            term_clauses: list[str] = []
            for term in terms:
                pattern = f"%{term}%"
                term_clauses.append(
                    f"(subject ILIKE ${idx} OR predicate ILIKE ${idx} "
                    f"OR object ILIKE ${idx})"
                )
                params.append(pattern)
                idx += 1
            clauses.append("(" + " OR ".join(term_clauses) + ")")

        params.append(limit)
        sql = (
            "SELECT id, subject, predicate, object, confidence, "
            "valid_from, valid_until, recorded_at, sources, embedding "
            "FROM facts "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY recorded_at DESC LIMIT ${idx}"
        )

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_fact(r) for r in rows]

    async def all_facts(self) -> list[Fact]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, subject, predicate, object, confidence, "
                "valid_from, valid_until, recorded_at, sources, embedding "
                "FROM facts ORDER BY recorded_at DESC"
            )
        return [_row_to_fact(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_fact(row: Any) -> Fact:
    embedding = row["embedding"]
    if embedding is not None and not isinstance(embedding, list):
        embedding = list(embedding)
    sources = row["sources"]
    if sources is None:
        sources = []
    return Fact(
        id=row["id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        confidence=row["confidence"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        recorded_at=row["recorded_at"],
        sources=list(sources),
    )
