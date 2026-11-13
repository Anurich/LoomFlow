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

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from ..core.protocols import Embedder
from ..core.types import Fact, _normalize_predicate
from ..vectorstore._pgvector import (
    HNSW_MAX_DIMENSIONS,
    hnsw_scan,
    index_reference,
    table_reference,
    validate_schema,
    validate_table_prefix,
)
from .facts import plan_supersession

logger = logging.getLogger(__name__)


class PostgresFactStore:
    """Postgres-backed bi-temporal fact store.

    The table is ``facts`` by default. ``table_prefix`` (e.g.
    ``"myapp_"`` → ``myapp_facts``) and ``schema`` move it out of the
    way of an application's own tables; both must be plain
    identifiers.
    """

    def __init__(
        self,
        pool: Any,
        *,
        embedder: Embedder | None = None,
        table_prefix: str = "",
        schema: str | None = None,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._table_prefix = validate_table_prefix(table_prefix)
        self._schema = validate_schema(schema)
        self._table = table_reference(self._schema, self._table_prefix, "facts")
        # Validate the longest derived index name up front too.
        self._index("facts_user_subject_predicate_idx")

    def _index(self, name: str) -> str:
        return index_reference(self._table_prefix, name)

    # ---- factory ---------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        embedder: Embedder | None = None,
        min_size: int = 1,
        max_size: int = 10,
        table_prefix: str = "",
        schema: str | None = None,
    ) -> PostgresFactStore:
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

        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            init=_setup,
        )
        return cls(
            pool, embedder=embedder, table_prefix=table_prefix, schema=schema
        )

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
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"  id TEXT PRIMARY KEY,"
                f"  user_id TEXT,"
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
            f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS user_id TEXT;",
            (
                f"CREATE INDEX IF NOT EXISTS {self._index('facts_subject_idx')} "
                f"ON {self._table} (subject);"
            ),
            (
                "CREATE INDEX IF NOT EXISTS "
                f"{self._index('facts_user_subject_predicate_idx')} "
                f"ON {self._table} (user_id, subject, predicate);"
            ),
        ]
        # pgvector can't HNSW-index more than 2000 dimensions; above
        # that the index is skipped (``init_schema`` logs it) and
        # recall falls back to an exact scan.
        if self._embedder is not None and self._dimensions() <= HNSW_MAX_DIMENSIONS:
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {self._index('facts_embedding_idx')} "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops) "
                "WHERE embedding IS NOT NULL;"
            )
        return statements

    def _dimensions(self) -> int:
        return self._embedder.dimensions if self._embedder is not None else 1

    async def init_schema(self) -> None:
        if self._embedder is not None and self._dimensions() > HNSW_MAX_DIMENSIONS:
            logger.warning(
                "PostgresFactStore: the embedder produces %d-dimensional "
                "vectors, above pgvector's %d-dimension limit for HNSW "
                "indexes; no vector index is created on %s and fact recall "
                "scans the table (exact, but slow when large).",
                self._dimensions(),
                HNSW_MAX_DIMENSIONS,
                self._table,
            )
        async with self._pool.acquire() as conn:
            if self._schema is not None:
                await conn.execute(
                    f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";'
                )
            for stmt in self.schema_sql():
                await conn.execute(stmt)

    # ---- mutation --------------------------------------------------------

    async def append(self, fact: Fact) -> str:
        embedding: list[float] | None = None
        if self._embedder is not None:
            triple = f"{fact.subject} {fact.predicate} {fact.object}"
            embedding = await self._embedder.embed(triple)

        async with self._pool.acquire() as conn:
            await self._append_on_conn(conn, fact, embedding)
        return fact.id

    async def append_many(self, facts: Iterable[Fact]) -> list[str]:
        """Batched append: ONE ``embed_batch`` call for all triples
        and ONE pooled connection + transaction for the whole batch
        (instead of a per-fact acquire + embed + supersede + insert).

        The supersede UPDATE and INSERT still run per fact *inside*
        the transaction — a single executemany of all UPDATEs before
        all INSERTs would break intra-batch supersession (fact B in
        the batch must be able to close off fact A appended just
        before it). The transaction makes the batch atomic: a
        cancellation mid-batch rolls back cleanly.
        """
        materialised = list(facts)
        if not materialised:
            return []

        embeddings: list[list[float] | None]
        if self._embedder is not None:
            triples = [
                f"{f.subject} {f.predicate} {f.object}"
                for f in materialised
            ]
            embeddings = list(await self._embedder.embed_batch(triples))
        else:
            embeddings = [None] * len(materialised)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for fact, embedding in zip(
                    materialised, embeddings, strict=True
                ):
                    await self._append_on_conn(conn, fact, embedding)
        return [f.id for f in materialised]

    async def _append_on_conn(
        self,
        conn: Any,
        fact: Fact,
        embedding: list[float] | None,
    ) -> None:
        """Supersede + insert one fact on an already-acquired
        connection. Shared by :meth:`append` and :meth:`append_many`.
        """
        # Valid-time supersession (``plan_supersession``) over the
        # contradicting facts in this user's namespace — a fact from
        # user A never invalidates user B's claim. ``IS NOT DISTINCT
        # FROM`` makes the ``NULL = NULL`` case (anonymous bucket)
        # work correctly.
        rows = await conn.fetch(
            "SELECT id, user_id, subject, predicate, object, confidence, "
            "valid_from, valid_until, recorded_at, sources, embedding "
            f"FROM {self._table} WHERE user_id IS NOT DISTINCT FROM $1 "
            "AND subject = $2 AND predicate = $3 AND object != $4",
            fact.user_id,
            fact.subject,
            fact.predicate,
            fact.object,
        )
        closes, until = plan_supersession(fact, [_row_to_fact(r) for r in rows])
        for existing_id, valid_until in closes.items():
            await conn.execute(
                f"UPDATE {self._table} SET valid_until = $1 WHERE id = $2",
                valid_until,
                existing_id,
            )
        fact = fact.model_copy(update={"valid_until": until})
        await conn.execute(
            f"INSERT INTO {self._table} "
            "(id, user_id, subject, predicate, object, confidence, "
            " valid_from, valid_until, recorded_at, sources, embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) "
            "ON CONFLICT (id) DO NOTHING;",
            fact.id,
            fact.user_id,
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

    # ---- queries ---------------------------------------------------------

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[Fact]:
        # Hard namespace partition by ``user_id``. Always present in
        # the WHERE clause; uses ``IS NOT DISTINCT FROM`` so the
        # anonymous (NULL) bucket matches itself.
        clauses: list[str] = ["user_id IS NOT DISTINCT FROM $1"]
        params: list[Any] = [user_id]
        idx = 2
        if subject is not None:
            clauses.append(f"subject = ${idx}")
            params.append(subject)
            idx += 1
        if predicate is not None:
            clauses.append(f"predicate = ${idx}")
            params.append(_normalize_predicate(predicate))
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
            "SELECT id, user_id, subject, predicate, object, confidence, "
            "valid_from, valid_until, recorded_at, sources, embedding "
            f"FROM {self._table} "
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
        user_id: str | None = None,
    ) -> list[Fact]:
        if self._embedder is not None:
            return await self._recall_embedding(query, limit, valid_at, user_id)
        return await self._recall_ilike(query, limit, valid_at, user_id)

    async def _recall_embedding(
        self,
        query: str,
        limit: int,
        valid_at: datetime | None,
        user_id: str | None,
    ) -> list[Fact]:
        assert self._embedder is not None
        query_embedding = await self._embedder.embed(query)

        # Cosine distance via pgvector's ``<=>`` operator. Hard
        # ``user_id`` partition + NULL-embedding exclusion + optional
        # bi-temporal window.
        clauses = ["embedding IS NOT NULL", "user_id IS NOT DISTINCT FROM $1"]
        params: list[Any] = [user_id]
        idx = 2
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
            "SELECT id, user_id, subject, predicate, object, confidence, "
            "valid_from, valid_until, recorded_at, sources, embedding "
            f"FROM {self._table} "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY embedding <=> ${embed_idx} "
            f"LIMIT ${idx}"
        )

        async with self._pool.acquire() as conn, hnsw_scan(conn, fetch=limit, filtered=True):
            rows = await conn.fetch(sql, *params)
        return [_row_to_fact(r) for r in rows]

    async def _recall_ilike(
        self,
        query: str,
        limit: int,
        valid_at: datetime | None,
        user_id: str | None,
    ) -> list[Fact]:
        clauses = ["user_id IS NOT DISTINCT FROM $1"]
        params: list[Any] = [user_id]
        idx = 2
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
            "SELECT id, user_id, subject, predicate, object, confidence, "
            "valid_from, valid_until, recorded_at, sources, embedding "
            f"FROM {self._table} "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY recorded_at DESC LIMIT ${idx}"
        )

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_fact(r) for r in rows]

    async def all_facts(self) -> list[Fact]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, user_id, subject, predicate, object, confidence, "
                "valid_from, valid_until, recorded_at, sources, embedding "
                f"FROM {self._table} ORDER BY recorded_at DESC"
            )
        return [_row_to_fact(r) for r in rows]

    # ---- GDPR surface ------------------------------------------------------

    async def delete(
        self,
        *,
        user_id: str | None = None,
        before: datetime | None = None,
    ) -> int:
        """Delete every fact in the ``user_id`` partition (optionally
        only those recorded before ``before``). Returns the number of
        rows removed."""
        clauses = ["user_id IS NOT DISTINCT FROM $1"]
        params: list[Any] = [user_id]
        if before is not None:
            clauses.append("recorded_at < $2")
            params.append(before)
        sql = f"DELETE FROM {self._table} WHERE " + " AND ".join(clauses)
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params)
        return _parse_delete_count(result)

    async def count(self, *, user_id: str | None = None) -> int:
        """``SELECT COUNT(*)`` over the ``user_id`` partition."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) AS c FROM {self._table} "
                "WHERE user_id IS NOT DISTINCT FROM $1",
                user_id,
            )
        return int(row["c"]) if row else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_delete_count(asyncpg_result: str) -> int:
    """asyncpg's ``Connection.execute`` returns a status string like
    ``"DELETE 17"``; parse the trailing count (``0`` when absent)."""
    parts = str(asyncpg_result).strip().split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def _row_to_fact(row: Any) -> Fact:
    sources = row["sources"]
    if sources is None:
        sources = []
    # Older rows from a pre-migration schema may not include user_id.
    try:
        user_id_val = row["user_id"]
    except (KeyError, IndexError):
        user_id_val = None
    return Fact(
        id=row["id"],
        user_id=user_id_val,
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        confidence=row["confidence"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        recorded_at=row["recorded_at"],
        sources=list(sources),
    )
