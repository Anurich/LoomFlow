"""PostgresFactStore tests.

DDL snapshot tests run anywhere — they assert on the SQL strings the
backend would emit. Append/query routing tests use a fake asyncpg-shaped
pool so we don't need a live database. Live integration is gated on
``JEEVES_TEST_PG_DSN``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from jeevesagent.core.types import Fact
from jeevesagent.memory.embedder import HashEmbedder
from jeevesagent.memory.postgres_facts import PostgresFactStore

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# DDL — runs anywhere
# ---------------------------------------------------------------------------


def test_schema_sql_includes_pgvector_table_and_indexes() -> None:
    store = PostgresFactStore(pool=None, embedder=HashEmbedder(dimensions=384))
    statements = "\n".join(store.schema_sql())
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
    assert "CREATE TABLE IF NOT EXISTS facts" in statements
    assert "vector(384)" in statements
    assert "facts_subject_idx" in statements
    assert "facts_subject_predicate_idx" in statements
    assert "hnsw" in statements  # only when embedder configured


def test_schema_sql_skips_hnsw_index_when_no_embedder() -> None:
    store = PostgresFactStore(pool=None, embedder=None)
    statements = "\n".join(store.schema_sql())
    # Vector column placeholder still gets emitted (dim=1) so the
    # column exists for storage; the HNSW index does not.
    assert "facts_embedding_idx" not in statements


def test_schema_dimensions_track_embedder() -> None:
    store = PostgresFactStore(pool=None, embedder=HashEmbedder(dimensions=128))
    assert any("vector(128)" in s for s in store.schema_sql())


# ---------------------------------------------------------------------------
# Fake asyncpg pool — for offline routing tests
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    async def execute(self, sql: str, *args: Any) -> str:
        self._store.executed.append((sql, args))
        return "OK"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._store.queried.append((sql, args))
        return self._store.next_rows


class _FakeStore:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.queried: list[tuple[str, tuple[Any, ...]]] = []
        self.next_rows: list[Any] = []


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *_: Any) -> None:
        return None


class _FakePool:
    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(_FakeConn(self._store))


def _fact(
    *,
    subject: str = "user",
    predicate: str = "name_is",
    object_: str = "Alice",
    valid_from: datetime | None = None,
) -> Fact:
    base = valid_from or datetime.now(UTC)
    return Fact(
        subject=subject,
        predicate=predicate,
        object=object_,
        valid_from=base,
        recorded_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Append routing
# ---------------------------------------------------------------------------


async def test_append_emits_supersede_then_insert() -> None:
    store_state = _FakeStore()
    pool = _FakePool(store_state)
    store = PostgresFactStore(pool=pool)

    await store.append(_fact())

    sqls = [sql for sql, _ in store_state.executed]
    assert any("UPDATE facts SET valid_until" in s for s in sqls)
    assert any("INSERT INTO facts" in s for s in sqls)
    # Supersede must run first so the new fact's row doesn't get
    # invalidated by its own update.
    update_idx = next(
        i for i, s in enumerate(sqls) if s.startswith("UPDATE")
    )
    insert_idx = next(
        i for i, s in enumerate(sqls) if s.startswith("INSERT")
    )
    assert update_idx < insert_idx


async def test_append_with_embedder_runs_embed_and_passes_vector() -> None:
    store_state = _FakeStore()
    pool = _FakePool(store_state)
    store = PostgresFactStore(
        pool=pool,
        embedder=HashEmbedder(dimensions=32),
    )

    await store.append(_fact())

    insert = next(
        (sql, args)
        for sql, args in store_state.executed
        if sql.startswith("INSERT")
    )
    sql, args = insert
    # 10 placeholders ⇒ embedding is the 10th positional arg.
    assert "$10" in sql
    assert isinstance(args[9], list)
    assert len(args[9]) == 32


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------


async def test_query_with_filters_assembles_clauses() -> None:
    store_state = _FakeStore()
    pool = _FakePool(store_state)
    store = PostgresFactStore(pool=pool)

    base = datetime(2026, 1, 1, tzinfo=UTC)
    await store.query(
        subject="alice",
        predicate="lives_in",
        valid_at=base,
        limit=4,
    )

    sql, args = store_state.queried[0]
    assert "subject = $1" in sql
    assert "predicate = $2" in sql
    assert "valid_from <= $3" in sql
    assert "ORDER BY recorded_at DESC LIMIT $4" in sql
    assert args[0] == "alice"
    assert args[1] == "lives_in"
    assert args[2] == base
    assert args[3] == 4


async def test_recall_text_uses_pgvector_distance_when_embedder_set() -> None:
    store_state = _FakeStore()
    pool = _FakePool(store_state)
    store = PostgresFactStore(
        pool=pool,
        embedder=HashEmbedder(dimensions=32),
    )

    await store.recall_text("anything", limit=3)

    sql, args = store_state.queried[0]
    assert "embedding <=>" in sql
    assert "embedding IS NOT NULL" in sql
    # Embedding arg is a vector list.
    assert any(isinstance(a, list) for a in args)


async def test_recall_text_without_embedder_uses_ilike() -> None:
    store_state = _FakeStore()
    pool = _FakePool(store_state)
    store = PostgresFactStore(pool=pool)

    await store.recall_text("alice tokyo", limit=3)
    sql, _ = store_state.queried[0]
    assert "ILIKE" in sql


async def test_recall_text_time_window_passed_through() -> None:
    store_state = _FakeStore()
    pool = _FakePool(store_state)
    store = PostgresFactStore(pool=pool, embedder=HashEmbedder(dimensions=8))

    base = datetime(2026, 1, 1, tzinfo=UTC)
    await store.recall_text("anything", limit=5, valid_at=base)
    sql, args = store_state.queried[0]
    assert "valid_from <= $1" in sql
    assert args[0] == base


# ---------------------------------------------------------------------------
# PostgresMemory + PostgresFactStore wiring (mocked)
# ---------------------------------------------------------------------------


async def test_postgres_memory_init_schema_runs_fact_store_schema_too() -> None:
    """When ``with_facts=True`` was used, ``init_schema`` should also
    apply the facts schema."""

    from jeevesagent.memory.postgres import PostgresMemory

    state = _FakeStore()
    pool = _FakePool(state)

    fact_store = PostgresFactStore(
        pool=pool,
        embedder=HashEmbedder(dimensions=16),
    )
    mem = PostgresMemory(
        pool=pool,
        embedder=HashEmbedder(dimensions=16),
        fact_store=fact_store,
    )
    await mem.init_schema()

    sqls = [sql for sql, _ in state.executed]
    # Memory schema produced episodes table.
    assert any("episodes" in s for s in sqls)
    # Fact-store schema produced facts table.
    assert any("CREATE TABLE IF NOT EXISTS facts" in s for s in sqls)


# ---------------------------------------------------------------------------
# Live integration — requires a real Postgres
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("JEEVES_TEST_PG_DSN"),
    reason="JEEVES_TEST_PG_DSN env var not set",
)
async def test_live_postgres_fact_store_roundtrip() -> None:  # pragma: no cover
    dsn = os.environ["JEEVES_TEST_PG_DSN"]
    store = await PostgresFactStore.connect(
        dsn, embedder=HashEmbedder(dimensions=16)
    )
    try:
        await store.init_schema()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        await store.append(_fact(predicate="lives_in", object_="Tokyo", valid_from=base))
        await store.append(
            _fact(
                predicate="lives_in",
                object_="Paris",
                valid_from=base + timedelta(days=30),
            )
        )
        # supersession applied
        on_feb = base + timedelta(days=45)
        facts = await store.query(predicate="lives_in", valid_at=on_feb)
        assert {f.object for f in facts} == {"Paris"}
    finally:
        await store.aclose()
