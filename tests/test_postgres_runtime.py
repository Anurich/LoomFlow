"""PostgresJournalStore + PostgresRuntime tests.

DDL snapshot + fake-pool routing tests run anywhere. A live integration
test is gated on ``JEEVES_TEST_PG_DSN``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from loomflow.runtime import PostgresJournalStore, PostgresRuntime
from loomflow.runtime.journal import JournalEntry

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# DDL snapshot
# ---------------------------------------------------------------------------


def test_schema_sql_uses_postgres_types() -> None:
    sql_blocks = PostgresJournalStore.schema_sql()
    joined = "\n".join(sql_blocks)
    assert "CREATE TABLE IF NOT EXISTS journal_steps" in joined
    assert "CREATE TABLE IF NOT EXISTS journal_streams" in joined
    # Postgres types — not the SQLite ones:
    assert "BYTEA" in joined
    assert "DOUBLE PRECISION" in joined
    assert "BLOB" not in joined  # would be SQLite


# ---------------------------------------------------------------------------
# Fake asyncpg pool — for offline routing tests
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, store: _FakeState) -> None:
        self._store = store

    async def execute(self, sql: str, *args: Any) -> str:
        self._store.executed.append((sql, args))
        return "OK"

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self._store.queried.append((sql, args))
        return self._store.next_row


class _FakeState:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.queried: list[tuple[str, tuple[Any, ...]]] = []
        self.next_row: Any = None


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *_: Any) -> None:
        return None


class _FakePool:
    def __init__(self, state: _FakeState) -> None:
        self._state = state

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(_FakeConn(self._state))


# ---------------------------------------------------------------------------
# put_step / get_step / streams round-trip via fake pool
# ---------------------------------------------------------------------------


async def test_put_step_emits_upsert_with_pickled_value() -> None:
    state = _FakeState()
    store = PostgresJournalStore(pool=_FakePool(state))

    await store.put_step("s1", "step_a", {"foo": 42})

    sql, args = state.executed[0]
    assert "INSERT INTO journal_steps" in sql
    assert "ON CONFLICT (session_id, step_name) DO UPDATE" in sql
    assert args[0] == "s1"
    assert args[1] == "step_a"
    # Third arg is the pickled value bytes.
    assert isinstance(args[2], bytes | bytearray)


async def test_get_step_returns_none_for_missing_row() -> None:
    state = _FakeState()
    state.next_row = None
    store = PostgresJournalStore(pool=_FakePool(state))
    out = await store.get_step("s1", "missing")
    assert out is None


async def test_get_step_unpickles_stored_value() -> None:
    import pickle

    state = _FakeState()
    state.next_row = {
        "value": pickle.dumps({"answer": 42}),
        "created_at": 1234567.89,
    }
    store = PostgresJournalStore(pool=_FakePool(state))
    entry = await store.get_step("s1", "step_a")
    assert isinstance(entry, JournalEntry)
    assert entry.value == {"answer": 42}
    assert entry.created_at == 1234567.89


async def test_put_stream_pickles_chunk_list() -> None:
    state = _FakeState()
    store = PostgresJournalStore(pool=_FakePool(state))
    await store.put_stream("s1", "stream_a", ["a", "b", "c"])
    sql, args = state.executed[0]
    assert "INSERT INTO journal_streams" in sql
    assert isinstance(args[2], bytes | bytearray)


async def test_get_stream_unpickles_chunks() -> None:
    import pickle

    state = _FakeState()
    state.next_row = {"chunks": pickle.dumps(["a", "b", "c"])}
    store = PostgresJournalStore(pool=_FakePool(state))
    chunks = await store.get_stream("s1", "stream_a")
    assert chunks == ["a", "b", "c"]


async def test_init_schema_runs_both_table_ddls() -> None:
    state = _FakeState()
    store = PostgresJournalStore(pool=_FakePool(state))
    await store.init_schema()
    sqls = [s for s, _ in state.executed]
    assert any("journal_steps" in s for s in sqls)
    assert any("journal_streams" in s for s in sqls)


# ---------------------------------------------------------------------------
# PostgresRuntime — convenience subclass
# ---------------------------------------------------------------------------


async def test_postgres_runtime_uses_postgres_journal_store() -> None:
    state = _FakeState()
    runtime = PostgresRuntime(pool=_FakePool(state))
    assert isinstance(runtime.store, PostgresJournalStore)
    # name is overridden so telemetry can distinguish backends.
    assert runtime.name == "postgres"


async def test_postgres_runtime_init_schema_delegates_to_store() -> None:
    state = _FakeState()
    runtime = PostgresRuntime(pool=_FakePool(state))
    await runtime.init_schema()
    sqls = [s for s, _ in state.executed]
    assert any("journal_steps" in s for s in sqls)


# ---------------------------------------------------------------------------
# Live integration — gated on real Postgres
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("JEEVES_TEST_PG_DSN"),
    reason="JEEVES_TEST_PG_DSN env var not set",
)
async def test_live_postgres_runtime_replay() -> None:  # pragma: no cover
    dsn = os.environ["JEEVES_TEST_PG_DSN"]
    runtime = await PostgresRuntime.connect(dsn)
    try:
        await runtime.init_schema()
        counter = {"runs": 0}

        async def expensive() -> str:
            counter["runs"] += 1
            return f"v{counter['runs']}"

        async with runtime.session("live-replay-test"):
            v1 = await runtime.step("step_a", expensive)
            v2 = await runtime.step("step_a", expensive)

        assert v1 == "v1"
        assert v2 == "v1"  # replayed from journal
        assert counter["runs"] == 1
    finally:
        await runtime.aclose()
