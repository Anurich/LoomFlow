"""Checkpoint substrate tests (G4a) — put/get/latest/list/retention on
all three journal stores, JSON round-trip fidelity, prune coverage, and
the runtime-level delegation surface.

Postgres is exercised through the same fake-asyncpg-pool pattern as
``tests/test_postgres_runtime.py``: SQL routing and argument shapes are
asserted offline; live behavior is covered by Sqlite parity.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from loomflow.core.types import Message, Role, ToolCall, Usage
from loomflow.runtime import (
    Checkpoint,
    CheckpointMeta,
    InMemoryJournalStore,
    InProcRuntime,
    JournaledRuntime,
    PostgresJournalStore,
    SqliteJournalStore,
)

pytestmark = pytest.mark.anyio


def _mk_checkpoint(
    session_id: str = "s1",
    turn: int = 1,
    *,
    at: datetime | None = None,
    **kwargs: Any,
) -> Checkpoint:
    extra: dict[str, Any] = dict(kwargs)
    if at is not None:
        extra["created_at"] = at
    return Checkpoint(
        session_id=session_id,
        turn=turn,
        messages=[
            Message(role=Role.USER, content=f"prompt for turn {turn}"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(
                    ToolCall(tool="search", args={"q": "loomflow", "n": turn}),
                ),
            ),
        ],
        cumulative_usage=Usage(
            input_tokens=100 * turn,
            cached_input_tokens=7,
            output_tokens=42 * turn,
            cost_usd=0.003 * turn,
        ),
        cursor=f"node-{turn}",
        **extra,
    )


def _t(seconds: int) -> datetime:
    return datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC) + timedelta(
        seconds=seconds
    )


# ---------------------------------------------------------------------------
# put / get / latest / list — InMemory + Sqlite (real behavior)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["inmemory", "sqlite"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> InMemoryJournalStore | SqliteJournalStore:
    if request.param == "inmemory":
        return InMemoryJournalStore()
    return SqliteJournalStore(tmp_path / "ckpt.db")


async def test_put_then_get_roundtrip(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    cp = _mk_checkpoint(turn=3)
    await store.put_checkpoint(cp)
    loaded = await store.get_checkpoint("s1", cp.checkpoint_id)
    assert loaded == cp


async def test_get_missing_returns_none(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    assert await store.get_checkpoint("s1", "ckpt_missing") is None
    assert await store.get_latest_checkpoint("s1") is None
    assert await store.list_checkpoints("s1") == []


async def test_get_latest_returns_newest(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    for turn in range(1, 4):
        await store.put_checkpoint(
            _mk_checkpoint(turn=turn, at=_t(turn))
        )
    latest = await store.get_latest_checkpoint("s1")
    assert latest is not None
    assert latest.turn == 3
    assert latest.cursor == "node-3"


async def test_list_returns_meta_newest_first(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    cps = [_mk_checkpoint(turn=t, at=_t(t)) for t in range(1, 5)]
    for cp in cps:
        await store.put_checkpoint(cp)

    metas = await store.list_checkpoints("s1")
    assert [m.turn for m in metas] == [4, 3, 2, 1]
    assert [m.checkpoint_id for m in metas] == [
        cp.checkpoint_id for cp in reversed(cps)
    ]
    # meta rows are cheap: id/turn/created_at only, no transcript
    assert all(isinstance(m, CheckpointMeta) for m in metas)
    assert not hasattr(metas[0], "messages")
    # created_at round-trips through the store's epoch column
    assert abs((metas[0].created_at - _t(4)).total_seconds()) < 1e-3

    limited = await store.list_checkpoints("s1", limit=2)
    assert [m.turn for m in limited] == [4, 3]


async def test_sessions_are_isolated(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    cp_a = _mk_checkpoint("sess-a", turn=1, at=_t(1))
    cp_b = _mk_checkpoint("sess-b", turn=9, at=_t(2))
    await store.put_checkpoint(cp_a)
    await store.put_checkpoint(cp_b)

    assert await store.get_checkpoint("sess-a", cp_b.checkpoint_id) is None
    latest_a = await store.get_latest_checkpoint("sess-a")
    assert latest_a is not None and latest_a.turn == 1
    assert [m.turn for m in await store.list_checkpoints("sess-b")] == [9]


# ---------------------------------------------------------------------------
# JSON round-trip fidelity (no pickle)
# ---------------------------------------------------------------------------


async def test_messages_and_usage_survive_json_roundtrip(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    cp = _mk_checkpoint(turn=2)
    await store.put_checkpoint(cp)
    loaded = await store.get_checkpoint("s1", cp.checkpoint_id)

    assert loaded is not None
    assert loaded.messages == cp.messages
    assert loaded.messages[1].tool_calls[0].tool == "search"
    assert loaded.messages[1].tool_calls[0].args == {"q": "loomflow", "n": 2}
    assert loaded.cumulative_usage == cp.cumulative_usage
    assert loaded.cursor == cp.cursor
    assert loaded.created_at == cp.created_at


def test_checkpoint_serializes_to_plain_json() -> None:
    cp = _mk_checkpoint()
    payload = cp.model_dump_json()
    parsed = json.loads(payload)  # valid JSON, not pickle
    assert parsed["session_id"] == "s1"
    assert Checkpoint.model_validate_json(payload) == cp


async def test_sqlite_payload_column_is_json_text(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "ckpt.db")
    cp = _mk_checkpoint()
    await store.put_checkpoint(cp)
    row = store._conn.execute(
        "SELECT payload FROM checkpoints WHERE checkpoint_id = ?",
        (cp.checkpoint_id,),
    ).fetchone()
    assert isinstance(row[0], str)
    assert json.loads(row[0])["turn"] == 1


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["inmemory", "sqlite"])
async def test_retention_keeps_last_n_per_session(
    backend: str, tmp_path: Path
) -> None:
    capped: InMemoryJournalStore | SqliteJournalStore
    if backend == "inmemory":
        capped = InMemoryJournalStore(max_checkpoints_per_session=3)
    else:
        capped = SqliteJournalStore(
            tmp_path / "capped.db", max_checkpoints_per_session=3
        )

    cps = [_mk_checkpoint(turn=t, at=_t(t)) for t in range(1, 6)]
    for cp in cps:
        await capped.put_checkpoint(cp)

    metas = await capped.list_checkpoints("s1")
    assert [m.turn for m in metas] == [5, 4, 3]  # oldest pruned on put
    # the pruned ones are really gone
    assert await capped.get_checkpoint("s1", cps[0].checkpoint_id) is None
    # other sessions have their own budget
    await capped.put_checkpoint(_mk_checkpoint("other", turn=1, at=_t(9)))
    assert len(await capped.list_checkpoints("s1")) == 3
    assert len(await capped.list_checkpoints("other")) == 1


# ---------------------------------------------------------------------------
# prune() covers checkpoints
# ---------------------------------------------------------------------------


async def test_prune_by_session_clears_checkpoints(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    await store.put_checkpoint(_mk_checkpoint("gone", turn=1, at=_t(1)))
    await store.put_checkpoint(_mk_checkpoint("kept", turn=2, at=_t(2)))

    await store.prune(session_id="gone")

    assert await store.get_latest_checkpoint("gone") is None
    assert await store.list_checkpoints("gone") == []
    kept = await store.get_latest_checkpoint("kept")
    assert kept is not None and kept.turn == 2


async def test_prune_before_drops_only_old_checkpoints(
    store: InMemoryJournalStore | SqliteJournalStore,
) -> None:
    await store.put_checkpoint(_mk_checkpoint(turn=1, at=_t(0)))
    await store.put_checkpoint(_mk_checkpoint(turn=2, at=_t(100)))

    await store.prune(before=_t(50))

    metas = await store.list_checkpoints("s1")
    assert [m.turn for m in metas] == [2]


# ---------------------------------------------------------------------------
# runtime-level surface — JournaledRuntime delegates, InProc is in-memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["inproc", "journaled"])
async def test_runtime_checkpoint_surface(kind: str) -> None:
    runtime: InProcRuntime | JournaledRuntime
    if kind == "inproc":
        runtime = InProcRuntime()
    else:
        runtime = JournaledRuntime()

    cps = [_mk_checkpoint(turn=t, at=_t(t)) for t in range(1, 4)]
    for cp in cps:
        await runtime.put_checkpoint(cp)

    latest = await runtime.get_latest_checkpoint("s1")
    assert latest is not None and latest.turn == 3
    assert (
        await runtime.get_checkpoint("s1", cps[0].checkpoint_id)
    ) == cps[0]
    assert [m.turn for m in await runtime.list_checkpoints("s1")] == [3, 2, 1]
    assert await runtime.get_latest_checkpoint("elsewhere") is None


async def test_runtime_retention_kwarg_is_honoured() -> None:
    for runtime in (
        InProcRuntime(max_checkpoints_per_session=2),
        JournaledRuntime(max_checkpoints_per_session=2),
    ):
        for t in range(1, 5):
            await runtime.put_checkpoint(_mk_checkpoint(turn=t, at=_t(t)))
        assert [m.turn for m in await runtime.list_checkpoints("s1")] == [4, 3]


async def test_journaled_runtime_delegates_to_its_store() -> None:
    store = InMemoryJournalStore()
    runtime = JournaledRuntime(store)
    cp = _mk_checkpoint()
    await runtime.put_checkpoint(cp)
    # visible straight through the store — same object graph
    assert await store.get_latest_checkpoint("s1") == cp


# ---------------------------------------------------------------------------
# Postgres — fake-pool routing (same pattern as test_postgres_runtime.py)
# ---------------------------------------------------------------------------


class _FakeState:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.queried: list[tuple[str, tuple[Any, ...]]] = []
        self.next_row: Any = None
        self.next_rows: list[Any] = []


class _FakeConn:
    def __init__(self, state: _FakeState) -> None:
        self._state = state

    async def execute(self, sql: str, *args: Any) -> str:
        self._state.executed.append((sql, args))
        return "OK"

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self._state.queried.append((sql, args))
        return self._state.next_row

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._state.queried.append((sql, args))
        return self._state.next_rows


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


def test_pg_schema_includes_checkpoints_table_and_index() -> None:
    joined = "\n".join(PostgresJournalStore.schema_sql())
    assert "CREATE TABLE IF NOT EXISTS checkpoints" in joined
    assert "JSONB" in joined
    assert "session_id, created_at DESC" in joined


async def test_pg_put_checkpoint_upserts_json_and_prunes() -> None:
    state = _FakeState()
    store = PostgresJournalStore(
        pool=_FakePool(state), max_checkpoints_per_session=7
    )
    cp = _mk_checkpoint(turn=2, at=_t(5))
    await store.put_checkpoint(cp)

    insert_sql, insert_args = state.executed[0]
    assert "INSERT INTO checkpoints" in insert_sql
    assert "ON CONFLICT (checkpoint_id) DO UPDATE" in insert_sql
    assert insert_args[0] == "s1"
    assert insert_args[1] == cp.checkpoint_id
    assert insert_args[2] == 2
    assert insert_args[3] == _t(5).timestamp()
    # payload is a JSON string, not pickle bytes
    assert isinstance(insert_args[4], str)
    assert json.loads(insert_args[4])["cursor"] == "node-2"

    retention_sql, retention_args = state.executed[1]
    assert "DELETE FROM checkpoints" in retention_sql
    assert "ORDER BY created_at DESC" in retention_sql
    assert retention_args == ("s1", 7)


async def test_pg_get_checkpoint_parses_json_payload() -> None:
    cp = _mk_checkpoint(turn=4, at=_t(1))
    state = _FakeState()
    state.next_row = {"payload": cp.model_dump_json()}
    store = PostgresJournalStore(pool=_FakePool(state))

    loaded = await store.get_checkpoint("s1", cp.checkpoint_id)
    assert loaded == cp
    sql, args = state.queried[0]
    assert "WHERE session_id = $1 AND checkpoint_id = $2" in sql
    assert args == ("s1", cp.checkpoint_id)


async def test_pg_get_latest_orders_newest_first() -> None:
    cp = _mk_checkpoint(turn=6, at=_t(2))
    state = _FakeState()
    state.next_row = {"payload": cp.model_dump_json()}
    store = PostgresJournalStore(pool=_FakePool(state))

    latest = await store.get_latest_checkpoint("s1")
    assert latest == cp
    sql, _ = state.queried[0]
    assert "ORDER BY created_at DESC, checkpoint_id DESC LIMIT 1" in sql


async def test_pg_list_checkpoints_returns_meta() -> None:
    state = _FakeState()
    state.next_rows = [
        {"checkpoint_id": "ckpt_b", "turn": 2, "created_at": _t(2).timestamp()},
        {"checkpoint_id": "ckpt_a", "turn": 1, "created_at": _t(1).timestamp()},
    ]
    store = PostgresJournalStore(pool=_FakePool(state))

    metas = await store.list_checkpoints("s1", limit=10)
    assert [m.checkpoint_id for m in metas] == ["ckpt_b", "ckpt_a"]
    assert metas[0].turn == 2
    assert metas[0].created_at == _t(2)
    sql, args = state.queried[0]
    assert "SELECT checkpoint_id, turn, created_at FROM checkpoints" in sql
    assert args == ("s1", 10)


async def test_pg_prune_covers_checkpoints() -> None:
    state = _FakeState()
    store = PostgresJournalStore(pool=_FakePool(state))
    await store.prune(session_id="s1")
    sqls = [s for s, _ in state.executed]
    assert any("DELETE FROM checkpoints" in s for s in sqls)
