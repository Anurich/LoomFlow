"""Regression tests for the runtime review fixes.

Covers:

* Journal keys fingerprint step inputs — same session + step name with
  DIFFERENT inputs re-executes instead of replaying a stale answer.
* ``idempotency_key`` is honoured: it replaces the argument hash so
  retried tool calls that differ only in per-attempt metadata dedupe.
* Stream journaling honesty — abandoned / failing streams are never
  recorded as complete.
* SQLite store: long-lived WAL connection with busy_timeout; a real
  ``:memory:`` journal; safe concurrent use.
* ``JournalStore.prune`` on the in-memory, SQLite and Postgres stores.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow import Agent
from loomflow.core.types import Message, Role
from loomflow.runtime import (
    InMemoryJournalStore,
    JournaledRuntime,
    PostgresJournalStore,
    SqliteRuntime,
)
from loomflow.runtime.journal import SqliteJournalStore

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# 1. Input fingerprint in the journal key
# ---------------------------------------------------------------------------


async def test_step_same_inputs_replay_different_inputs_reexecute() -> None:
    calls: list[tuple[int, int]] = []

    async def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    runtime = JournaledRuntime(InMemoryJournalStore())
    async with runtime.session("s1"):
        assert await runtime.step("add", add, 1, 2) == 3
        assert await runtime.step("add", add, 1, 2) == 3  # cache hit
        assert await runtime.step("add", add, 7, 8) == 15  # cache miss
    assert calls == [(1, 2), (7, 8)]


async def test_step_kwargs_participate_in_fingerprint() -> None:
    calls = {"count": 0}

    async def fn(*, mode: str) -> str:
        calls["count"] += 1
        return mode

    runtime = JournaledRuntime(InMemoryJournalStore())
    async with runtime.session("s1"):
        assert await runtime.step("k", fn, mode="fast") == "fast"
        assert await runtime.step("k", fn, mode="slow") == "slow"
        assert await runtime.step("k", fn, mode="fast") == "fast"
    assert calls["count"] == 2


async def test_pydantic_args_fingerprint_is_stable_across_instances() -> None:
    """Two separately-constructed but equal Message lists must hash to
    the same fingerprint — this is what makes crash-resume replay work
    when the seed messages are rebuilt in a new process."""

    calls = {"count": 0}

    async def model_call(messages: list[Message]) -> str:
        calls["count"] += 1
        return messages[-1].content

    def build_messages(prompt: str) -> list[Message]:
        return [
            Message(role=Role.SYSTEM, content="be brief"),
            Message(role=Role.USER, content=prompt),
        ]

    runtime = JournaledRuntime(InMemoryJournalStore())
    async with runtime.session("s1"):
        first = await runtime.step(
            "model_call_0", model_call, build_messages("hello")
        )
        # Fresh-but-equal objects: cache hit.
        replay = await runtime.step(
            "model_call_0", model_call, build_messages("hello")
        )
        # Different prompt: cache miss.
        fresh = await runtime.step(
            "model_call_0", model_call, build_messages("goodbye")
        )
    assert first == replay == "hello"
    assert fresh == "goodbye"
    assert calls["count"] == 2


async def test_stream_step_different_inputs_reexecute() -> None:
    runs = {"count": 0}

    async def gen(base: int) -> AsyncIterator[int]:
        runs["count"] += 1
        yield base
        yield base + 1

    runtime = JournaledRuntime(InMemoryJournalStore())
    async with runtime.session("s1"):
        first = [c async for c in runtime.stream_step("g", gen, 10)]
        replay = [c async for c in runtime.stream_step("g", gen, 10)]
        fresh = [c async for c in runtime.stream_step("g", gen, 99)]
    assert first == replay == [10, 11]
    assert fresh == [99, 100]
    assert runs["count"] == 2


async def test_agent_rerun_with_new_prompt_does_not_replay_old_answer() -> None:
    """End-to-end regression for the headline bug: calling run() again
    with the same session_id but a DIFFERENT prompt must not replay the
    first run's answer verbatim."""

    runtime = JournaledRuntime(InMemoryJournalStore())
    agent = Agent("hi", model="echo", runtime=runtime)

    r1 = await agent.run("alpha", session_id="reused")
    assert "alpha" in r1.output

    # Pre-fix, this replayed "Echo: alpha" verbatim from the journal.
    r2 = await agent.run("beta", session_id="reused")
    assert "beta" in r2.output
    assert r2.output != r1.output


# ---------------------------------------------------------------------------
# 2. idempotency_key wiring
# ---------------------------------------------------------------------------


async def test_idempotency_key_dedupes_across_differing_args() -> None:
    """When provided, idempotency_key replaces the argument hash — a
    retried tool call with a fresh per-attempt call_id still dedupes
    onto the original journal entry."""

    calls = {"count": 0}

    async def tool_call(tool: str, args: dict[str, Any], *, call_id: str) -> str:
        calls["count"] += 1
        return f"{tool}({args})#{calls['count']}"

    runtime = JournaledRuntime(InMemoryJournalStore())
    async with runtime.session("s1"):
        first = await runtime.step(
            "tool_call_1_0",
            tool_call,
            "search",
            {"q": "x"},
            call_id="attempt-1",
            idempotency_key="K1",
        )
        retried = await runtime.step(
            "tool_call_1_0",
            tool_call,
            "search",
            {"q": "x"},
            call_id="attempt-2",  # different args — same idempotency key
            idempotency_key="K1",
        )
    assert first == retried
    assert calls["count"] == 1


async def test_different_idempotency_key_reexecutes() -> None:
    calls = {"count": 0}

    async def fn() -> int:
        calls["count"] += 1
        return calls["count"]

    runtime = JournaledRuntime(InMemoryJournalStore())
    async with runtime.session("s1"):
        a = await runtime.step("t", fn, idempotency_key="K1")
        b = await runtime.step("t", fn, idempotency_key="K2")
    assert (a, b) == (1, 2)
    assert calls["count"] == 2


# ---------------------------------------------------------------------------
# 3. Stream journaling honesty
# ---------------------------------------------------------------------------


async def test_abandoned_stream_is_not_recorded_as_complete() -> None:
    runs = {"count": 0}

    async def gen() -> AsyncIterator[int]:
        runs["count"] += 1
        yield 1
        yield 2
        yield 3

    store = InMemoryJournalStore()
    runtime = JournaledRuntime(store)
    async with runtime.session("s1"):
        agen = runtime.stream_step("g", gen)
        assert await agen.__anext__() == 1
        await agen.aclose()  # consumer walks away mid-stream

        # Nothing journaled: the partial drain must not masquerade as
        # a completed stream.
        assert store.stream_keys() == []

        # A full drain re-executes the producer and records properly.
        chunks = [c async for c in runtime.stream_step("g", gen)]
    assert chunks == [1, 2, 3]
    assert runs["count"] == 2
    assert len(store.stream_keys()) == 1


async def test_failing_stream_is_not_recorded() -> None:
    async def gen() -> AsyncIterator[int]:
        yield 1
        raise RuntimeError("boom")

    store = InMemoryJournalStore()
    runtime = JournaledRuntime(store)
    async with runtime.session("s1"):
        with pytest.raises(RuntimeError, match="boom"):
            _ = [c async for c in runtime.stream_step("g", gen)]
        assert store.stream_keys() == []


# ---------------------------------------------------------------------------
# 4. SQLite store: WAL, busy_timeout, :memory:, concurrency
# ---------------------------------------------------------------------------


async def test_sqlite_store_uses_wal_and_busy_timeout(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")
    await store.put_step("s1", "k", "v")

    # WAL is persistent — visible from an independent connection.
    probe = sqlite3.connect(tmp_path / "j.db")
    try:
        mode = probe.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        probe.close()
    assert mode == "wal"

    # busy_timeout is per-connection; check the store's own.
    timeout = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout == 5000
    await store.aclose()


async def test_sqlite_memory_journal_actually_persists_within_store() -> None:
    """``:memory:`` used to open a fresh (empty) database per operation;
    the long-lived connection makes the ephemeral journal real."""
    runtime = SqliteRuntime(":memory:")
    calls = {"count": 0}

    async def fn() -> int:
        calls["count"] += 1
        return calls["count"]

    async with runtime.session("s1"):
        a = await runtime.step("k", fn)
        b = await runtime.step("k", fn)
    assert (a, b) == (1, 1)
    assert calls["count"] == 1


async def test_sqlite_store_concurrent_writes_are_serialized(
    tmp_path: Path,
) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")

    async def put(i: int) -> None:
        await store.put_step("s1", f"k{i}", i)

    async with anyio.create_task_group() as tg:
        for i in range(20):
            tg.start_soon(put, i)

    for i in range(20):
        entry = await store.get_step("s1", f"k{i}")
        assert entry is not None
        assert entry.value == i
    await store.aclose()


async def test_sqlite_two_stores_share_one_file(tmp_path: Path) -> None:
    """Two live store instances (≈ two concurrent runs) coordinate on
    the same file instead of erroring — the docstring's promise."""
    a = SqliteJournalStore(tmp_path / "j.db")
    b = SqliteJournalStore(tmp_path / "j.db")
    await a.put_step("s1", "from_a", 1)
    await b.put_step("s2", "from_b", 2)
    entry = await b.get_step("s1", "from_a")
    assert entry is not None
    assert entry.value == 1
    await a.aclose()
    await b.aclose()


async def test_sqlite_aclose_is_idempotent(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")
    await store.aclose()
    await store.aclose()  # no ProgrammingError on double close


# ---------------------------------------------------------------------------
# 5. prune()
# ---------------------------------------------------------------------------


async def test_inmemory_prune_by_session() -> None:
    store = InMemoryJournalStore()
    await store.put_step("keep", "k", 1)
    await store.put_step("drop", "k", 2)
    await store.put_stream("keep", "g", ["a"])
    await store.put_stream("drop", "g", ["b"])

    await store.prune(session_id="drop")

    assert await store.get_step("keep", "k") is not None
    assert await store.get_step("drop", "k") is None
    assert await store.get_stream("keep", "g") == ["a"]
    assert await store.get_stream("drop", "g") is None


async def test_inmemory_prune_by_before_timestamp() -> None:
    store = InMemoryJournalStore()
    await store.put_step("s1", "k", 1)

    # Cutoff in the past: nothing qualifies.
    await store.prune(before=datetime.now() - timedelta(hours=1))
    assert await store.get_step("s1", "k") is not None

    # Cutoff in the future: everything qualifies.
    await store.prune(before=datetime.now() + timedelta(hours=1))
    assert await store.get_step("s1", "k") is None


async def test_inmemory_prune_without_filters_purges_everything() -> None:
    store = InMemoryJournalStore()
    await store.put_step("s1", "k", 1)
    await store.put_stream("s2", "g", ["a"])
    await store.prune()
    assert store.step_keys() == []
    assert store.stream_keys() == []


async def test_sqlite_prune_by_session_and_time(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")
    await store.put_step("keep", "k", 1)
    await store.put_step("drop", "k", 2)
    await store.put_stream("drop", "g", ["x"])

    await store.prune(session_id="drop")
    assert await store.get_step("keep", "k") is not None
    assert await store.get_step("drop", "k") is None
    assert await store.get_stream("drop", "g") is None

    # Time-based purge of the remainder.
    await store.prune(before=datetime.now() + timedelta(hours=1))
    assert await store.get_step("keep", "k") is None
    await store.aclose()


# ---- Postgres prune: SQL emission via a fake asyncpg pool ---------------


class _FakeState:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []


class _FakeConn:
    def __init__(self, state: _FakeState) -> None:
        self._state = state

    async def execute(self, sql: str, *args: Any) -> str:
        self._state.executed.append((sql, args))
        return "OK"


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


async def test_postgres_prune_emits_filtered_deletes() -> None:
    state = _FakeState()
    store = PostgresJournalStore(pool=_FakePool(state))
    cutoff = datetime.now()

    await store.prune(before=cutoff, session_id="s1")

    sqls = [s for s, _ in state.executed]
    assert any("DELETE FROM journal_steps" in s for s in sqls)
    assert any("DELETE FROM journal_streams" in s for s in sqls)
    for sql, args in state.executed:
        assert "created_at < $1" in sql
        assert "session_id = $2" in sql
        assert args == (cutoff.timestamp(), "s1")


async def test_postgres_prune_without_filters_deletes_all() -> None:
    state = _FakeState()
    store = PostgresJournalStore(pool=_FakePool(state))
    await store.prune()
    assert state.executed == [
        ("DELETE FROM journal_steps", ()),
        ("DELETE FROM journal_streams", ()),
    ]
