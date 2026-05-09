"""SQLite-backed runtime tests.

The killer property: a journal written by one process can be replayed
by a fresh runtime instance pointing at the same DB file. We simulate
"process restart" by closing one runtime and constructing another
against the same path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from loomflow import Agent
from loomflow.runtime import SqliteRuntime
from loomflow.runtime.journal import SqliteJournalStore

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# SqliteJournalStore basics
# ---------------------------------------------------------------------------


async def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")
    await store.put_step("s1", "step1", {"foo": 42, "list": [1, 2, 3]})
    entry = await store.get_step("s1", "step1")
    assert entry is not None
    assert entry.value == {"foo": 42, "list": [1, 2, 3]}


async def test_sqlite_store_returns_none_for_unknown_key(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")
    assert await store.get_step("s1", "missing") is None
    assert await store.get_stream("s1", "missing") is None


async def test_sqlite_store_replaces_existing_value(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")
    await store.put_step("s1", "k", "first")
    await store.put_step("s1", "k", "second")
    entry = await store.get_step("s1", "k")
    assert entry is not None
    assert entry.value == "second"


async def test_sqlite_store_streams_roundtrip(tmp_path: Path) -> None:
    store = SqliteJournalStore(tmp_path / "j.db")
    await store.put_stream("s1", "g", ["a", "b", "c"])
    chunks = await store.get_stream("s1", "g")
    assert chunks == ["a", "b", "c"]


async def test_sqlite_store_creates_parent_directory(tmp_path: Path) -> None:
    deep = tmp_path / "nested" / "deep" / "j.db"
    SqliteJournalStore(deep)
    assert deep.parent.exists()


# ---------------------------------------------------------------------------
# Cross-instance persistence
# ---------------------------------------------------------------------------


async def test_journal_persists_across_runtime_instances(tmp_path: Path) -> None:
    counter = {"calls": 0}

    async def expensive() -> str:
        counter["calls"] += 1
        return f"v{counter['calls']}"

    db = tmp_path / "journal.db"

    rt1 = SqliteRuntime(db)
    async with rt1.session("fixed-id"):
        first = await rt1.step("step1", expensive)

    # Simulate a process restart: new SqliteRuntime against the same
    # file, same session id ⇒ replays the cached value.
    rt2 = SqliteRuntime(db)
    async with rt2.session("fixed-id"):
        second = await rt2.step("step1", expensive)

    assert first == "v1"
    assert second == "v1"
    assert counter["calls"] == 1


async def test_stream_journal_persists_across_runtime_instances(
    tmp_path: Path,
) -> None:
    runs = {"count": 0}

    async def gen() -> AsyncIterator[str]:
        runs["count"] += 1
        for x in ("a", "b", "c"):
            yield x

    db = tmp_path / "journal.db"

    rt1 = SqliteRuntime(db)
    async with rt1.session("s1"):
        first = [c async for c in rt1.stream_step("stream1", gen)]

    rt2 = SqliteRuntime(db)
    async with rt2.session("s1"):
        second = [c async for c in rt2.stream_step("stream1", gen)]

    assert first == ["a", "b", "c"]
    assert second == ["a", "b", "c"]
    assert runs["count"] == 1


async def test_concurrent_sessions_do_not_collide(tmp_path: Path) -> None:
    """Two distinct session_ids in the same DB file have independent
    rows for the same step name."""
    counter = {"calls": 0}

    async def fn() -> int:
        counter["calls"] += 1
        return counter["calls"]

    rt = SqliteRuntime(tmp_path / "j.db")
    async with rt.session("alpha"):
        a = await rt.step("k", fn)
    async with rt.session("beta"):
        b = await rt.step("k", fn)

    assert a == 1
    assert b == 2


async def test_runtime_path_property(tmp_path: Path) -> None:
    rt = SqliteRuntime(tmp_path / "subdir" / "j.db")
    assert rt.path == (tmp_path / "subdir" / "j.db")


# ---------------------------------------------------------------------------
# Agent + SqliteRuntime
# ---------------------------------------------------------------------------


async def test_agent_run_with_sqlite_runtime_writes_journal(
    tmp_path: Path,
) -> None:
    db = tmp_path / "j.db"
    rt = SqliteRuntime(db)
    agent = Agent("hi", model="echo", runtime=rt)

    result = await agent.run("hello")

    # The journal should now have entries for this session.
    entry = await rt.store.get_step(
        result.session_id, "persist_episode_1"
    )
    # The persisted value is the episode id (a string from
    # InMemoryMemory.remember).
    assert entry is not None
    assert isinstance(entry.value, str)
    assert entry.value.startswith("ep_")
