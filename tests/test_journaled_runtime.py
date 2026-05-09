"""JournaledRuntime tests — replay correctness, session isolation,
parallel-task contextvar propagation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from loomflow import Agent, tool
from loomflow.core.types import ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.runtime import InMemoryJournalStore, JournaledRuntime

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# step() replay
# ---------------------------------------------------------------------------


async def test_step_runs_once_per_session_and_replays_thereafter() -> None:
    counter = {"calls": 0}

    async def increment() -> int:
        counter["calls"] += 1
        return counter["calls"]

    runtime = JournaledRuntime(InMemoryJournalStore())

    async with runtime.session("s1"):
        first = await runtime.step("inc", increment)
        second = await runtime.step("inc", increment)

    assert first == 1
    assert second == 1
    assert counter["calls"] == 1


async def test_step_outside_session_runs_every_time() -> None:
    counter = {"calls": 0}

    async def increment() -> int:
        counter["calls"] += 1
        return counter["calls"]

    runtime = JournaledRuntime(InMemoryJournalStore())

    # No `runtime.session(...)` context: journal is bypassed.
    a = await runtime.step("inc", increment)
    b = await runtime.step("inc", increment)
    assert a == 1
    assert b == 2


async def test_different_sessions_are_isolated() -> None:
    counter = {"calls": 0}

    async def increment() -> int:
        counter["calls"] += 1
        return counter["calls"]

    runtime = JournaledRuntime(InMemoryJournalStore())

    async with runtime.session("alpha"):
        a = await runtime.step("inc", increment)
    async with runtime.session("beta"):
        b = await runtime.step("inc", increment)

    assert a == 1
    assert b == 2  # different session ⇒ executed again


async def test_step_with_arguments_caches_per_step_name() -> None:
    """Cached value depends on step_name, not the args. Same name + same
    session ⇒ second call returns the first call's result regardless of
    what args you pass."""

    async def add(a: int, b: int) -> int:
        return a + b

    runtime = JournaledRuntime(InMemoryJournalStore())
    async with runtime.session("s1"):
        first = await runtime.step("add", add, 2, 3)
        # Different args, same step_name: replay returns cached.
        second = await runtime.step("add", add, 100, 200)
    assert first == 5
    assert second == 5


# ---------------------------------------------------------------------------
# stream_step() replay
# ---------------------------------------------------------------------------


async def test_stream_step_replays_chunks_without_re_executing() -> None:
    call_count = {"runs": 0}

    async def gen() -> AsyncIterator[int]:
        call_count["runs"] += 1
        for x in (1, 2, 3):
            yield x

    runtime = JournaledRuntime(InMemoryJournalStore())

    async with runtime.session("s1"):
        first = [c async for c in runtime.stream_step("g", gen)]
        second = [c async for c in runtime.stream_step("g", gen)]

    assert first == [1, 2, 3]
    assert second == [1, 2, 3]
    assert call_count["runs"] == 1  # underlying generator only ran once


async def test_stream_step_outside_session_runs_each_time() -> None:
    call_count = {"runs": 0}

    async def gen() -> AsyncIterator[int]:
        call_count["runs"] += 1
        yield 42

    runtime = JournaledRuntime(InMemoryJournalStore())
    [_ async for _ in runtime.stream_step("g", gen)]
    [_ async for _ in runtime.stream_step("g", gen)]
    assert call_count["runs"] == 2


# ---------------------------------------------------------------------------
# contextvar propagation across spawned tasks
# ---------------------------------------------------------------------------


async def test_contextvar_propagates_into_spawned_tasks() -> None:
    """anyio task groups must inherit the runtime session ContextVar so
    parallel tool dispatches inside ``_dispatch_tools`` get the right
    cache lookups."""

    counter = {"calls": 0}

    async def increment(label: str) -> str:
        counter["calls"] += 1
        return f"{label}:{counter['calls']}"

    runtime = JournaledRuntime(InMemoryJournalStore())

    async with runtime.session("s1"):
        results: list[Any] = [None, None]

        async def _spawn(i: int, label: str) -> None:
            results[i] = await runtime.step(f"step_{i}", increment, label)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_spawn, 0, "a")
            tg.start_soon(_spawn, 1, "b")

        # Replay: same session, same step names ⇒ cached.
        again: list[Any] = [None, None]

        async def _spawn_again(i: int, label: str) -> None:
            again[i] = await runtime.step(f"step_{i}", increment, label)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_spawn_again, 0, "X")
            tg.start_soon(_spawn_again, 1, "Y")

    assert results == again  # parallel + replay produced identical values
    assert counter["calls"] == 2  # never executed again on replay


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


async def test_agent_run_journals_episode_persistence() -> None:
    runtime = JournaledRuntime(InMemoryJournalStore())
    agent = Agent("hi", model="echo", runtime=runtime)

    result = await agent.run("hello")

    store = runtime.store
    assert isinstance(store, InMemoryJournalStore)
    keys = store.step_keys()
    # The loop calls runtime.step("persist_episode_<turns>", ...).
    assert any(
        sid == result.session_id and name.startswith("persist_episode_")
        for sid, name in keys
    )


async def test_agent_run_journals_each_tool_call() -> None:
    @tool
    async def ping() -> str:
        """Return pong."""
        return "pong"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="ping", args={})]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    runtime = JournaledRuntime(InMemoryJournalStore())
    agent = Agent("hi", model=model, tools=[ping], runtime=runtime)

    result = await agent.run("ping?")

    store = runtime.store
    assert isinstance(store, InMemoryJournalStore)
    step_names = {n for s, n in store.step_keys() if s == result.session_id}
    # `tool_call_<turn>_<slot>` is the key pattern from the loop.
    assert any(name.startswith("tool_call_") for name in step_names)


async def test_replay_a_full_run_against_same_session_id() -> None:
    """Same session_id forces the journaled replay path. Tool functions
    are wrapped to assert they're never re-executed in replay."""

    actual_calls = {"count": 0}

    @tool
    async def echo(msg: str) -> str:
        """Echo back."""
        actual_calls["count"] += 1
        return f"echoed:{msg}"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="echo", args={"msg": "hi"})
                ]
            ),
            ScriptedTurn(text="thanks"),
        ]
    )
    runtime = JournaledRuntime(InMemoryJournalStore())

    # First run records.
    async with runtime.session("fixed"):
        # Manually wire the loop's step calls — simulating a run.
        result = await runtime.step(
            "tool_call_1_0",
            _call_tool_directly,
            echo,
            {"msg": "hi"},
        )
        assert result == "echoed:hi"
        assert actual_calls["count"] == 1

    # Second run with same session+step name replays the cache.
    async with runtime.session("fixed"):
        replay_result = await runtime.step(
            "tool_call_1_0",
            _call_tool_directly,
            echo,
            {"msg": "different"},  # ignored on replay
        )
        assert replay_result == "echoed:hi"
        assert actual_calls["count"] == 1  # echo NOT called again

    _ = model  # silence unused-import warning when Agent isn't invoked


async def _call_tool_directly(tool_obj: Any, args: dict[str, Any]) -> Any:
    return await tool_obj.execute(args)
