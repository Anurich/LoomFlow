"""Abandoning a stream must not break the caller.

``async for ev in agent.stream(...): break`` is the documented way to
stop early. The stream used to yield from inside a task group, so the
group's cancel scope was later torn down from another task and the
caller's next, unrelated ``await`` raised ``CancelledError``.
"""

from __future__ import annotations

import gc
import importlib.util
from contextlib import aclosing
from typing import Any

import anyio
import pytest

from loomflow import END, Agent, ScriptedModel, ScriptedTurn, Workflow, tool
from loomflow.architecture import MultiAgentDebate, Supervisor
from loomflow.core.types import EventKind, ToolCall

pytestmark = pytest.mark.anyio


def _slow_tool(state: dict[str, str]) -> Any:
    @tool
    async def slow() -> str:
        """Slow tool."""
        try:
            await anyio.sleep(5)
        except BaseException:
            state["slow"] = "cancelled"
            raise
        state["slow"] = "finished"
        return "done"

    return slow


def _calls_slow(final: str = "final") -> ScriptedModel:
    return ScriptedModel(
        [
            ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="slow", args={})]),
            ScriptedTurn(text=final),
        ]
    )


async def _caller_still_works() -> None:
    await anyio.sleep(0.05)
    result = await Agent("t", model="echo").run("still alive?")
    assert "still alive?" in result.output


async def _wait_for(state: dict[str, str], value: str) -> None:
    with anyio.fail_after(2):
        while state.get("slow") != value:
            gc.collect()
            await anyio.sleep(0.01)


async def test_agent_stream_break_without_aclose_leaves_caller_usable() -> None:
    state: dict[str, str] = {}
    agent = Agent("t", model=_calls_slow(), tools=[_slow_tool(state)])
    async for ev in agent.stream("go"):
        if ev.kind == EventKind.TOOL_CALL:
            break
    await _caller_still_works()
    # The abandoned run is cancelled, not left running in the background.
    await _wait_for(state, "cancelled")


async def test_agent_stream_aclose_cancels_the_inflight_tool_first() -> None:
    state: dict[str, str] = {}
    agent = Agent("t", model=_calls_slow(), tools=[_slow_tool(state)])
    async with aclosing(agent.stream("go")) as events:
        async for ev in events:
            if ev.kind == EventKind.TOOL_CALL:
                break
    assert state.get("slow") == "cancelled"
    await _caller_still_works()


async def test_agent_stream_reraises_the_raw_error() -> None:
    class Exploding(ScriptedModel):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("model exploded")
            yield  # pragma: no cover

    agent = Agent("t", model=Exploding([]))
    with pytest.raises(ValueError, match="model exploded"):
        async for _ in agent.stream("go"):
            pass


@pytest.mark.parametrize("shape", ["chain", "fanout"])
async def test_workflow_stream_break_leaves_caller_usable(shape: str) -> None:
    async def fast(x: Any) -> Any:
        return x

    async def slow(x: Any) -> Any:
        await anyio.sleep(5)
        return x

    if shape == "chain":
        wf = Workflow.chain([fast, slow], name="c")
        stop_at = "slow"
    else:
        wf = Workflow("f")
        wf.add_node("S", fast)
        wf.add_node("B", slow)
        wf.add_node("C", slow)
        wf.set_start("S")
        wf.add_edge("S", "B")
        wf.add_edge("S", "C")
        wf.add_edge("B", END)
        wf.add_edge("C", END)
        stop_at = "B"
    async for ev in wf.stream("x"):
        if ev.kind == EventKind.WORKFLOW_STEP_STARTED and ev.payload["node"] == stop_at:
            break
    await _caller_still_works()


async def test_supervisor_stream_break_cancels_the_worker() -> None:
    state: dict[str, str] = {}
    worker = Agent("worker", model=_calls_slow(), tools=[_slow_tool(state)])
    manager = Agent(
        "manager",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="d1",
                            tool="delegate",
                            args={"worker": "w", "instructions": "go"},
                        )
                    ]
                ),
                ScriptedTurn(text="done"),
            ]
        ),
        architecture=Supervisor(workers={"w": worker}),
    )
    async with aclosing(manager.stream("go")) as events:
        async for ev in events:
            if ev.kind == EventKind.TOOL_CALL and ev.payload["call"]["tool"] == "slow":
                break
    assert state.get("slow") == "cancelled"
    await _caller_still_works()


async def test_debate_stream_break_leaves_caller_usable() -> None:
    state: dict[str, str] = {}
    debaters = [
        Agent("a", model=_calls_slow("42"), tools=[_slow_tool(state)]),
        Agent("b", model=_calls_slow("42"), tools=[_slow_tool(state)]),
    ]
    agent = Agent(
        "debate",
        model=ScriptedModel([ScriptedTurn(text="unused")]),
        architecture=MultiAgentDebate(debaters=debaters, rounds=1),
    )
    async for ev in agent.stream("6*7?"):
        if ev.kind == EventKind.TOOL_CALL:
            break
    await _caller_still_works()
    await _wait_for(state, "cancelled")


@pytest.mark.parametrize(
    "anyio_backend",
    [
        pytest.param(
            "trio",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("trio") is None, reason="trio not installed"
            ),
        )
    ],
)
async def test_trio_stream_closed_early_with_aclosing() -> None:
    # trio has no unstructured spawn, so the stream keeps a task group;
    # closing it early must not surface GeneratorExit in an exception group.
    state: dict[str, str] = {}
    agent = Agent("t", model=_calls_slow(), tools=[_slow_tool(state)])
    async with aclosing(agent.stream("go")) as events:
        async for ev in events:
            if ev.kind == EventKind.TOOL_CALL:
                break
    assert state.get("slow") == "cancelled"
    await _caller_still_works()
