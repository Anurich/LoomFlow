"""Regression tests for the Workflow validation findings.

Each test reproduces a finding from the validation pass and fails on
the code before its fix.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, tool
from loomflow.core.context import get_run_context
from loomflow.core.types import EventKind, ToolCall
from loomflow.security.audit import InMemoryAuditLog
from loomflow.workflow import END, Workflow, step

pytestmark = pytest.mark.anyio


async def _echo(x: Any) -> Any:
    return x


async def test_edges_to_unregistered_nodes_fail_before_running() -> None:
    ran: list[str] = []

    async def a(x: Any) -> str:
        ran.append("A")
        return "a"

    wf = Workflow("typo")
    wf.add_node("A", a)
    wf.set_start("A")
    wf.add_edge("A", "Bee")
    with pytest.raises(ValueError, match="A -> Bee"):
        await wf.run("x")
    assert ran == []

    routed = Workflow("typo-router")
    routed.add_node("A", a)
    routed.set_start("A")
    routed.add_router("A", lambda v: "x", {"x": "missing"})
    with pytest.raises(ValueError, match="A -> missing"):
        await routed.run("x")


def _loop_with_fan_out(order: list[str], stamps: dict[str, list[tuple[float, float]]], joins: list[Any]) -> Workflow:
    iterations = {"n": 0}

    async def plan(x: Any) -> str:
        iterations["n"] += 1
        return "q"

    def branch(name: str) -> Any:
        async def run(x: Any) -> str:
            start = anyio.current_time()
            await anyio.sleep(0.1)
            stamps.setdefault(name, []).append((start, anyio.current_time()))
            return name

        return run

    async def synth(x: Any) -> str:
        joins.append(x)
        return "done" if iterations["n"] >= 2 else "again"

    nodes = {"plan": plan, "r1": branch("r1"), "r2": branch("r2"), "r2b": branch("r2b"), "synth": synth}
    wf = Workflow("loop", max_steps=50)
    for name in order:
        wf.add_node(name, nodes[name])
    wf.set_start("plan")
    wf.add_edge("plan", "r1")
    wf.add_edge("plan", "r2")
    wf.add_edge("r2", "r2b")
    wf.add_edge("r1", "synth")
    wf.add_edge("r2b", "synth")
    wf.add_router("synth", lambda v: v, {"again": "plan", "done": END})
    return wf


@pytest.mark.parametrize(
    "order", [["plan", "r1", "r2", "r2b", "synth"], ["plan", "synth", "r1", "r2", "r2b"]]
)
async def test_fan_out_inside_a_loop_runs_concurrently_and_joins_once(order: list[str]) -> None:
    stamps: dict[str, list[tuple[float, float]]] = {}
    joins: list[Any] = []
    await _loop_with_fan_out(order, stamps, joins).run("x")
    assert joins == [["r1", "r2b"], ["r1", "r2b"]]
    (_, r1_end), (r2_start, _) = stamps["r1"][0], stamps["r2"][0]
    assert r2_start < r1_end  # r1 and r2 overlapped


@pytest.mark.parametrize("delays", [(0.05, 0.1), (0.1, 0.05)])
async def test_output_of_separate_end_branches_does_not_depend_on_timing(
    delays: tuple[float, float],
) -> None:
    def after(delay: float, value: str) -> Any:
        async def run(x: Any) -> str:
            await anyio.sleep(delay)
            return value

        return run

    wf = Workflow("ends")
    wf.add_node("A", _echo)
    wf.add_node("B", after(delays[0], "B"))
    wf.add_node("C", after(delays[1], "C"))
    wf.set_start("A")
    wf.add_edge("A", "B")
    wf.add_edge("A", "C")
    assert (await wf.run("x")).output == "C"


async def test_a_dead_end_branch_does_not_become_the_output() -> None:
    async def slow_dead_end(x: Any) -> str:
        await anyio.sleep(0.1)
        return "dead end"

    wf = Workflow("p6")
    for name, fn in {"A": _echo, "B": slow_dead_end, "C": _echo, "D": _echo}.items():
        wf.add_node(name, fn)
    wf.set_start("A")
    wf.add_edge("A", "B")
    wf.add_edge("A", "C")
    wf.add_edge("C", "D")
    wf.add_router("B", lambda v: "stop", {"stop": END})
    assert (await wf.run("x")).output == "x"


async def test_a_failing_node_emits_an_error_event() -> None:
    async def boom(x: Any) -> str:
        raise ValueError("branch exploded")

    wf = Workflow.chain([_echo, boom])
    kinds: list[EventKind] = []
    with pytest.raises(ValueError, match="branch exploded"):
        async for ev in wf.stream("x"):
            kinds.append(ev.kind)
    assert kinds[-1] == EventKind.ERROR


async def test_parallel_raises_the_branch_exception_itself() -> None:
    async def bad(x: Any) -> str:
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await Workflow.parallel([_echo, bad]).run("x")


async def test_parallel_max_concurrency() -> None:
    state = {"now": 0, "peak": 0}

    async def branch(x: Any) -> int:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await anyio.sleep(0.02)
        state["now"] -= 1
        return 0

    await Workflow.parallel([branch] * 6, max_concurrency=2).run("x")
    assert state["peak"] == 2


async def test_nested_workflow_sees_the_outer_run_context() -> None:
    seen: list[Any] = []

    async def inner_step(x: Any) -> Any:
        ctx = get_run_context()
        seen.append((ctx.user_id, ctx.session_id, ctx.metadata.get("tenant")))
        return x

    outer = Workflow.chain([Workflow.chain([inner_step], name="inner")], name="outer")
    await outer.run("x", user_id="alice", session_id="s-1", metadata={"tenant": "acme"})
    assert seen == [("alice", "s-1", "acme")]


async def test_agent_node_tools_see_the_workflow_metadata() -> None:
    seen: list[Any] = []

    @tool
    async def probe() -> str:
        """Probe."""
        ctx = get_run_context()
        seen.append((ctx.user_id, ctx.metadata.get("tenant")))
        return "ok"

    agent = Agent(
        "x",
        model=ScriptedModel([ScriptedTurn(tool_calls=[ToolCall(tool="probe", args={})]), ScriptedTurn(text="done")]),
        tools=[probe],
    )
    await Workflow.chain([agent]).run("go", user_id="alice", metadata={"tenant": "acme"})
    assert seen == [("alice", "acme")]


async def test_step_decorator_writes_audit_entries_for_the_step() -> None:
    log = InMemoryAuditLog(secret="k")

    @step
    async def classify(x: Any) -> str:
        return "label"

    async def node(x: Any) -> str:
        return await classify(x)

    await Workflow.chain([node], audit_log=log).run("x", session_id="s")
    steps = [e.payload.get("step") for e in await log.all_entries() if "step" in e.payload]
    assert steps == ["classify", "classify"]


async def test_an_interrupted_agent_node_stops_the_run() -> None:
    @tool
    async def look() -> str:
        """Look."""
        return "nothing"

    stuck = Agent(
        "stuck",
        model=ScriptedModel([ScriptedTurn(tool_calls=[ToolCall(tool="look", args={})]) for _ in range(5)]),
        tools=[look],
        max_turns=1,
    )
    received: list[Any] = []

    async def publish(x: Any) -> str:
        received.append(x)
        return "published"

    async def draft(x: Any) -> str:
        return "draft0"

    result = await Workflow.chain([draft, stuck, publish]).run("go")
    assert received == []
    assert result.interrupted
    assert result.interruption_reason is not None and "max_turns" in result.interruption_reason
    assert result.output == "draft0"
