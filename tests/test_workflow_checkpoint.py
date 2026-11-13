"""Opt-in checkpoint / resume for Workflow runs
(``Workflow.run(..., checkpointer=, session_id=)``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.core.types import EventKind, ToolCall, Usage
from loomflow.workflow import (
    END,
    START,
    InMemoryCheckpointer,
    SqliteCheckpointer,
    Workflow,
    WorkflowCheckpointer,
    WorkflowCheckpointError,
)

pytestmark = pytest.mark.anyio


class Order(BaseModel):
    id: str
    total: float


@dataclass
class Receipt:
    order: Order
    reference: str


class _Flaky:
    """Counts calls per node and fails the named ones until told not to."""

    def __init__(self, *failing: str) -> None:
        self.calls: list[str] = []
        self.failing = set(failing)

    def node(self, name: str, fn: Callable[[Any], Any] | None = None, delay: float = 0.0) -> Any:
        async def run(value: Any) -> Any:
            await anyio.sleep(delay)
            self.calls.append(name)
            if name in self.failing:
                raise RuntimeError(f"{name} crashed")
            return fn(value) if fn is not None else f"{name}({value})"

        run.__name__ = name
        return run


def _order_chain(flaky: _Flaky) -> Workflow:
    return Workflow.chain(
        [flaky.node("charge"), flaky.node("ship"), flaky.node("notify")], name="order"
    )


# ---------------------------------------------------------------------------
# The basic promise: completed nodes don't run again
# ---------------------------------------------------------------------------


async def test_without_a_checkpointer_a_rerun_starts_over() -> None:
    flaky = _Flaky("ship")
    wf = _order_chain(flaky)
    with pytest.raises(RuntimeError, match="ship crashed"):
        await wf.run("o1", session_id="order-1")
    flaky.failing.clear()
    result = await wf.run("o1", session_id="order-1")
    assert flaky.calls == ["charge", "ship", "charge", "ship", "notify"]
    assert result.resumed is False


async def test_a_rerun_skips_completed_nodes_and_continues() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("ship")
    wf = _order_chain(flaky)
    with pytest.raises(RuntimeError, match="ship crashed"):
        await wf.run("o1", session_id="order-1", checkpointer=store)
    assert len(store.keys()) == 1

    flaky.failing.clear()
    result = await wf.run("o1", session_id="order-1", checkpointer=store)
    assert flaky.calls == ["charge", "ship", "ship", "notify"]  # charge ran once
    assert result.output == "notify(ship(charge(o1)))"
    assert result.resumed is True
    assert result.visited == ["charge", "ship", "notify"]
    assert result.per_step["charge"] == "charge(o1)"
    assert store.keys() == []  # finished: nothing left behind


async def test_sqlite_checkpoints_survive_a_restart(tmp_path: Path) -> None:
    db = tmp_path / "state" / "wf.db"
    flaky = _Flaky("ship")
    first = SqliteCheckpointer(db)
    with pytest.raises(RuntimeError):
        await _order_chain(flaky).run("o1", session_id="order-1", checkpointer=first)
    await first.aclose()

    # A new process: a new store on the same file, a rebuilt workflow.
    flaky.failing.clear()
    second = SqliteCheckpointer(db)
    assert isinstance(second, WorkflowCheckpointer)
    result = await _order_chain(flaky).run("o1", session_id="order-1", checkpointer=second)
    assert flaky.calls == ["charge", "ship", "ship", "notify"]
    assert result.resumed is True and result.output == "notify(ship(charge(o1)))"
    await second.aclose()


async def test_a_finished_run_is_not_replayed_into_the_next_one() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky()
    wf = _order_chain(flaky)
    await wf.run("o1", session_id="order-1", checkpointer=store)
    result = await wf.run("o1", session_id="order-1", checkpointer=store)
    assert flaky.calls == ["charge", "ship", "notify"] * 2
    assert result.resumed is False


async def test_a_different_input_is_a_new_request() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("ship")
    wf = _order_chain(flaky)
    with pytest.raises(RuntimeError):
        await wf.run("o1", session_id="chat-7", checkpointer=store)
    flaky.failing.clear()
    result = await wf.run("o2", session_id="chat-7", checkpointer=store)
    assert result.resumed is False
    assert result.output == "notify(ship(charge(o2)))"
    assert flaky.calls == ["charge", "ship", "charge", "ship", "notify"]


async def test_users_and_sessions_have_separate_checkpoints() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("ship")
    wf = _order_chain(flaky)
    with pytest.raises(RuntimeError):
        await wf.run("o1", session_id="s", user_id="alice", checkpointer=store)
    flaky.failing.clear()
    other = await wf.run("o1", session_id="s", user_id="bob", checkpointer=store)
    assert other.resumed is False
    mine = await wf.run("o1", session_id="s", user_id="alice", checkpointer=store)
    assert mine.resumed is True


async def test_clear_checkpoint_starts_the_run_over() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("ship")
    wf = _order_chain(flaky)
    with pytest.raises(RuntimeError):
        await wf.run("o1", session_id="order-1", checkpointer=store)
    await wf.clear_checkpoint(store, session_id="order-1")
    flaky.failing.clear()
    result = await wf.run("o1", session_id="order-1", checkpointer=store)
    assert result.resumed is False and flaky.calls.count("charge") == 2


async def test_a_checkpointer_needs_a_session_id() -> None:
    with pytest.raises(ValueError, match="needs a session_id"):
        await _order_chain(_Flaky()).run("o1", checkpointer=InMemoryCheckpointer())


# ---------------------------------------------------------------------------
# Scheduler state: fan-out, joins, loops, routers
# ---------------------------------------------------------------------------


async def test_a_fan_out_resumes_only_the_unfinished_branch() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("slow")
    wf = Workflow("fan")
    wf.add_node("start", flaky.node("start"))
    wf.add_node("slow", flaky.node("slow", delay=0.03))
    wf.add_node("fast", flaky.node("fast"))
    wf.add_node("join", flaky.node("join", fn=lambda v: v))
    wf.set_start("start")
    wf.add_edge("start", "slow").add_edge("start", "fast")
    wf.add_edge("slow", "join").add_edge("fast", "join").add_edge("join", END)
    with pytest.raises(RuntimeError, match="slow crashed"):
        await wf.run("x", session_id="s", checkpointer=store)
    assert flaky.calls == ["start", "fast", "slow"]

    flaky.failing.clear()
    result = await wf.run("x", session_id="s", checkpointer=store)
    assert flaky.calls == ["start", "fast", "slow", "slow", "join"]
    # The join still gets both inputs, in edge declaration order.
    assert result.output == ["slow(start(x))", "fast(start(x))"]
    assert result.visited == ["start", "fast", "slow", "join"]


async def test_a_sibling_cancelled_by_a_failure_runs_again() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("boom")
    wf = Workflow("fan")
    wf.add_node("start", flaky.node("start"))
    wf.add_node("boom", flaky.node("boom"))
    wf.add_node("slow", flaky.node("slow", delay=0.05))
    wf.set_start("start")
    wf.add_edge("start", "boom").add_edge("start", "slow")
    wf.add_edge("boom", END).add_edge("slow", END)
    with pytest.raises(RuntimeError):
        await wf.run("x", session_id="s", checkpointer=store)
    assert flaky.calls == ["start", "boom"]  # slow was cancelled mid-sleep
    flaky.failing.clear()
    await wf.run("x", session_id="s", checkpointer=store)
    assert sorted(flaky.calls[2:]) == ["boom", "slow"]


async def test_a_loop_resumes_mid_iteration_with_its_counters() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky()
    state = {"fail_at": 3}

    async def bump(value: int) -> int:
        flaky.calls.append(f"bump{value}")
        if value + 1 == state["fail_at"]:
            raise RuntimeError("crash in the loop")
        return value + 1

    wf = Workflow("loop", max_visits_per_node=5)
    wf.add_node("bump", bump)
    wf.set_start("bump")
    wf.add_router("bump", lambda v: "again" if v < 5 else "done", {"again": "bump", "done": END})
    with pytest.raises(RuntimeError, match="crash in the loop"):
        await wf.run(0, session_id="s", checkpointer=store)
    assert flaky.calls == ["bump0", "bump1", "bump2"]

    state["fail_at"] = -1
    result = await wf.run(0, session_id="s", checkpointer=store)
    assert flaky.calls == ["bump0", "bump1", "bump2", "bump2", "bump3", "bump4"]
    assert result.output == 5
    assert result.visited == ["bump"] * 5  # visit cap of 5 not tripped by the retry


async def test_the_visit_cap_counts_across_a_resume() -> None:
    store = InMemoryCheckpointer()
    state = {"crash": True}

    async def spin(value: int) -> int:
        if value == 2 and state["crash"]:
            raise RuntimeError("crash")
        return value + 1

    wf = Workflow("spin", max_visits_per_node=4)
    wf.add_node("spin", spin)
    wf.set_start("spin")
    wf.add_edge("spin", "spin")
    with pytest.raises(RuntimeError, match="crash"):
        await wf.run(0, session_id="s", checkpointer=store)
    state["crash"] = False
    with pytest.raises(RuntimeError, match="max_visits_per_node=4"):
        await wf.run(0, session_id="s", checkpointer=store)


async def test_a_routed_workflow_resumes_with_the_original_input() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("refund")
    seen: list[Any] = []

    async def classify(text: str) -> str:
        flaky.calls.append("classify")
        return "refund"

    async def refund(text: str) -> str:
        seen.append(text)
        flaky.calls.append("refund")
        if "refund" in flaky.failing:
            raise RuntimeError("refund crashed")
        return f"refunded: {text}"

    wf = Workflow.route(classify, {"refund": refund})
    with pytest.raises(RuntimeError):
        await wf.run("order 4471 please", session_id="s", checkpointer=store)
    flaky.failing.clear()
    result = await wf.run("order 4471 please", session_id="s", checkpointer=store)
    assert flaky.calls == ["classify", "refund", "refund"]
    assert seen == ["order 4471 please"] * 2
    assert result.output == "refunded: order 4471 please"


async def test_an_entry_router_is_not_evaluated_again() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("b")
    picks: list[str] = []

    def pick(value: str) -> str:
        picks.append(value)
        return "go"

    wf = Workflow("entry")
    wf.add_node("a", flaky.node("a")).add_node("b", flaky.node("b"))
    wf.add_router(START, pick, {"go": "a"})
    wf.add_edge("a", "b").add_edge("b", END)
    with pytest.raises(RuntimeError):
        await wf.run("x", session_id="s", checkpointer=store)
    flaky.failing.clear()
    await wf.run("x", session_id="s", checkpointer=store)
    assert picks == ["x"] and flaky.calls == ["a", "b", "b"]


async def test_cancellation_leaves_the_run_resumable() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky()
    delay = {"ship": 10.0}

    async def ship(value: str) -> str:
        flaky.calls.append("ship")
        await anyio.sleep(delay["ship"])
        return f"ship({value})"

    wf = Workflow.chain([flaky.node("charge"), ship], name="order")
    with anyio.move_on_after(0.05):
        await wf.run("o1", session_id="s", checkpointer=store)
    delay["ship"] = 0
    result = await wf.run("o1", session_id="s", checkpointer=store)
    assert flaky.calls == ["charge", "ship", "ship"]
    assert result.resumed is True and result.output == "ship(charge(o1))"


# ---------------------------------------------------------------------------
# What is stored
# ---------------------------------------------------------------------------


async def test_models_and_dataclasses_round_trip(tmp_path: Path) -> None:
    store = SqliteCheckpointer(tmp_path / "wf.db")
    state = {"crash": True}

    async def price(order_id: str) -> Order:
        return Order(id=order_id, total=12.5)

    async def charge(order: Order) -> Receipt:
        assert isinstance(order, Order)
        if state["crash"]:
            raise RuntimeError("card declined")
        return Receipt(order=order, reference="r-1")

    wf = Workflow.chain([price, charge], name="typed")
    with pytest.raises(RuntimeError):
        await wf.run("o1", session_id="s", checkpointer=store)
    state["crash"] = False
    result = await wf.run("o1", session_id="s", checkpointer=store)
    assert result.output == Receipt(order=Order(id="o1", total=12.5), reference="r-1")
    assert result.per_step["price"] == Order(id="o1", total=12.5)
    await store.aclose()


async def test_an_output_that_cannot_be_stored_names_its_node() -> None:
    async def lock(_value: Any) -> Any:
        return threading.Lock()

    wf = Workflow.chain([_Flaky().node("first"), lock], name="bad")
    with pytest.raises(WorkflowCheckpointError, match="the output of node 'lock' in workflow 'bad'"):
        await wf.run("x", session_id="s", checkpointer=InMemoryCheckpointer())
    # Without a checkpointer the same workflow is fine.
    assert (await wf.run("x")).visited == ["first", "lock"]


async def test_an_input_that_cannot_be_stored_is_refused() -> None:
    wf = _order_chain(_Flaky())
    with pytest.raises(WorkflowCheckpointError, match="the input of workflow 'order'"):
        await wf.run(threading.Lock(), session_id="s", checkpointer=InMemoryCheckpointer())


async def test_a_changed_graph_discards_the_checkpoint() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("ship")
    with pytest.raises(RuntimeError):
        await _order_chain(flaky).run("o1", session_id="s", checkpointer=store)
    flaky.failing.clear()
    changed = Workflow.chain([flaky.node("charge"), flaky.node("ship")], name="order")
    with pytest.warns(UserWarning, match="written by a different graph"):
        result = await changed.run("o1", session_id="s", checkpointer=store)
    assert result.resumed is False and flaky.calls == ["charge", "ship", "charge", "ship"]


async def test_stream_marks_a_resumed_run_and_its_replayed_steps() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky("ship")
    wf = _order_chain(flaky)
    with pytest.raises(RuntimeError):
        await wf.run("o1", session_id="s", checkpointer=store)
    flaky.failing.clear()
    events = [ev async for ev in wf.stream("o1", session_id="s", checkpointer=store)]
    assert events[0].kind == EventKind.WORKFLOW_STARTED and events[0].payload["resumed"] is True
    completed = [
        (ev.payload["node"], ev.payload.get("replayed", False))
        for ev in events
        if ev.kind == EventKind.WORKFLOW_STEP_COMPLETED
    ]
    assert completed == [("charge", True), ("ship", False), ("notify", False)]
    started = [ev.payload["node"] for ev in events if ev.kind == EventKind.WORKFLOW_STEP_STARTED]
    assert started == ["ship", "notify"]


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


async def test_an_interrupted_agent_node_leaves_the_run_resumable() -> None:
    store = InMemoryCheckpointer()
    flaky = _Flaky()
    # First run: the agent asks for a tool and runs out of turns
    # (interrupted). Second run: it answers.
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="nope", args={})],
                usage=Usage(input_tokens=10, output_tokens=1, cost_usd=0.01),
            ),
            ScriptedTurn(text="approved", usage=Usage(input_tokens=20, output_tokens=2, cost_usd=0.02)),
        ]
    )
    reviewer = Agent("review the order", model=model, max_turns=1, auto_extract=False)
    wf = Workflow("review")
    wf.add_node("prepare", flaky.node("prepare"))
    wf.add_node("review", reviewer)
    wf.add_node("ship", flaky.node("ship"))
    wf.set_start("prepare")
    wf.add_edge("prepare", "review").add_edge("review", "ship").add_edge("ship", END)

    first = await wf.run("o1", session_id="s", checkpointer=store)
    assert first.interrupted is True
    assert first.interruption_reason == "review:max_turns_exceeded"
    assert flaky.calls == ["prepare"]
    assert len(store.keys()) == 1  # still resumable

    second = await wf.run("o1", session_id="s", checkpointer=store)
    assert second.interrupted is False and second.resumed is True
    assert flaky.calls == ["prepare", "ship"]  # prepare did not run again
    assert second.output == "ship(approved)"
    assert second.visited == ["prepare", "review", "ship"]
    # Usage covers both attempts.
    assert second.cost_usd == pytest.approx(0.03)
    assert store.keys() == []
