"""G16 — parallel DAG scheduling for :class:`loomflow.Workflow`.

The executor is a readiness-based scheduler: fan-out branches
(multiple ``add_edge`` calls from one source) run concurrently in
an anyio task group, bounded by ``max_concurrency``; a node with
several delivered in-edges is an AND-join receiving a list ordered
by edge declaration order. Sequential graphs (chains, routers,
cycles) must behave exactly as they did under the sequential
single-cursor walker — same events, same order, same caps.

Concurrency assertions use enter/exit timestamp *overlap* (branch A
must enter before branch B exits and vice versa) rather than
wall-clock sums, which is far less flaky on loaded CI machines.
"""

from __future__ import annotations

from typing import Any

import anyio
import anyio.lowlevel
import pytest

from loomflow import END, Workflow
from loomflow.core.types import EventKind

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Concurrency: two independent branches overlap in time
# ---------------------------------------------------------------------------


async def test_two_independent_branches_run_concurrently() -> None:
    """A → (B, C): both branches receive A's output and their
    execution intervals overlap — they did not run one after the
    other."""
    stamps: dict[str, tuple[float, float]] = {}

    async def source(x: str) -> str:
        return f"{x}:A"

    def _branch(tag: str) -> Any:
        async def _run(v: str) -> str:
            enter = anyio.current_time()
            await anyio.sleep(0.2)
            stamps[tag] = (enter, anyio.current_time())
            return f"{v}:{tag}"

        _run.__name__ = tag
        return _run

    wf = Workflow("fan-out")
    wf.add_node("A", source)
    wf.add_node("B", _branch("B"))
    wf.add_node("C", _branch("C"))
    wf.set_start("A")
    wf.add_edge("A", "B")
    wf.add_edge("A", "C")  # second edge from A = fan-out
    wf.add_edge("B", END)
    wf.add_edge("C", END)

    result = await wf.run("in")

    # Both branches ran, each with the fan-out source's output.
    assert result.per_step["B"] == "in:A:B"
    assert result.per_step["C"] == "in:A:C"
    # Overlap: each branch entered before the other exited.
    b_enter, b_exit = stamps["B"]
    c_enter, c_exit = stamps["C"]
    assert b_enter < c_exit and c_enter < b_exit, (
        f"branches did not overlap: B={stamps['B']} C={stamps['C']}"
    )


async def test_fan_out_branches_each_receive_source_output() -> None:
    seen: dict[str, Any] = {}

    async def src(x: int) -> int:
        return x + 1

    async def left(v: int) -> int:
        seen["left"] = v
        return v

    async def right(v: int) -> int:
        seen["right"] = v
        return v

    wf = Workflow()
    wf.add_node("src", src)
    wf.add_node("left", left)
    wf.add_node("right", right)
    wf.set_start("src")
    wf.add_edge("src", "left")
    wf.add_edge("src", "right")
    wf.add_edge("left", END)
    wf.add_edge("right", END)

    await wf.run(1)
    assert seen == {"left": 2, "right": 2}


# ---------------------------------------------------------------------------
# Diamond: A → (B, C) → D — AND-join delivers both results
# ---------------------------------------------------------------------------


async def test_diamond_join_receives_both_branch_results() -> None:
    """D is an AND-join: it runs once, after BOTH B and C, and its
    input is the list of branch results in edge DECLARATION order
    (B before C) even though C completes first."""
    join_input: list[Any] = []

    async def a(x: str) -> str:
        return f"{x}|a"

    async def b(v: str) -> str:
        await anyio.sleep(0.1)  # B finishes AFTER C
        return f"{v}|b"

    async def c(v: str) -> str:
        return f"{v}|c"

    async def d(v: Any) -> str:
        join_input.append(v)
        return f"joined:{v}"

    wf = Workflow("diamond")
    wf.add_node("A", a)
    wf.add_node("B", b)
    wf.add_node("C", c)
    wf.add_node("D", d)
    wf.set_start("A")
    wf.add_edge("A", "B")
    wf.add_edge("A", "C")
    wf.add_edge("B", "D")  # declared first → index 0 in the join
    wf.add_edge("C", "D")
    wf.add_edge("D", END)

    result = await wf.run("x")

    # D ran exactly once, with both results, declaration-ordered.
    assert join_input == [["x|a|b", "x|a|c"]]
    assert result.output == "joined:['x|a|b', 'x|a|c']"
    assert result.visited.count("D") == 1
    # All four nodes visited exactly once.
    assert sorted(result.visited) == ["A", "B", "C", "D"]


async def test_diamond_join_events_place_D_after_both_branches() -> None:
    async def passthrough(v: Any) -> Any:
        return v

    wf = Workflow("diamond-events")
    for n in ("A", "B", "C", "D"):
        wf.add_node(n, passthrough)
    wf.set_start("A")
    wf.add_edge("A", "B")
    wf.add_edge("A", "C")
    wf.add_edge("B", "D")
    wf.add_edge("C", "D")
    wf.add_edge("D", END)

    events: list[tuple[str, str | None]] = []
    async for ev in wf.stream(0):
        events.append((ev.kind.value, ev.payload.get("node")))

    completed = [n for k, n in events if k == "workflow_step_completed"]
    started = [n for k, n in events if k == "workflow_step_started"]
    # Per-node events fire for every branch (no opaque fan_out node).
    assert sorted(started) == ["A", "B", "C", "D"]
    assert sorted(completed) == ["A", "B", "C", "D"]
    # D starts only after both B and C completed.
    d_start = events.index(("workflow_step_started", "D"))
    assert events.index(("workflow_step_completed", "B")) < d_start
    assert events.index(("workflow_step_completed", "C")) < d_start


async def test_join_after_router_receives_bare_value() -> None:
    """A merge point fed by a router gets ONE delivery (the router
    took a single branch) and therefore receives the bare value,
    exactly like the sequential model — not a one-element list."""
    merge_inputs: list[Any] = []

    async def classify(x: str) -> str:
        return x

    async def yes(_v: str) -> str:
        return "took-yes"

    async def no(_v: str) -> str:
        return "took-no"

    async def merge(v: Any) -> Any:
        merge_inputs.append(v)
        return v

    wf = Workflow("router-merge")
    wf.add_node("classify", classify)
    wf.add_node("yes", yes)
    wf.add_node("no", no)
    wf.add_node("merge", merge)
    wf.set_start("classify")
    wf.add_router("classify", lambda v: v, {"yes": "yes", "no": "no"})
    wf.add_edge("yes", "merge")
    wf.add_edge("no", "merge")
    wf.add_edge("merge", END)

    result = await wf.run("yes")
    assert merge_inputs == ["took-yes"]  # bare value, no list
    assert result.output == "took-yes"


# ---------------------------------------------------------------------------
# Routers still run exactly one branch
# ---------------------------------------------------------------------------


async def test_router_runs_exactly_one_branch() -> None:
    calls: list[str] = []

    async def classify(x: str) -> str:
        return x

    def _handler(tag: str) -> Any:
        async def _run(v: str) -> str:
            calls.append(tag)
            return tag

        return _run

    wf = Workflow("router")
    wf.add_node("classify", classify)
    wf.add_node("left", _handler("left"))
    wf.add_node("right", _handler("right"))
    wf.set_start("classify")
    wf.add_router(
        "classify", lambda v: v, {"left": "left", "right": "right"}
    )
    wf.add_edge("left", END)
    wf.add_edge("right", END)

    events: list[tuple[str, str | None]] = []
    async for ev in wf.stream("left"):
        events.append((ev.kind.value, ev.payload.get("node")))

    assert calls == ["left"]  # the untaken branch NEVER ran
    # ... and no events were emitted for it either.
    assert ("workflow_step_started", "right") not in events
    assert ("workflow_step_completed", "right") not in events


# ---------------------------------------------------------------------------
# Cycles + caps still terminate under the concurrent scheduler
# ---------------------------------------------------------------------------


async def test_cycle_hits_max_visits_cap() -> None:
    async def a(x: int) -> int:
        return x

    async def b(x: int) -> int:
        return x

    wf = Workflow("runaway", max_visits_per_node=5)
    wf.add_node("a", a)
    wf.add_node("b", b)
    wf.add_edge("a", "b")
    wf.add_edge("b", "a")  # cycle with no termination
    wf.set_start("a")

    with pytest.raises(RuntimeError, match="re-entered .* more than"):
        await wf.run(1)


async def test_cycle_hits_max_steps_cap_globally() -> None:
    """The global step budget is owned by the single scheduler task,
    so it stays exact even with fan-out in the graph."""

    async def n(x: int) -> int:
        return x

    wf = Workflow("zigzag", max_steps=4, max_visits_per_node=100)
    wf.add_node("a", n)
    wf.add_node("b", n)
    wf.add_node("c", n)
    wf.add_edge("a", "b")
    wf.add_edge("b", "c")
    wf.add_edge("c", "a")
    wf.set_start("a")

    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        await wf.run(1)


async def test_router_cycle_terminates_and_matches_sequential_trace() -> None:
    """The refinement-loop shape (A → B → classify → (C|D|END) → B)
    still produces the exact sequential visit trace."""
    iteration = {"n": 0}

    async def step(x: str) -> str:
        return x

    async def classify(x: str) -> str:
        iteration["n"] += 1
        if iteration["n"] == 1:
            return "to_c"
        if iteration["n"] == 2:
            return "to_d"
        return "done"

    wf = Workflow("loop")
    for name in ("A", "B", "C", "D"):
        wf.add_node(name, step)
    wf.add_node("classify", classify)
    wf.add_edge("A", "B")
    wf.add_edge("B", "classify")
    wf.add_router(
        "classify", lambda r: r, {"to_c": "C", "to_d": "D", "done": END}
    )
    wf.add_edge("C", "B")
    wf.add_edge("D", "B")
    wf.set_start("A")

    result = await wf.run("start")
    assert result.visited == [
        "A", "B", "classify", "C",
        "B", "classify", "D",
        "B", "classify",
    ]


# ---------------------------------------------------------------------------
# max_concurrency
# ---------------------------------------------------------------------------


async def test_max_concurrency_caps_simultaneous_nodes() -> None:
    """Five ready branches, cap of 2: the high-water mark of nodes
    inside their bodies never exceeds 2."""
    running = {"now": 0, "high": 0}

    def _branch(i: int) -> Any:
        async def _run(v: int) -> int:
            running["now"] += 1
            running["high"] = max(running["high"], running["now"])
            await anyio.sleep(0.05)
            running["now"] -= 1
            return v

        _run.__name__ = f"b{i}"
        return _run

    wf = Workflow("capped", max_concurrency=2)
    wf.add_node("src", lambda x: x)
    wf.set_start("src")
    for i in range(5):
        name = f"b{i}"
        wf.add_node(name, _branch(i))
        wf.add_edge("src", name)
        wf.add_edge(name, END)

    await wf.run(0)
    assert running["high"] <= 2
    assert running["high"] >= 1


async def test_default_concurrency_actually_fans_out() -> None:
    """With the default cap (8), five branches reach a high-water
    mark above 1 — proof the scheduler runs them concurrently."""
    running = {"now": 0, "high": 0}

    def _branch(i: int) -> Any:
        async def _run(v: int) -> int:
            running["now"] += 1
            running["high"] = max(running["high"], running["now"])
            await anyio.sleep(0.1)
            running["now"] -= 1
            return v

        _run.__name__ = f"b{i}"
        return _run

    wf = Workflow("wide")
    wf.add_node("src", lambda x: x)
    wf.set_start("src")
    for i in range(5):
        name = f"b{i}"
        wf.add_node(name, _branch(i))
        wf.add_edge("src", name)
        wf.add_edge(name, END)

    await wf.run(0)
    assert running["high"] >= 2


def test_max_concurrency_validation() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        Workflow(max_concurrency=0)


# ---------------------------------------------------------------------------
# Failure: a failing branch cancels in-flight siblings, raw exception
# ---------------------------------------------------------------------------


async def test_failing_branch_cancels_sibling_and_raises_raw() -> None:
    sibling_finished = {"done": False}

    async def src(x: int) -> int:
        return x

    async def boom(_v: int) -> int:
        # A couple of checkpoints so the sibling definitely started.
        for _ in range(3):
            await anyio.lowlevel.checkpoint()
        raise ValueError("branch blew up")

    async def slow(v: int) -> int:
        await anyio.sleep(30)  # cancelled by boom's failure
        sibling_finished["done"] = True
        return v

    wf = Workflow("fail-fast")
    wf.add_node("src", src)
    wf.add_node("boom", boom)
    wf.add_node("slow", slow)
    wf.set_start("src")
    wf.add_edge("src", "boom")
    wf.add_edge("src", "slow")
    wf.add_edge("boom", END)
    wf.add_edge("slow", END)

    failed: list[dict[str, Any]] = []
    # The RAW exception propagates — not an ExceptionGroup.
    with pytest.raises(ValueError, match="branch blew up"):
        async for ev in wf.stream(1):
            if ev.kind == EventKind.WORKFLOW_STEP_FAILED:
                failed.append(dict(ev.payload))

    assert not sibling_finished["done"], "sibling was not cancelled"
    assert failed and failed[0]["node"] == "boom"
    assert "branch blew up" in failed[0]["error"]


# ---------------------------------------------------------------------------
# Regression: sequential graphs are event-for-event identical
# ---------------------------------------------------------------------------


async def test_sequential_chain_event_sequence_is_unchanged() -> None:
    """The exact event sequence the sequential walker produced for
    a two-step chain — kinds, order, and payload content."""

    async def a(x: int) -> int:
        return x + 1

    async def b(x: int) -> int:
        return x * 10

    wf = Workflow.chain([a, b], name="seq")

    events: list[tuple[str, dict[str, Any]]] = []
    async for ev in wf.stream(2):
        events.append((ev.kind.value, dict(ev.payload)))

    assert events == [
        ("workflow_started", {"workflow": "seq", "input": 2}),
        ("workflow_step_started", {"workflow": "seq", "node": "a"}),
        (
            "workflow_step_completed",
            {"workflow": "seq", "node": "a", "output": 3},
        ),
        ("workflow_step_started", {"workflow": "seq", "node": "b"}),
        (
            "workflow_step_completed",
            {"workflow": "seq", "node": "b", "output": 30},
        ),
        ("workflow_completed", {"workflow": "seq", "output": 30}),
    ]


async def test_sequential_router_event_sequence_is_unchanged() -> None:
    async def classify(x: str) -> str:
        return x

    async def hit(_v: str) -> str:
        return "handled"

    wf = Workflow("routed")
    wf.add_node("classify", classify)
    wf.add_node("hit", hit)
    wf.add_node("miss", hit)
    wf.set_start("classify")
    wf.add_router("classify", lambda v: v, {"go": "hit", "no": "miss"})
    wf.add_edge("hit", END)
    wf.add_edge("miss", END)

    events: list[tuple[str, dict[str, Any]]] = []
    async for ev in wf.stream("go"):
        events.append((ev.kind.value, dict(ev.payload)))

    assert events == [
        ("workflow_started", {"workflow": "routed", "input": "go"}),
        (
            "workflow_step_started",
            {"workflow": "routed", "node": "classify"},
        ),
        (
            "workflow_step_completed",
            {"workflow": "routed", "node": "classify", "output": "go"},
        ),
        ("workflow_step_started", {"workflow": "routed", "node": "hit"}),
        (
            "workflow_step_completed",
            {"workflow": "routed", "node": "hit", "output": "handled"},
        ),
        ("workflow_completed", {"workflow": "routed", "output": "handled"}),
    ]


async def test_breaking_out_of_stream_cancels_inflight_branches() -> None:
    """The stream docstring promises consumers can break out early
    to cancel — that must tear down in-flight branch workers without
    surfacing an ExceptionGroup, and leave the instance reusable."""
    finished: list[str] = []

    async def src(x: int) -> int:
        return x

    async def slow(v: int) -> int:
        await anyio.sleep(30)
        finished.append("slow")
        return v

    wf = Workflow("break-early")
    wf.add_node("src", src)
    wf.add_node("l", slow)
    wf.add_node("r", slow)
    wf.set_start("src")
    wf.add_edge("src", "l")
    wf.add_edge("src", "r")
    wf.add_edge("l", END)
    wf.add_edge("r", END)

    gen = wf.stream(1)
    seen = 0
    async for _ev in gen:
        seen += 1
        if seen == 3:  # started(src), completed(src), started(l)
            break
    await gen.aclose()

    assert not finished, "in-flight branches were not cancelled"
    # The instance is reusable after an abandoned stream.
    result = await Workflow.chain([src]).run(5)
    assert result.output == 5


async def test_all_events_carry_the_same_session_id() -> None:
    """Interleaved branch events still share one session_id."""

    async def src(x: int) -> int:
        return x

    wf = Workflow("sid")
    wf.add_node("src", src)
    wf.add_node("l", src)
    wf.add_node("r", src)
    wf.set_start("src")
    wf.add_edge("src", "l")
    wf.add_edge("src", "r")
    wf.add_edge("l", END)
    wf.add_edge("r", END)

    sids = set()
    async for ev in wf.stream(1, session_id="s-42"):
        sids.add(ev.session_id)
    assert sids == {"s-42"}
