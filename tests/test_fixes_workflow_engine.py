"""Regression tests for the workflow-engine review fixes: a
timing-independent output, per-wave joins, cancellable sync nodes,
per-node timeout / retries, one trace per run, cancelled spans, and
usage on the result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.core.types import Usage
from loomflow.observability import ConsoleTelemetry, FileTelemetry, InMemoryTelemetry
from loomflow.workflow import END, Workflow, step

pytestmark = pytest.mark.anyio


def _node(name: str, delay: float = 0.0) -> Any:
    async def fn(_value: Any) -> str:
        await anyio.sleep(delay)
        return name

    fn.__name__ = name
    return fn


# ---------------------------------------------------------------------------
# 7. The output doesn't depend on which sibling finishes first
# ---------------------------------------------------------------------------


async def _two_branches(y_delay: float, x_delay: float = 0.0) -> Any:
    # A -> P1 -> P2 -> X -> END   (deeper)
    # A -> Y -> Z -> END
    wf = Workflow("timing")
    for name, delay in [("A", 0), ("P1", 0), ("P2", 0), ("X", x_delay), ("Y", y_delay), ("Z", 0)]:
        wf.add_node(name, _node(name, delay))
    wf.set_start("A")
    wf.add_edge("A", "P1").add_edge("P1", "P2").add_edge("P2", "X").add_edge("X", END)
    wf.add_edge("A", "Y").add_edge("Y", "Z").add_edge("Z", END)
    return (await wf.run("in")).output


async def test_output_is_the_deepest_branch_whatever_the_timing() -> None:
    assert await _two_branches(y_delay=0.0) == "X"
    assert await _two_branches(y_delay=0.05) == "X"  # was "Z": the slow sibling won
    assert await _two_branches(y_delay=0.0, x_delay=0.05) == "X"


async def test_equally_deep_branches_go_to_the_one_declared_last() -> None:
    for first_delay, second_delay in [(0.0, 0.03), (0.03, 0.0)]:
        wf = Workflow("tie")
        wf.add_node("start", _node("start"))
        wf.add_node("first", _node("first", first_delay))
        wf.add_node("second", _node("second", second_delay))
        wf.set_start("start")
        wf.add_edge("start", "first").add_edge("start", "second")
        wf.add_edge("first", END).add_edge("second", END)
        assert (await wf.run("in")).output == "second"


async def test_the_last_loop_iteration_wins_over_a_deeper_earlier_ending() -> None:
    # A fans out to B and C. Wave 0: B goes on to the deep D; C loops
    # back. Wave 1: B ends, C ends. The output is from wave 1, not D.
    counts = {"B": 0, "C": 0}

    async def b(_value: Any) -> str:
        counts["B"] += 1
        return f"b{counts['B']}"

    async def c(_value: Any) -> str:
        counts["C"] += 1
        return f"c{counts['C']}"

    wf = Workflow("waves")
    wf.add_node("A", _node("A")).add_node("B", b).add_node("C", c).add_node("D", _node("D"))
    wf.set_start("A")
    wf.add_edge("A", "B").add_edge("A", "C").add_edge("D", END)
    wf.add_router("B", lambda v: "deep" if v == "b1" else "end", {"deep": "D", "end": END})
    wf.add_router("C", lambda v: "loop" if v == "c1" else "end", {"loop": "A", "end": END})
    result = await wf.run("in")
    assert result.output == "c2"  # wave 1; B and C tie on depth, C is declared last
    assert result.per_step["D"] == "D"


# ---------------------------------------------------------------------------
# 8. A join only merges its own wave's deliveries
# ---------------------------------------------------------------------------


async def test_a_join_never_mixes_deliveries_from_two_loop_iterations() -> None:
    seen: list[Any] = []
    count = {"c": 0}

    async def b(_value: Any) -> str:
        return f"b{count['c']}"

    async def c(_value: Any) -> str:
        await anyio.sleep(0.02)
        count["c"] += 1
        return f"c{count['c']}"

    async def join(value: Any) -> Any:
        seen.append(value)
        return value

    wf = Workflow("stale-join")
    wf.add_node("A", _node("A")).add_node("B", b).add_node("C", c).add_node("J", join)
    wf.set_start("A")
    wf.add_edge("A", "B").add_edge("A", "C").add_edge("B", "J").add_edge("J", END)
    wf.add_router("C", lambda v: "loop" if v == "c1" else "join", {"loop": "A", "join": "J"})
    result = await wf.run("in")
    # Wave 0 delivered only B's value (C looped back); wave 1 delivered
    # both. Before the fix J ran once with ['b0', 'b1', 'c2'].
    assert seen == ["b0", ["b1", "c2"]]
    assert result.output == ["b1", "c2"]


async def test_a_fan_out_inside_a_loop_still_joins_once_per_iteration() -> None:
    joined: list[Any] = []
    rounds = {"n": 0}

    async def head(_value: Any) -> int:
        rounds["n"] += 1
        return rounds["n"]

    async def left(value: int) -> str:
        await anyio.sleep(0.01)
        return f"l{value}"

    async def right(value: int) -> str:
        return f"r{value}"

    async def join(value: Any) -> Any:
        joined.append(value)
        return value

    wf = Workflow("loop")
    wf.add_node("head", head).add_node("left", left).add_node("right", right).add_node("join", join)
    wf.set_start("head")
    wf.add_edge("head", "left").add_edge("head", "right")
    wf.add_edge("left", "join").add_edge("right", "join")
    wf.add_router("join", lambda v: "again" if v == ["l1", "r1"] else "done", {"again": "head", "done": END})
    result = await wf.run("in")
    assert joined == [["l1", "r1"], ["l2", "r2"]]
    assert result.output == ["l2", "r2"]


# ---------------------------------------------------------------------------
# 9. Sync nodes are cancellable; per-node timeout and retries
# ---------------------------------------------------------------------------


async def test_a_failing_sibling_does_not_wait_for_a_sync_node() -> None:
    def slow(_value: Any) -> str:
        time.sleep(1.0)
        return "slow"

    async def boom(_value: Any) -> str:
        raise ValueError("boom")

    wf = Workflow("s")
    wf.add_node("a", _node("a")).add_node("slow", slow).add_node("boom", boom)
    wf.set_start("a")
    wf.add_edge("a", "slow").add_edge("a", "boom")
    started = time.perf_counter()
    with pytest.raises(ValueError, match="boom"):
        await wf.run(1)
    assert time.perf_counter() - started < 0.5


async def test_an_outer_timeout_does_not_wait_for_a_sync_node() -> None:
    def slow(_value: Any) -> str:
        time.sleep(1.0)
        return "slow"

    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.1):
            await Workflow.chain([slow]).run(1)
    assert time.perf_counter() - started < 0.5


async def test_a_node_timeout_fails_the_node() -> None:
    def slow_sync(_value: Any) -> str:
        time.sleep(1.0)
        return "slow"

    for fn in (_node("slow_async", delay=5), slow_sync):
        wf = Workflow("t")
        wf.add_node("slow", fn, timeout=0.05)
        wf.set_start("slow")
        started = time.perf_counter()
        with pytest.raises(TimeoutError, match="step 'slow' timed out after 0.05s"):
            await wf.run("x")
        assert time.perf_counter() - started < 0.5


async def test_a_node_is_retried_with_backoff_then_succeeds() -> None:
    attempts: list[float] = []

    async def flaky(_value: Any) -> str:
        attempts.append(time.perf_counter())
        if len(attempts) < 3:
            raise RuntimeError("try again")
        return "done"

    wf = Workflow("r")
    wf.add_node("flaky", flaky, retries=2, retry_backoff=0.02)
    wf.set_start("flaky")
    result = await wf.run("x")
    assert result.output == "done" and result.visited == ["flaky"]
    assert len(attempts) == 3
    assert attempts[1] - attempts[0] >= 0.02 and attempts[2] - attempts[1] >= 0.04


async def test_retries_run_out_and_the_last_failure_fails_the_run() -> None:
    calls = {"n": 0}

    async def broken(_value: Any) -> str:
        calls["n"] += 1
        raise RuntimeError(f"failure {calls['n']}")

    wf = Workflow("r")
    wf.add_node("broken", broken, retries=1, retry_backoff=0)
    wf.set_start("broken")
    with pytest.raises(RuntimeError, match="failure 2"):
        await wf.run("x")
    assert calls["n"] == 2


async def test_a_timed_out_attempt_is_retried() -> None:
    calls = {"n": 0}

    async def slow_once(_value: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            await anyio.sleep(5)
        return "ok"

    wf = Workflow("r")
    wf.add_node("n", slow_once, timeout=0.05, retries=1, retry_backoff=0)
    wf.set_start("n")
    assert (await wf.run("x")).output == "ok"


async def test_step_decorator_takes_timeout_and_retries() -> None:
    calls = {"n": 0}

    @step(retries=1, retry_backoff=0)
    async def flaky(value: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("once")
        return value.upper()

    @step(timeout=0.05)
    async def slow(value: str) -> str:
        await anyio.sleep(5)
        return value

    assert await flaky("hi") == "HI"
    assert flaky.__name__ == "flaky"
    with pytest.raises(TimeoutError, match="step 'slow' timed out"):
        await slow("x")


def test_bad_node_options_are_refused() -> None:
    wf = Workflow("bad")
    with pytest.raises(ValueError, match="timeout"):
        wf.add_node("a", _node("a"), timeout=0)
    with pytest.raises(ValueError, match="retries"):
        wf.add_node("b", _node("b"), retries=-1)
    with pytest.raises(ValueError, match="retry_backoff"):
        wf.add_node("c", _node("c"), retries=1, retry_backoff=-1)


# ---------------------------------------------------------------------------
# 10. One trace per run; cancelled spans are recorded as cancelled
# ---------------------------------------------------------------------------


async def test_a_run_is_one_trace_under_a_root_span() -> None:
    tel = InMemoryTelemetry()
    await Workflow.chain([_node("s1"), _node("s2")], telemetry=tel, name="pipeline").run(
        "x", user_id="u1", session_id="sess-1"
    )
    spans = tel.spans()
    [root] = [s for s in spans if s.name == "loom.workflow"]
    steps = [s for s in spans if s.name == "loom.workflow.step"]
    assert root.parent_span_id is None
    assert root.attributes["workflow"] == "pipeline"
    assert root.attributes["session_id"] == "sess-1" and root.attributes["user_id"] == "u1"
    assert [s.attributes["step"] for s in steps] == ["s1", "s2"]
    assert {s.trace_id for s in steps} == {root.trace_id}
    assert {s.parent_span_id for s in steps} == {root.span_id}


async def test_two_runs_are_two_traces() -> None:
    tel = InMemoryTelemetry()
    wf = Workflow.chain([_node("s1")], telemetry=tel)
    await wf.run("x")
    await wf.run("y")
    roots = [s for s in tel.spans() if s.name == "loom.workflow"]
    assert len(roots) == 2 and roots[0].trace_id != roots[1].trace_id


async def test_a_failed_run_marks_the_root_span() -> None:
    async def boom(_value: Any) -> str:
        raise ValueError("boom")

    tel = InMemoryTelemetry()
    with pytest.raises(ValueError):
        await Workflow.chain([boom], telemetry=tel).run("x")
    [root] = [s for s in tel.spans() if s.name == "loom.workflow"]
    assert root.exception == "ValueError('boom')" and root.cancelled is False


async def test_a_cancelled_span_is_recorded_as_cancelled() -> None:
    tel = InMemoryTelemetry()
    with anyio.move_on_after(0.05):
        await Workflow.chain([_node("hang", delay=10)], telemetry=tel).run("x")
    by_name = {s.name: s for s in tel.spans()}
    for name in ("loom.workflow.step", "loom.workflow"):
        assert by_name[name].cancelled is True
        assert by_name[name].exception == "CancelledError"  # was None: looked like success


async def test_file_telemetry_writes_cancelled_spans(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with anyio.move_on_after(0.05):
        await Workflow.chain([_node("hang", delay=10)], telemetry=FileTelemetry(path)).run("x")
    assert path.exists(), "the cancelled spans were never written"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["name"] for r in records} == {"loom.workflow.step", "loom.workflow"}
    assert all(r["cancelled"] is True and r["exception"] == "CancelledError" for r in records)


async def test_a_span_that_raises_is_not_marked_cancelled(tmp_path: Path) -> None:
    mem, file = InMemoryTelemetry(), FileTelemetry(tmp_path / "t.jsonl")
    for tel in (mem, file):
        with pytest.raises(ValueError):
            async with tel.trace("work"):
                raise ValueError("bad")
    assert mem.spans()[0].exception == "ValueError('bad')" and mem.spans()[0].cancelled is False
    record = json.loads((tmp_path / "t.jsonl").read_text())
    assert record["cancelled"] is False and record["exception"] == "ValueError('bad')"


async def test_console_telemetry_prints_cancelled(capsys: pytest.CaptureFixture[str]) -> None:
    import sys

    tel = ConsoleTelemetry(stream=sys.stdout)
    with anyio.move_on_after(0.05):
        async with tel.trace("work"):
            await anyio.sleep(10)
    assert "[CANCELLED]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 11. Usage and cost on the result
# ---------------------------------------------------------------------------


def _agent(text: str, usage: Usage) -> Agent:
    return Agent("be terse", model=ScriptedModel([ScriptedTurn(text=text, usage=usage)]), auto_extract=False)


async def test_result_adds_up_the_agent_nodes_usage() -> None:
    first = _agent("one", Usage(input_tokens=100, output_tokens=10, cost_usd=0.01))
    second = _agent("two", Usage(input_tokens=200, cached_input_tokens=50, output_tokens=20, cost_usd=0.02))
    result = await Workflow.chain([first, _node("plain"), second]).run("go")
    usage = result.usage
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (300, 50, 30)
    assert usage.cost_usd == pytest.approx(0.03)
    assert result.cost_usd == pytest.approx(0.03)


async def test_usage_counts_parallel_branches_and_nested_workflows() -> None:
    a = _agent("a", Usage(input_tokens=1, output_tokens=1, cost_usd=0.5))
    b = _agent("b", Usage(input_tokens=2, output_tokens=2, cost_usd=0.25))
    inner = Workflow.chain([_agent("c", Usage(input_tokens=4, output_tokens=4, cost_usd=1.0))], name="inner")
    outer = Workflow("outer")
    outer.add_node("fan", Workflow.parallel([a, b]))
    outer.add_node("inner", inner)
    outer.set_start("fan")
    outer.add_edge("fan", "inner").add_edge("inner", END)
    result = await outer.run("go")
    assert result.cost_usd == pytest.approx(1.75)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (7, 7)


async def test_a_workflow_without_agents_reports_zero_usage() -> None:
    result = await Workflow.chain([_node("a")]).run("x")
    assert result.usage == Usage() and result.cost_usd == 0.0
