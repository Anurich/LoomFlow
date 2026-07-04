"""Eval harness (G9) tests: metrics, judge, harness, thresholds, dataset, online.

Everything runs offline against ScriptedModel / in-process fakes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import anyio
import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, tool
from loomflow.core.ids import new_id
from loomflow.core.types import (
    Event,
    Message,
    ModelChunk,
    RunResult,
    ToolCall,
    ToolResult,
    Usage,
)
from loomflow.eval import (
    EVAL_USER_ID,
    Case,
    Contains,
    Dataset,
    EvalHarness,
    EvalReport,
    ExactMatch,
    LLMJudge,
    Metric,
    MultiStepCoherence,
    OnlineScorer,
    ToolExecutionSuccess,
    ToolSelectionAccuracy,
)
from loomflow.eval.harness import CaseResult

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers: synthetic RunResults and events
# ---------------------------------------------------------------------------


def make_result(output: str = "", session_id: str = "s1") -> RunResult:
    now = datetime.now(UTC)
    return RunResult(
        session_id=session_id,
        output=output,
        turns=1,
        started_at=now,
        finished_at=now,
    )


def tc_event(
    tool_name: str,
    args: dict[str, Any] | None = None,
    call_id: str | None = None,
    session_id: str = "s1",
) -> Event:
    call = ToolCall(id=call_id or new_id("tcall"), tool=tool_name, args=args or {})
    return Event.tool_call(session_id, call)


def tr_event(
    call_id: str,
    *,
    ok: bool = True,
    denied: bool = False,
    session_id: str = "s1",
) -> Event:
    if denied:
        result = ToolResult.denied_(call_id, "not allowed")
    elif not ok:
        result = ToolResult.error_(call_id, "boom")
    else:
        result = ToolResult.success(call_id, "fine")
    return Event.tool_result(session_id, result)


# ---------------------------------------------------------------------------
# Metric math on synthetic events
# ---------------------------------------------------------------------------


async def test_tool_selection_accuracy_fraction() -> None:
    case = Case(input="q", expected_tools=["alpha", "beta"])
    events = [tc_event("alpha"), tc_event("gamma")]
    metric = ToolSelectionAccuracy()
    assert metric.applies(case)
    assert await metric.score(case, make_result(), events) == 0.5


async def test_tool_selection_accuracy_order_insensitive_and_vacuous() -> None:
    metric = ToolSelectionAccuracy()
    case = Case(input="q", expected_tools=["b", "a"])
    events = [tc_event("a"), tc_event("b")]
    assert await metric.score(case, make_result(), events) == 1.0
    # Empty expected list: vacuously perfect.
    assert await metric.score(Case(input="q", expected_tools=[]), make_result(), []) == 1.0
    # No ground truth: metric does not apply.
    assert not metric.applies(Case(input="q"))


async def test_tool_execution_success_counts_errors_and_denials() -> None:
    events = [
        tc_event("a", call_id="c1"),
        tr_event("c1", ok=True),
        tc_event("a", call_id="c2"),
        tr_event("c2", ok=False),
        tc_event("a", call_id="c3"),
        tr_event("c3", denied=True),
        tc_event("a", call_id="c4"),
        tr_event("c4", ok=True),
    ]
    metric = ToolExecutionSuccess()
    assert await metric.score(Case(input="q"), make_result(), events) == 0.5
    # No tool activity at all: vacuous success.
    assert await metric.score(Case(input="q"), make_result(), []) == 1.0


async def test_multi_step_coherence_penalises_duplicates() -> None:
    metric = MultiStepCoherence()
    # Distinct calls: fully coherent.
    events = [tc_event("a", {"x": 1}), tc_event("a", {"x": 2}), tc_event("b")]
    assert await metric.score(Case(input="q"), make_result(), events) == 1.0
    # One byte-identical repeat (earlier call succeeded): -0.25.
    events = [
        tc_event("a", {"x": 1}, call_id="c1"),
        tr_event("c1", ok=True),
        tc_event("a", {"x": 1}, call_id="c2"),
        tr_event("c2", ok=True),
    ]
    assert await metric.score(Case(input="q"), make_result(), events) == pytest.approx(0.75)


async def test_multi_step_coherence_penalises_error_retry_loops() -> None:
    metric = MultiStepCoherence()
    # Identical retry of a FAILED call: duplicate (-0.25) + retry (-0.25).
    events = [
        tc_event("a", {"x": 1}, call_id="c1"),
        tr_event("c1", ok=False),
        tc_event("a", {"x": 1}, call_id="c2"),
        tr_event("c2", ok=False),
    ]
    assert await metric.score(Case(input="q"), make_result(), events) == pytest.approx(0.5)
    # Fewer than two calls: nothing to be incoherent about.
    assert await metric.score(Case(input="q"), make_result(), [tc_event("a")]) == 1.0


async def test_multi_step_coherence_clamps_at_zero() -> None:
    metric = MultiStepCoherence()
    events: list[Event] = []
    for i in range(6):
        cid = f"c{i}"
        events.append(tc_event("a", {"x": 1}, call_id=cid))
        events.append(tr_event(cid, ok=False))
    score = await metric.score(Case(input="q"), make_result(), events)
    assert score == 0.0


# ---------------------------------------------------------------------------
# ExactMatch / Contains
# ---------------------------------------------------------------------------


async def test_exact_match_strips_whitespace() -> None:
    metric = ExactMatch()
    case = Case(input="q", expected="42")
    assert await metric.score(case, make_result("  42\n"), []) == 1.0
    assert await metric.score(case, make_result("43"), []) == 0.0
    assert not metric.applies(Case(input="q"))


async def test_contains_substring_and_case_folding() -> None:
    case = Case(input="q", expected="Answer")
    assert await Contains().score(case, make_result("the Answer is 42"), []) == 1.0
    assert await Contains().score(case, make_result("the answer is 42"), []) == 0.0
    insensitive = Contains(case_sensitive=False)
    assert await insensitive.score(case, make_result("the ANSWER is 42"), []) == 1.0
    assert not Contains().applies(Case(input="q"))


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class _StreamOnlyModel:
    """Model without ``complete`` — exercises the stream-drain path."""

    name = "stream-only"

    def __init__(self, text: str) -> None:
        self._text = text

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
        yield ModelChunk(kind="text", text=self._text)
        yield ModelChunk(kind="finish", finish_reason="stop", usage=Usage())


async def test_judge_parses_score_line() -> None:
    model = ScriptedModel([ScriptedTurn(text="Solid answer.\nscore: 0.8")])
    judge = LLMJudge(model)
    score = await judge.score(Case(input="q", expected="ref"), make_result("out"), [])
    assert score == pytest.approx(0.8)
    assert model.remaining == 0


async def test_judge_ignores_prose_numbers_and_retries_once() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(text="I found 0 errors and 1 issue. Great job overall."),
            ScriptedTurn(text="score: 0.4"),
        ]
    )
    judge = LLMJudge(model)
    score = await judge.score(Case(input="q"), make_result("out"), [])
    assert score == pytest.approx(0.4)
    assert model.remaining == 0  # both turns consumed: retry happened


async def test_judge_neutral_fallback_on_double_parse_failure() -> None:
    model = ScriptedModel([ScriptedTurn(text="no score here"), ScriptedTurn(text="still nothing")])
    judge = LLMJudge(model)
    with pytest.warns(UserWarning, match="neutral score"):
        score = await judge.score(Case(input="q"), make_result("out"), [])
    assert score == 0.5


async def test_judge_drains_stream_when_no_complete() -> None:
    judge = LLMJudge(_StreamOnlyModel("reasoning...\nscore: 1"))
    score = await judge.score(Case(input="q"), make_result("out"), [])
    assert score == 1.0


async def test_judge_clamps_score_into_unit_interval() -> None:
    model = ScriptedModel([ScriptedTurn(text="score: 1.0")])
    score = await LLMJudge(model).score(Case(input="q"), make_result(""), [])
    assert score == 1.0


async def test_same_model_judge_anti_pattern_warns() -> None:
    model = ScriptedModel([ScriptedTurn(text="hi")])
    agent = Agent("test", model=model)
    with pytest.warns(UserWarning, match="same-model-judge"):
        EvalHarness(agent, metrics=[LLMJudge(model=agent.model)])


async def test_distinct_judge_model_does_not_warn() -> None:
    agent = Agent("test", model=ScriptedModel([ScriptedTurn(text="hi")]))
    judge = LLMJudge(ScriptedModel([ScriptedTurn(text="score: 1")]))
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        EvalHarness(agent, metrics=[judge])


# ---------------------------------------------------------------------------
# Harness end-to-end (ScriptedModel agent)
# ---------------------------------------------------------------------------


async def test_harness_end_to_end_five_cases() -> None:
    answer = "the answer is 42"
    model = ScriptedModel([ScriptedTurn(text=answer) for _ in range(5)])
    agent = Agent("test", model=model)
    dataset = Dataset([Case(input=f"q{i}", expected=answer) for i in range(5)])
    harness = EvalHarness(agent, metrics=[ExactMatch(), Contains()], concurrency=2)

    report = await harness.run(dataset)

    assert len(report.results) == 5
    assert report.passed
    assert report.mean("exact_match") == 1.0
    assert report.min("exact_match") == 1.0
    assert report.mean("contains") == 1.0
    summary = report.summary()
    assert summary["exact_match"]["count"] == 5.0
    for r in report.results:
        assert r.error is None
        assert r.output == answer
        assert r.session_id is not None
    data = json.loads(report.to_json())
    assert data["passed"] is True
    assert data["metrics"]["exact_match"]["mean"] == 1.0
    assert len(data["cases"]) == 5


async def test_harness_captures_tool_trace_via_emit() -> None:
    @tool
    async def echo_back(msg: str) -> str:
        """Echo back the message."""
        return f"echoed:{msg}"

    model = ScriptedModel(
        [
            ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="echo_back", args={"msg": "hi"})]),
            ScriptedTurn(text="all done"),
        ]
    )
    agent = Agent("test", model=model, tools=[echo_back])
    harness = EvalHarness(
        agent,
        metrics=[ToolSelectionAccuracy(), ToolExecutionSuccess(), MultiStepCoherence()],
        concurrency=1,
    )
    report = await harness.run(Dataset([Case(input="go", expected_tools=["echo_back"])]))

    assert report.passed
    (case_result,) = report.results
    assert case_result.scores["tool_selection_accuracy"] == 1.0
    assert case_result.scores["tool_execution_success"] == 1.0
    assert case_result.scores["multi_step_coherence"] == 1.0


class _CountingAgent:
    """Fake agent that records peak concurrency and honours the emit seam."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.current = 0
        self.max_seen = 0
        self.user_ids: list[str | None] = []
        self.session_ids: list[str | None] = []
        self.fail_on = fail_on

    async def run(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        emit: Callable[[Event], Awaitable[None]] | None = None,
    ) -> RunResult:
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        try:
            await anyio.sleep(0.02)
            if self.fail_on is not None and prompt == self.fail_on:
                raise RuntimeError("scripted failure")
            self.user_ids.append(user_id)
            self.session_ids.append(session_id)
            if emit is not None:
                await emit(Event.started(session_id or "s", prompt))
            return make_result(output=f"ok:{prompt}", session_id=session_id or "s")
        finally:
            self.current -= 1


async def test_harness_respects_concurrency_limit() -> None:
    agent = _CountingAgent()
    harness = EvalHarness(agent, metrics=[], concurrency=2)
    report = await harness.run(Dataset([Case(input=f"q{i}") for i in range(6)]))
    assert len(report.results) == 6
    assert agent.max_seen <= 2
    assert agent.max_seen == 2  # the limiter actually allowed parallelism
    # Fresh session per case, eval user id partition on every run.
    assert len(set(agent.session_ids)) == 6
    assert set(agent.user_ids) == {EVAL_USER_ID}


async def test_harness_rejects_bad_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        EvalHarness(_CountingAgent(), metrics=[], concurrency=0)


async def test_harness_records_case_error_without_aborting_suite() -> None:
    agent = _CountingAgent(fail_on="q1")
    harness = EvalHarness(agent, metrics=[Contains()], concurrency=2)
    report = await harness.run(Dataset([Case(input=f"q{i}", expected="ok") for i in range(3)]))
    assert len(report.results) == 3
    assert not report.passed
    errored = [r for r in report.results if r.error is not None]
    assert len(errored) == 1
    assert "RuntimeError" in (errored[0].error or "")
    with pytest.raises(AssertionError, match="errored"):
        report.assert_thresholds({})


# ---------------------------------------------------------------------------
# Threshold gate
# ---------------------------------------------------------------------------


def _report(scores_per_case: list[dict[str, float]]) -> EvalReport:
    now = datetime.now(UTC)
    return EvalReport(
        results=[
            CaseResult(case_id=f"c{i}", input=f"q{i}", output="o", scores=scores)
            for i, scores in enumerate(scores_per_case)
        ],
        started_at=now,
        finished_at=now,
    )


async def test_assert_thresholds_passes_when_met() -> None:
    report = _report([{"m": 0.9}, {"m": 1.0}])
    report.assert_thresholds({"m": 0.9})  # mean 0.95 >= 0.9: no raise


async def test_assert_thresholds_fails_and_lists_all_failures() -> None:
    report = _report([{"m": 0.2}, {"m": 0.4}])
    with pytest.raises(AssertionError) as excinfo:
        report.assert_thresholds({"m": 0.9, "missing_metric": 0.5})
    message = str(excinfo.value)
    assert "m: mean 0.3000 < threshold 0.90" in message
    assert "missing_metric: no scores recorded" in message


async def test_report_mean_min_only_over_applicable_cases() -> None:
    report = _report([{"m": 0.5}, {}, {"m": 1.0}])
    assert report.mean("m") == pytest.approx(0.75)
    assert report.min("m") == 0.5
    assert report.summary()["m"]["count"] == 2.0
    assert report.mean("absent") is None


# ---------------------------------------------------------------------------
# Dataset JSONL round-trip
# ---------------------------------------------------------------------------


async def test_dataset_jsonl_round_trip(tmp_path: Any) -> None:
    dataset = Dataset(
        [
            Case(input="what is 2+2?", expected="4", metadata={"topic": "math"}),
            Case(input="fetch the page", expected_tools=["http_get"], id="case_fixed"),
            Case(input="no ground truth"),
        ]
    )
    path = tmp_path / "cases.jsonl"
    dataset.to_jsonl(path)
    loaded = Dataset.from_jsonl(path)
    assert len(loaded) == 3
    assert [c.model_dump() for c in loaded] == [c.model_dump() for c in dataset]
    assert loaded[1].id == "case_fixed"
    # Blank lines are tolerated; malformed JSON is a loud error with location.
    path.write_text('{"input": "a"}\n\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        Dataset.from_jsonl(path)


# ---------------------------------------------------------------------------
# Online scorer
# ---------------------------------------------------------------------------


async def test_online_sampling_is_deterministic_for_a_seed() -> None:
    async def sampled_indices(scorer: OnlineScorer) -> list[int]:
        hits: list[int] = []
        for i in range(30):
            if await scorer.maybe_score(f"p{i}", make_result("out"), []) is not None:
                hits.append(i)
        return hits

    a = OnlineScorer([ToolExecutionSuccess()], sample_rate=0.5, seed=42)
    b = OnlineScorer([ToolExecutionSuccess()], sample_rate=0.5, seed=42)
    c = OnlineScorer([ToolExecutionSuccess()], sample_rate=0.5, seed=7)
    hits_a, hits_b, hits_c = (
        await sampled_indices(a),
        await sampled_indices(b),
        await sampled_indices(c),
    )
    assert hits_a == hits_b  # same seed, same sequence -> identical sample
    assert 0 < len(hits_a) < 30  # actually sampling, not all-or-nothing
    assert hits_a != hits_c  # different seed diverges


async def test_online_rollup_aggregates_by_day() -> None:
    scorer = OnlineScorer([ToolExecutionSuccess()], sample_rate=1.0, seed=0)
    events = [tc_event("a", call_id="c1"), tr_event("c1", ok=False)]
    await scorer.maybe_score("p1", make_result("x"), events)  # 0.0
    await scorer.maybe_score("p2", make_result("y"), [])  # 1.0 (vacuous)
    rollup = scorer.rollup()
    assert rollup["seen"] == 2
    assert rollup["sampled"] == 2
    day = datetime.now(UTC).date().isoformat()
    stats = rollup["days"][day]["tool_execution_success"]
    assert stats["count"] == 2.0
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["min"] == 0.0


async def test_online_skips_ground_truth_metrics_on_live_traffic() -> None:
    # ExactMatch needs `expected`; live traffic has none -> no score recorded.
    scorer = OnlineScorer([ExactMatch(), ToolExecutionSuccess()], sample_rate=1.0)
    scores = await scorer.maybe_score("p", make_result("out"), [])
    assert scores == {"tool_execution_success": 1.0}


class _SlowMetric:
    name = "slow"

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float:
        await anyio.sleep(0.05)
        return 1.0


async def test_online_submit_is_fire_and_forget() -> None:
    metric: Metric = _SlowMetric()
    scorer = OnlineScorer([metric], sample_rate=1.0, seed=0)
    async with scorer:
        start = anyio.current_time()
        scorer.submit("p", make_result("out"), [])
        elapsed = anyio.current_time() - start
        assert elapsed < 0.04  # returned without awaiting the slow judge
        assert scorer.rollup()["sampled"] == 0  # not scored yet
    # Task group drained on exit: the background score has landed.
    rollup = scorer.rollup()
    assert rollup["sampled"] == 1
    day = next(iter(rollup["days"]))
    assert rollup["days"][day]["slow"]["count"] == 1.0


async def test_online_submit_requires_context_manager() -> None:
    scorer = OnlineScorer([], sample_rate=1.0)
    with pytest.raises(RuntimeError, match="async with"):
        scorer.submit("p", make_result("out"), [])


async def test_online_scorer_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        OnlineScorer([], sample_rate=1.5)


async def test_harness_online_shares_metric_suite() -> None:
    agent = _CountingAgent()
    harness = EvalHarness(agent, metrics=[ToolExecutionSuccess()])
    scorer = harness.online(sample_rate=1.0, seed=1)
    scores = await scorer.maybe_score("p", make_result("out"), [])
    assert scores == {"tool_execution_success": 1.0}


# ---------------------------------------------------------------------------
# Metric protocol conformance
# ---------------------------------------------------------------------------


def test_builtin_metrics_satisfy_protocol() -> None:
    model = ScriptedModel([])
    metrics: list[Metric] = [
        ToolSelectionAccuracy(),
        ToolExecutionSuccess(),
        MultiStepCoherence(),
        ExactMatch(),
        Contains(),
        LLMJudge(model),
    ]
    for metric in metrics:
        assert isinstance(metric, Metric)
        assert isinstance(metric.name, str) and metric.name
