"""Regression tests for the eval-harness review fixes: per-case
isolation, judge parse failures, judge spend, case timeouts,
interrupted runs, duplicate metric names and non-score values.

Everything runs offline against in-process fakes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.core.types import Message, ModelChunk, Role, ToolCall, Usage
from loomflow.eval import (
    EVAL_USER_ID,
    Case,
    EvalHarness,
    JudgeParseError,
    LLMJudge,
    OnlineScorer,
    record_metric_usage,
)

pytestmark = pytest.mark.anyio


class _Canned:
    """A model that always gives the same reply, at a fixed price."""

    name = "canned"

    def __init__(self, text: str, cost: float = 0.0, delay: float = 0.0) -> None:
        self.text = text
        self.cost = cost
        self.delay = delay
        self.calls = 0
        self.seen: list[str] = []

    async def complete(self, messages: list[Message], **kwargs: Any) -> Any:
        self.calls += 1
        self.seen.append("\n".join(str(m.content) for m in messages))
        if self.delay:
            await anyio.sleep(self.delay)
        usage = Usage(input_tokens=10, output_tokens=5, cost_usd=self.cost)
        return self.text, [], usage, "stop"

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
        text, _calls, usage, finish = await self.complete(messages, **kwargs)
        yield ModelChunk(kind="text", text=text)
        yield ModelChunk(kind="finish", finish_reason=finish, usage=usage)


class _Recording(_Canned):
    """Replies with a per-call secret, so a leak into a later prompt shows."""

    async def complete(self, messages: list[Message], **kwargs: Any) -> Any:
        self.calls += 1
        self.seen.append("\n".join(str(m.content) for m in messages))
        user = [m for m in messages if m.role is Role.USER][-1].content
        return f"answer-to[{user}] SECRET-{self.calls}", [], Usage(), "stop"


def _gate_passes(report: Any, thresholds: dict[str, float]) -> bool:
    try:
        report.assert_thresholds(thresholds)
    except AssertionError:
        return False
    return True


# ---------------------------------------------------------------------------
# 1. Cases must not see each other through memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("memory", [None, "inmemory"])
async def test_a_case_never_sees_another_cases_answer(memory: str | None) -> None:
    model = _Recording("")
    kwargs: dict[str, Any] = {} if memory is None else {"memory": memory}
    agent = Agent("be terse", model=model, **kwargs)
    harness = EvalHarness(agent, [], concurrency=1)
    cases = [Case(input="what is the capital of France"), Case(input="what is the capital of France?")]
    first = await harness.run(cases)
    assert "SECRET" not in model.seen[1]
    # A second run of the suite on the same agent doesn't see the first.
    second = await harness.run(cases)
    assert all("SECRET" not in prompt for prompt in model.seen)
    users = [r.user_id for r in first.results + second.results]
    assert len(set(users)) == 4
    assert all(str(u).startswith(EVAL_USER_ID + ":") for u in users)
    assert len({r.session_id for r in first.results + second.results}) == 4


async def test_a_cases_memory_is_erased_when_it_ends() -> None:
    agent = Agent("be terse", model=_Recording(""), memory="inmemory")
    report = await EvalHarness(agent, [], concurrency=1).run([Case(input="q1"), Case(input="q2")])
    for result in report.results:
        profile = await agent.memory.profile(user_id=result.user_id)
        assert profile.episode_count == 0


# ---------------------------------------------------------------------------
# 2. An unparseable judge is an error, not a 0.5
# ---------------------------------------------------------------------------


async def test_an_unparseable_judge_raises_after_one_retry() -> None:
    model = _Canned("I cannot grade this.")
    judge = LLMJudge(model)
    with pytest.raises(JudgeParseError, match="no parseable 'score:' line"):
        await judge.score(Case(input="q"), _result(), [])
    assert model.calls == 2  # the first ask, then one retry
    assert "did not contain a parseable score line" in model.seen[1]


async def test_an_unparseable_judge_fails_the_gate_and_shows_in_the_report() -> None:
    agent = Agent("be terse", model=_Canned("42"))
    harness = EvalHarness(agent, [LLMJudge(_Canned("I cannot grade this."))])
    with pytest.warns(UserWarning, match="llm_judge"):
        report = await harness.run([Case(input="q1"), Case(input="q2")])
    assert report.passed is False
    assert report.summary() == {}  # no fabricated 0.5
    assert all("JudgeParseError" in r.metric_errors["llm_judge"] for r in report.results)
    assert not _gate_passes(report, {"llm_judge": 0.5})
    assert "JudgeParseError" in report.to_json()


# ---------------------------------------------------------------------------
# 3. Judge spend counts; cases can time out
# ---------------------------------------------------------------------------


async def test_judge_spend_is_in_the_report_and_the_cost_cap() -> None:
    agent = Agent("be terse", model=_Canned("42", cost=0.25))
    judge = LLMJudge(_Canned("score: 1", cost=1.0))
    report = await EvalHarness(agent, [judge], concurrency=1).run(
        [Case(id="a", input="q1"), Case(id="b", input="q2"), Case(id="c", input="q3")],
        max_cost_usd=1.0,
    )
    assert [r.case_id for r in report.results] == ["a"]
    assert report.skipped == ["b", "c"]
    assert report.results[0].cost_usd == pytest.approx(1.25)
    assert report.results[0].metric_cost_usd == pytest.approx(1.0)
    assert report.cost_usd == pytest.approx(1.25)
    assert json.loads(report.to_json())["cost_usd"] == pytest.approx(1.25)
    assert judge.usage.cost_usd == pytest.approx(1.0)
    assert judge.usage.input_tokens == 10


async def test_judge_spend_stays_with_its_own_case_under_concurrency() -> None:
    class _PerCaseJudgeModel(_Canned):
        async def complete(self, messages: list[Message], **kwargs: Any) -> Any:
            await anyio.sleep(0.01)
            cost = 2.0 if "expensive" in str(messages[-1].content) else 0.5
            return "score: 1", [], Usage(cost_usd=cost), "stop"

    agent = Agent("be terse", model=_Canned("42"))
    report = await EvalHarness(agent, [LLMJudge(_PerCaseJudgeModel(""))], concurrency=4).run(
        [Case(id="a", input="expensive"), Case(id="b", input="cheap"), Case(id="c", input="cheap too")]
    )
    assert {r.case_id: r.cost_usd for r in report.results} == {"a": 2.0, "b": 0.5, "c": 0.5}


async def test_a_custom_metric_can_report_its_spend() -> None:
    class Paid:
        name = "paid"

        async def score(self, case: Any, result: Any, events: Any) -> float:
            record_metric_usage(Usage(cost_usd=0.3))
            return 1.0

    report = await EvalHarness(Agent("x", model=_Canned("42")), [Paid()]).run([Case(input="q")])
    assert report.cost_usd == pytest.approx(0.3)
    record_metric_usage(Usage(cost_usd=1.0))  # outside a harness run: a no-op


async def test_a_case_that_times_out_fails_instead_of_hanging() -> None:
    agent = Agent("be terse", model=_Canned("42", delay=30))
    harness = EvalHarness(agent, [LLMJudge(_Canned("score: 1"))])
    with anyio.fail_after(5):
        report = await harness.run([Case(id="slow", input="q")], case_timeout=0.05)
    [result] = report.results
    assert result.error is not None and "case_timeout=0.05s" in result.error
    assert result.scores == {} and report.passed is False
    assert not _gate_passes(report, {})


async def test_a_judge_that_hangs_times_the_case_out() -> None:
    agent = Agent("be terse", model=_Canned("42"))
    harness = EvalHarness(agent, [LLMJudge(_Canned("score: 1", delay=30))])
    with anyio.fail_after(5):
        report = await harness.run([Case(input="q"), Case(input="q2")], case_timeout=0.05)
    assert [r.error is not None for r in report.results] == [True, True]


async def test_case_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="case_timeout"):
        await EvalHarness(Agent("x", model=_Canned("42")), []).run([], case_timeout=0)


# ---------------------------------------------------------------------------
# 4. An interrupted run is not a pass
# ---------------------------------------------------------------------------


def _looping_agent() -> Agent:
    turns = [ScriptedTurn(tool_calls=[ToolCall(id=f"c{i}", tool="nope", args={})]) for i in range(5)]
    return Agent("x", model=ScriptedModel(turns), max_turns=2)


async def test_an_interrupted_run_fails_the_case_by_default() -> None:
    report = await EvalHarness(_looping_agent(), []).run([Case(id="loop", input="q")])
    [result] = report.results
    assert result.interrupted is True
    assert result.interruption_reason == "max_turns_exceeded"
    assert result.error == "interrupted: max_turns_exceeded"
    assert report.passed is False
    assert not _gate_passes(report, {})


async def test_interrupted_runs_can_be_allowed() -> None:
    report = await EvalHarness(_looping_agent(), []).run([Case(input="q")], allow_interrupted=True)
    [result] = report.results
    assert result.interrupted is True and result.interruption_reason == "max_turns_exceeded"
    assert result.error is None and report.passed is True


# ---------------------------------------------------------------------------
# 5. Two metrics can't share a name
# ---------------------------------------------------------------------------


async def test_two_judges_with_their_own_names_both_score() -> None:
    accuracy = LLMJudge(_Canned("score: 0.0"), rubric="accuracy", name="accuracy")
    tone = LLMJudge(_Canned("score: 1.0"), rubric="tone", name="tone")
    assert LLMJudge(_Canned("score: 1")).name == "llm_judge"  # the default is unchanged
    report = await EvalHarness(Agent("x", model=_Canned("42")), [accuracy, tone]).run([Case(input="q")])
    assert report.results[0].scores == {"accuracy": 0.0, "tone": 1.0}


async def test_duplicate_metric_names_are_refused() -> None:
    agent = Agent("x", model=_Canned("42"))
    judges = [LLMJudge(_Canned("score: 0.0")), LLMJudge(_Canned("score: 1.0"))]
    with pytest.raises(ValueError, match="both named 'llm_judge'"):
        EvalHarness(agent, judges)
    with pytest.raises(ValueError, match="both named 'llm_judge'"):
        OnlineScorer(judges)


# ---------------------------------------------------------------------------
# 6. NaN / infinite / out-of-range scores are metric errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.5, "0.9", None])
async def test_a_non_score_is_a_metric_error(value: Any) -> None:
    class Bad:
        name = "m"

        async def score(self, case: Any, result: Any, events: Any) -> Any:
            return value

    harness = EvalHarness(Agent("x", model=_Canned("42")), [Bad()])
    with pytest.warns(UserWarning, match="metric 'm' raised"):
        report = await harness.run([Case(input="q")])
    assert report.results[0].scores == {}
    assert "m" in report.results[0].metric_errors
    assert report.passed is False and report.mean("m") is None
    assert not _gate_passes(report, {"m": 0.9})


async def test_the_gate_fails_a_nan_mean() -> None:
    # Belt and braces: a report built by hand with a NaN score.
    from datetime import UTC, datetime

    from loomflow.eval import EvalReport
    from loomflow.eval.harness import CaseResult

    now = datetime.now(UTC)
    report = EvalReport(
        results=[CaseResult(case_id="a", input="q", scores={"m": float("nan")})],
        started_at=now,
        finished_at=now,
    )
    assert not _gate_passes(report, {"m": 0.9})


def _result() -> Any:
    from datetime import UTC, datetime

    from loomflow.core.types import RunResult

    now = datetime.now(UTC)
    return RunResult(session_id="s", output="out", turns=1, started_at=now, finished_at=now)
