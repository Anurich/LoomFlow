"""Tests for loomflow.decisions — the System One decision layer.

Everything runs offline: ``ScriptedDecisions`` drives the seam tests,
a fake ``system_one`` client exercises ``JevModel``'s mapping and
usage accounting, and ``LLMDecisionModel`` runs over ``ScriptedModel``.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, EchoModel, ScriptedModel, ScriptedTurn, Usage
from loomflow.architecture import TreeOfThoughts
from loomflow.architecture.router import RouterRoute
from loomflow.core.types import ToolCall
from loomflow.decisions import (
    Choice,
    ChoiceDecision,
    DecisionApprovalPolicy,
    DecisionError,
    DecisionGuard,
    Decisions,
    JevModel,
    LLMDecisionModel,
    Noul,
    NoulDecision,
    Score,
    ScoreDecision,
    ScriptedDecisions,
    resolve_decision_model,
)
from loomflow.team import Team

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def test_score_decision_normalized_maps_levels_to_unit_interval() -> None:
    d = ScoreDecision(
        score=3.0,
        legend=["a", "b", "c", "d", "e"],
        probabilities=[0, 0, 0, 1, 0],
        confidence=1.0,
    )
    assert d.normalized == pytest.approx(0.75)
    # Single-level legends can't divide by zero.
    single = ScoreDecision(
        score=0.0, legend=["only"], probabilities=[1.0], confidence=1.0
    )
    assert single.normalized == 0.0


def test_decisions_getitem() -> None:
    d = Decisions(answers={"x": NoulDecision(noul=0.4)})
    assert d["x"].noul == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# ScriptedDecisions
# ---------------------------------------------------------------------------


async def test_scripted_decisions_standing_and_script_queue() -> None:
    sd = ScriptedDecisions(
        answers={"met": NoulDecision(noul=0.1)},
        script=[{"met": NoulDecision(noul=0.9)}],
    )
    first = await sd.decide("s1", questions={"met": Noul("done?")})
    second = await sd.decide("s2", questions={"met": Noul("done?")})
    assert first["met"].noul == pytest.approx(0.9)  # script overlay
    assert second["met"].noul == pytest.approx(0.1)  # standing default
    assert [state for state, _ in sd.calls] == ["s1", "s2"]


async def test_scripted_decisions_missing_answer_raises() -> None:
    sd = ScriptedDecisions()
    with pytest.raises(DecisionError, match="no answer"):
        await sd.decide("s", questions={"q": Noul("?")})


# ---------------------------------------------------------------------------
# JevModel — fake client (no SDK, no network)
# ---------------------------------------------------------------------------


class _FakeJevAnswer:
    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)


class _FakeJevUsage:
    input_tokens = 1000
    output_tokens = 0


class _FakeJevResponse:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.usage = _FakeJevUsage()


class _FakeJevClient:
    def __init__(self, answers: dict[str, Any]) -> None:
        self._answers = answers
        self.calls: list[dict[str, Any]] = []

    async def system_one(
        self, *, state: Any, questions: Any, model: Any = None
    ) -> Any:
        self.calls.append(
            {"state": state, "questions": questions, "model": model}
        )
        return _FakeJevResponse(self._answers)


async def test_jev_model_maps_answers_and_prices_usage() -> None:
    client = _FakeJevClient(
        {
            "dept": _FakeJevAnswer(
                choice="billing",
                probabilities={"billing": 0.8, "tech": 0.2},
                confidence=0.8,
            ),
            "urgent": _FakeJevAnswer(noul=0.95),
            "mood": _FakeJevAnswer(
                score=1.5,
                legend=["calm", "annoyed", "angry"],
                probabilities=[0.1, 0.3, 0.6],
                confidence=0.6,
            ),
        }
    )
    jev = JevModel(client=client)
    d = await jev.decide(
        "ticket text",
        questions={
            "dept": Choice("team?", criteria={"billing": "b", "tech": "t"}),
            "urgent": Noul("urgent?"),
            "mood": Score("mood?", criteria=["calm", "annoyed", "angry"]),
        },
    )
    dept = d["dept"]
    assert isinstance(dept, ChoiceDecision)
    assert dept.choice == "billing"
    urgent = d["urgent"]
    assert isinstance(urgent, NoulDecision)
    assert urgent.noul == pytest.approx(0.95)
    mood = d["mood"]
    assert isinstance(mood, ScoreDecision)
    assert mood.normalized == pytest.approx(0.75)
    # 1000 tokens at $0.042/M input, output free.
    assert d.usage.input_tokens == 1000
    assert d.usage.cost_usd == pytest.approx(1000 * 0.042 / 1_000_000)
    assert client.calls and client.calls[0]["state"] == "ticket text"


async def test_jev_model_missing_answer_raises() -> None:
    jev = JevModel(client=_FakeJevClient({}))
    with pytest.raises(DecisionError, match="missing answer"):
        await jev.decide("s", questions={"q": Noul("?")})


def test_jev_model_without_sdk_or_client_raises_import_error() -> None:
    with pytest.raises(ImportError, match="loomflow\\[typesafe\\]"):
        JevModel()


async def test_jev_model_score_mapping_forms_coerced_in_order() -> None:
    """The SDK returns Score legend/probabilities as MAPPINGS (levels
    keyed by number). ``list(mapping)`` yields keys — the regression
    that turned [0.1, 0.3, 0.6] into [0, 1, 2]. Both number-keyed and
    text-keyed forms must land in level order."""
    client = _FakeJevClient(
        {
            "mood": _FakeJevAnswer(
                score=1.5,
                legend={"0": "calm", "1": "annoyed", "2": "angry"},
                probabilities={"0": 0.1, "1": 0.3, "2": 0.6},
                confidence=0.6,
            ),
            "tone": _FakeJevAnswer(
                score=0.5,
                legend={0: "soft", 1: "loud"},
                probabilities={"loud": 0.4, "soft": 0.6},  # text-keyed
                confidence=0.6,
            ),
        }
    )
    jev = JevModel(client=client)
    d = await jev.decide(
        "s",
        questions={
            "mood": Score("?", criteria=["calm", "annoyed", "angry"]),
            "tone": Score("?", criteria=["soft", "loud"]),
        },
    )
    mood = d["mood"]
    assert isinstance(mood, ScoreDecision)
    assert mood.legend == ["calm", "annoyed", "angry"]
    assert mood.probabilities == pytest.approx([0.1, 0.3, 0.6])
    tone = d["tone"]
    assert isinstance(tone, ScoreDecision)
    assert tone.legend == ["soft", "loud"]
    assert tone.probabilities == pytest.approx([0.6, 0.4])


async def test_jev_model_forwards_model_version() -> None:
    client = _FakeJevClient({"q": _FakeJevAnswer(noul=0.5)})
    jev = JevModel("jev-1.13.0", client=client)
    await jev.decide("s", questions={"q": Noul("?")})
    assert client.calls[0]["model"] == "jev-1.13.0"


async def test_jev_model_falls_back_when_client_lacks_model_param() -> None:
    """Clients predating per-call model selection (or old fakes) get
    one precise retry without the kwarg; unrelated TypeErrors are
    not swallowed."""

    class _LegacyClient:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def system_one(self, *, state: Any, questions: Any) -> Any:
            self.calls.append(state)
            return _FakeJevResponse({"q": _FakeJevAnswer(noul=0.7)})

    legacy = _LegacyClient()
    jev = JevModel("jev-1.13.0", client=legacy)
    d = await jev.decide("s", questions={"q": Noul("?")})
    q = d["q"]
    assert isinstance(q, NoulDecision)
    assert q.noul == pytest.approx(0.7)
    assert legacy.calls == ["s"]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_resolver_passthrough_and_llm_wrap() -> None:
    sd = ScriptedDecisions()
    assert resolve_decision_model(sd) is sd
    assert resolve_decision_model(None) is None
    wrapped = resolve_decision_model("echo")
    assert isinstance(wrapped, LLMDecisionModel)
    wrapped2 = resolve_decision_model(EchoModel())
    assert isinstance(wrapped2, LLMDecisionModel)


def test_resolver_rejects_garbage() -> None:
    from loomflow import ConfigError

    with pytest.raises(ConfigError, match="decider="):
        resolve_decision_model(42)


# ---------------------------------------------------------------------------
# LLMDecisionModel — parse, coerce, retry
# ---------------------------------------------------------------------------


def _llm(*texts: str) -> LLMDecisionModel:
    return LLMDecisionModel(
        ScriptedModel([ScriptedTurn(text=t) for t in texts])
    )


async def test_llm_decisions_choice_score_noul() -> None:
    llm = _llm(
        '{"answers": {"dept": {"probabilities": {"a": 0.6, "b": 0.4}},'
        ' "ok": {"yes_probability": 0.2},'
        ' "level": {"level_probabilities": [0.0, 1.0]}}}'
    )
    d = await llm.decide(
        {"k": "v"},
        questions={
            "dept": Choice("?", criteria={"a": "", "b": ""}),
            "ok": Noul("?"),
            "level": Score("?", criteria=["lo", "hi"]),
        },
    )
    dept = d["dept"]
    assert isinstance(dept, ChoiceDecision)
    assert dept.choice == "a"
    assert dept.confidence == pytest.approx(0.6)
    level = d["level"]
    assert isinstance(level, ScoreDecision)
    assert level.score == pytest.approx(1.0)
    ok = d["ok"]
    assert isinstance(ok, NoulDecision)
    assert ok.noul == pytest.approx(0.2)


async def test_llm_decisions_tolerates_code_fences() -> None:
    llm = _llm(
        '```json\n{"answers": {"ok": {"yes_probability": 0.7}}}\n```'
    )
    d = await llm.decide("s", questions={"ok": Noul("?")})
    ok = d["ok"]
    assert isinstance(ok, NoulDecision)
    assert ok.noul == pytest.approx(0.7)


async def test_llm_decisions_retries_once_then_succeeds() -> None:
    llm = _llm(
        "not json at all",
        '{"answers": {"ok": {"yes_probability": 0.9}}}',
    )
    d = await llm.decide("s", questions={"ok": Noul("?")})
    ok = d["ok"]
    assert isinstance(ok, NoulDecision)
    assert ok.noul == pytest.approx(0.9)


async def test_llm_decisions_two_failures_raise() -> None:
    llm = _llm("garbage", "still garbage")
    with pytest.raises(DecisionError, match="after retry"):
        await llm.decide("s", questions={"ok": Noul("?")})


async def test_llm_decisions_choice_zero_mass_rejected() -> None:
    llm = _llm(
        '{"answers": {"dept": {"probabilities": {"zzz": 1.0}}}}',
        '{"answers": {"dept": {"probabilities": {"zzz": 1.0}}}}',
    )
    with pytest.raises(DecisionError, match="no probability"):
        await llm.decide(
            "s", questions={"dept": Choice("?", criteria={"a": ""})}
        )


async def test_llm_decisions_score_length_mismatch_rejected() -> None:
    llm = _llm(
        '{"answers": {"lv": {"level_probabilities": [1.0]}}}',
        '{"answers": {"lv": {"level_probabilities": [1.0]}}}',
    )
    with pytest.raises(DecisionError, match="probabilities for"):
        await llm.decide(
            "s", questions={"lv": Score("?", criteria=["a", "b"])}
        )


# ---------------------------------------------------------------------------
# DecisionApprovalPolicy — three tiers, per-tool bounds, fail-closed
# ---------------------------------------------------------------------------


def _policy(p_risky: float, **kwargs: Any) -> DecisionApprovalPolicy:
    return DecisionApprovalPolicy(
        ScriptedDecisions(answers={"risky": NoulDecision(noul=p_risky)}),
        **kwargs,
    )


async def test_policy_auto_allows_low_risk() -> None:
    decision = await _policy(0.01)(
        ToolCall(id="t", tool="read", args={}), "u"
    )
    assert not isinstance(decision, bool)
    assert decision.action == "allow"


async def test_policy_auto_denies_high_risk() -> None:
    decision = await _policy(0.99)(
        ToolCall(id="t", tool="bash", args={"cmd": "rm -rf /"}), "u"
    )
    assert not isinstance(decision, bool)
    assert decision.action == "deny"


async def test_policy_escalates_middle_band_to_handler() -> None:
    seen: list[str] = []

    async def handler(call: ToolCall, user_id: str | None) -> bool:
        seen.append(call.tool)
        return True

    decision = await _policy(0.5, escalate_to=handler)(
        ToolCall(id="t", tool="bash", args={}), "u"
    )
    assert decision is True
    assert seen == ["bash"]


async def test_policy_middle_band_without_handler_fails_closed() -> None:
    decision = await _policy(0.5)(
        ToolCall(id="t", tool="bash", args={}), "u"
    )
    assert not isinstance(decision, bool)
    assert decision.action == "deny"
    assert "fail-closed" in (decision.reason or "")


async def test_policy_per_tool_bounds_override() -> None:
    # 0.03 auto-allows by the default bound but bash is stricter.
    policy = _policy(0.03, per_tool={"bash": {"allow_below": 0.01}})
    read_d = await policy(ToolCall(id="a", tool="read", args={}), "u")
    bash_d = await policy(ToolCall(id="b", tool="bash", args={}), "u")
    assert not isinstance(read_d, bool) and read_d.action == "allow"
    assert not isinstance(bash_d, bool) and bash_d.action == "deny"


async def test_policy_decider_exception_fails_closed() -> None:
    class _Boom:
        name = "boom"

        async def decide(self, state: Any, *, questions: Any) -> Any:
            raise RuntimeError("network down")

    decision = await DecisionApprovalPolicy(_Boom())(
        ToolCall(id="t", tool="read", args={}), "u"
    )
    assert not isinstance(decision, bool)
    assert decision.action == "deny"
    assert "fail-closed" in (decision.reason or "")


def test_policy_threshold_validation() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        DecisionApprovalPolicy(
            ScriptedDecisions(), allow_below=0.9, deny_above=0.1
        )


# ---------------------------------------------------------------------------
# DecisionGuard
# ---------------------------------------------------------------------------


async def test_guard_blocks_at_threshold_and_allows_below() -> None:
    guard = DecisionGuard(
        ScriptedDecisions(
            script=[
                {"flag": NoulDecision(noul=0.9)},
                {"flag": NoulDecision(noul=0.1)},
            ]
        ),
        question=Noul("bad?"),
        threshold=0.8,
        name="test_guard",
        stages=("input",),
    )
    assert guard.stages == frozenset({"input"})
    blocked = await guard.check("evil text", stage="input")
    assert blocked.action == "block"
    assert "test_guard" in (blocked.reason or "")
    allowed = await guard.check("fine text", stage="input")
    assert allowed.action == "allow"


# ---------------------------------------------------------------------------
# Router seam — decider classification + confidence gate
# ---------------------------------------------------------------------------


def _routes() -> list[RouterRoute]:
    return [
        RouterRoute(
            name="billing",
            agent=Agent("billing bot", model=EchoModel()),
            description="payments",
        ),
        RouterRoute(
            name="tech",
            agent=Agent("tech bot", model=EchoModel()),
            description="bugs",
        ),
    ]


async def test_router_decider_routes_without_llm_classifier() -> None:
    decider = ScriptedDecisions(
        answers={
            "route": ChoiceDecision(
                choice="billing",
                probabilities={"billing": 0.9, "tech": 0.1},
                confidence=0.9,
            )
        }
    )
    team = Team.router(_routes(), model=EchoModel(), decider=decider)
    result = await team.run("my card was charged twice")
    # EchoModel echoes its prompt — the billing specialist ran.
    assert "charged twice" in result.output
    # The decider was consulted exactly once, with the user prompt.
    assert len(decider.calls) == 1
    assert decider.calls[0][0] == "my card was charged twice"


async def test_router_decider_low_confidence_takes_fallback() -> None:
    decider = ScriptedDecisions(
        answers={
            "route": ChoiceDecision(
                choice="tech",
                probabilities={"billing": 0.45, "tech": 0.55},
                confidence=0.55,
            )
        }
    )
    team = Team.router(
        _routes(),
        model=EchoModel(),
        decider=decider,
        require_confidence_above=0.8,
        fallback_route="billing",
    )
    result = await team.run("ambiguous request")
    assert "ambiguous request" in result.output  # fallback ran


# ---------------------------------------------------------------------------
# run_until seam — decider goal check with uncertain-band fallback
# ---------------------------------------------------------------------------


async def test_run_until_decider_reprompts_then_stops() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(text="still working"),
            ScriptedTurn(text="done: report written"),
        ]
    )
    decider = ScriptedDecisions(
        script=[
            {"met": NoulDecision(noul=0.05)},  # turn 1: keep going
            {"met": NoulDecision(noul=0.95)},  # turn 2: stop
        ]
    )
    agent = Agent(
        "work",
        model=model,
        run_until={"condition": "the report is written", "decider": decider},
    )
    result = await agent.run("write the report")
    assert result.output == "done: report written"
    assert len(decider.calls) == 2


async def test_run_until_decider_uncertain_falls_through_to_checker() -> None:
    model = ScriptedModel([ScriptedTurn(text="maybe finished?")])
    decider = ScriptedDecisions(
        answers={"met": NoulDecision(noul=0.5)}  # uncertain band
    )
    checker = ScriptedModel([ScriptedTurn(text="DONE")])
    agent = Agent(
        "work",
        model=model,
        run_until={
            "condition": "the task is complete",
            "decider": decider,
            "checker": checker,
        },
    )
    result = await agent.run("do the task")
    # Decider was uncertain; the LLM checker adjudicated DONE after
    # the single model turn.
    assert result.output == "maybe finished?"
    assert len(decider.calls) == 1


# ---------------------------------------------------------------------------
# TreeOfThoughts seam — decider evaluation
# ---------------------------------------------------------------------------


async def test_tot_evaluator_decider_scores_thoughts() -> None:
    main = ScriptedModel(
        [ScriptedTurn(text="1. a promising next step")]
    )
    decider = ScriptedDecisions(
        answers={
            "promise": ScoreDecision(
                score=4.0,
                legend=["l0", "l1", "l2", "l3", "l4"],
                probabilities=[0, 0, 0, 0, 1],
                confidence=1.0,
            )
        },
        usage=Usage(input_tokens=50, cost_usd=0.0000021),
    )
    agent = Agent(
        "solve",
        model=main,
        architecture=TreeOfThoughts(
            branch_factor=1,
            max_depth=1,
            beam_width=1,
            parallel=False,
            synthesize_final=False,
            evaluator_decider=decider,
        ),
    )
    result = await agent.run("problem")
    # normalized 4/4 = 1.0 >= solved_threshold → the thought wins and
    # (no synthesis) becomes the output.
    assert "promising next step" in result.output
    assert decider.calls, "evaluator decider never invoked"
    # The decider saw the rendered evaluator context.
    state = str(decider.calls[0][0])
    assert "problem" in state
