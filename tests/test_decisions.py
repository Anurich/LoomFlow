"""Tests for loomflow.decisions — the System One decision layer.

Everything runs offline: ``ScriptedDecisions`` drives the seam tests,
a fake ``system_one`` client exercises ``JevModel``'s mapping and
usage accounting, and ``LLMDecisionModel`` runs over ``ScriptedModel``.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import types
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import loomflow.decisions.jev as jev_module
from loomflow import Agent, EchoModel, ScriptedModel, ScriptedTurn, Usage
from loomflow.architecture import TreeOfThoughts
from loomflow.architecture.router import RouterRoute
from loomflow.core.types import Image, Message, ToolCall
from loomflow.decisions import (
    Box,
    BoxDecision,
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


@pytest.fixture(autouse=True)
def _no_default_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``JevModel()`` reads TYPESAFE_DEFAULT_MODEL; a developer's shell
    value must not change which model (and Box gate) tests run with."""
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)


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
    if importlib.util.find_spec("typesafe_sdk") is not None:
        pytest.skip("typesafe-sdk installed; nothing to fail on")
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


async def test_jev_model_never_resends_without_the_pinned_model() -> None:
    """A TypeError mentioning "model" (``response_model``, a pydantic
    BaseModel error, …) must propagate — not trigger a silent second
    request that drops the pinned version."""

    class _Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def system_one(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise TypeError("response_model must be a BaseModel subclass")

    client = _Client()
    with pytest.raises(TypeError, match="response_model"):
        await JevModel("jev-1.13.0", client=client).decide(
            "s", questions={"q": Noul("?")}
        )
    assert [c["model"] for c in client.calls] == ["jev-1.13.0"]


def test_jev_model_rejects_client_options_with_injected_client() -> None:
    with pytest.raises(ValueError, match="timeout, max_retries"):
        JevModel(client=object(), timeout=2, max_retries=0)
    # long_context is per request, so it still applies to any client.
    assert JevModel(client=object(), long_context=True).long_context


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


# ---------------------------------------------------------------------------
# Images, Box, Noul outcome keys, client options (Jev 1.13 / 1.14)
# ---------------------------------------------------------------------------

_PNG = Image.from_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
_JPEG = Image.from_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
_GIF = Image.from_bytes(b"GIF89a" + b"\x00" * 32)


def _raw(sent: Any) -> bytes:
    """Bytes of an image as handed to the client: ours, or the SDK's."""
    return sent.to_bytes() if isinstance(sent, Image) else bytes(sent.data)


class _KwargsJevClient:
    """Accepts the full ``system_one`` signature and records kwargs."""

    def __init__(
        self, answers: dict[str, Any], *, model: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self._answers = answers
        self._model = model
        self._request_id = request_id
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def system_one(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = _FakeJevResponse(self._answers)
        response.model = self._model  # type: ignore[attr-defined]
        response.request_id = self._request_id  # type: ignore[attr-defined]
        return response

    async def aclose(self) -> None:
        self.closed = True


def test_image_from_bytes_sniffs_media_type(tmp_path: Any) -> None:
    assert _PNG.media_type == "image/png"
    assert _JPEG.media_type == "image/jpeg"
    assert _GIF.media_type == "image/gif"
    assert _PNG.to_bytes().startswith(b"\x89PNG")
    # Unrecognised formats are not mislabelled as PNG.
    heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 16
    with pytest.raises(ValueError, match="unrecognised image format"):
        Image.from_bytes(heic)
    assert Image.from_bytes(heic, "image/heic").media_type == "image/heic"
    photo = tmp_path / "photo.heic"
    photo.write_bytes(heic)
    with pytest.raises(ValueError, match="photo.heic: unrecognised"):
        Image.from_path(photo)
    assert Image.from_path(photo, "image/heic").to_bytes() == heic


def test_noul_outcome_keys_normalise_to_wire_true_false() -> None:
    q = Noul("spam?", criteria={"yes": "unsolicited ads", "no": "anything else"})
    assert q.wire_criteria() == {"true": "unsolicited ads", "false": "anything else"}
    # criteria alone (no instructions) is a valid Noul on the wire.
    assert Noul(None, criteria={"true": "an ad"}).instructions is None
    with pytest.raises(ValueError, match="true"):
        Noul("?", criteria={"maybe": "x"})
    with pytest.raises(ValueError, match="instructions or"):
        Noul(None)
    # An alias and its wire key for one outcome would drop a description.
    with pytest.raises(ValueError, match="same outcome twice"):
        Noul("?", criteria={"yes": "a", "true": "b"})
    with pytest.raises(ValueError, match="same outcome twice"):
        Noul("?", criteria={"no": "a", "False": "b"})


def test_choice_and_score_need_criteria() -> None:
    with pytest.raises(ValueError, match="Choice"):
        Choice("?", criteria={})
    with pytest.raises(ValueError, match="Score"):
        Score("?", criteria=[])


async def test_jev_sends_noul_outcomes_as_true_false() -> None:
    client = _KwargsJevClient({"spam": _FakeJevAnswer(noul=0.1)})
    await JevModel(client=client).decide(
        "hi", questions={"spam": Noul("spam?", criteria={"yes": "ad"})}
    )
    sent = client.calls[0]["questions"]["spam"]
    assert sent.criteria == {"true": "ad"}


async def test_jev_images_opt_in_header_and_box_parsing() -> None:
    client = _KwargsJevClient(
        {
            "dog": _FakeJevAnswer(noul=0.97),
            "where": _FakeJevAnswer(
                box={"top": 0.2, "left": 0.19, "bottom": 0.66, "right": 0.68},
                confidence=0.9,
            ),
            "breed": _FakeJevAnswer(
                choice="lab", probabilities={"lab": 0.8, "pug": 0.2},
                confidence=0.8,
            ),
        },
        model="jev-1.14.0",
        request_id="req_123",
    )
    jev = JevModel("jev-preview", client=client)
    d = await jev.decide(
        {"photo": _PNG, "frames": [_JPEG, _PNG]},
        questions={
            "dog": Noul("Is there a dog in `photo`?"),
            "where": Box("the dog in `photo`"),
            # Choice options described by example images.
            "breed": Choice(
                "Which breed is in `photo`?",
                criteria={"lab": {"examples": [_JPEG]}, "pug": None},
            ),
        },
    )
    call = client.calls[0]
    assert call["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
    assert call["model"] == "jev-preview"
    # Images reach the client in place (SDK-less path passes ours
    # through; with the SDK installed they become typesafe_sdk.Image).
    assert _raw(call["state"]["photo"]) == _PNG.to_bytes()
    assert [_raw(f) for f in call["state"]["frames"]] == [
        _JPEG.to_bytes(), _PNG.to_bytes()
    ]
    # One distinct image → one object, so the SDK uploads it once.
    assert call["state"]["photo"] is call["state"]["frames"][1]
    lab = call["questions"]["breed"].criteria["lab"]
    assert [_raw(x) for x in lab["examples"]] == [_JPEG.to_bytes()]

    where = d.boxes["where"]
    assert (where.top, where.left, where.bottom, where.right) == (
        0.2, 0.19, 0.66, 0.68
    )
    assert where.confidence == pytest.approx(0.9)
    assert d.nouls["dog"].noul == pytest.approx(0.97)
    assert d.choices["breed"].choice == "lab"
    assert d.scores == {}
    # The versioned model that answered + support request id.
    assert d.model == "jev-1.14.0"
    assert d.request_id == "req_123"


async def test_jev_text_only_request_sends_no_opt_in_header() -> None:
    client = _KwargsJevClient({"q": _FakeJevAnswer(noul=0.5)})
    await JevModel(client=client).decide("text", questions={"q": Noul("?")})
    assert "extra_headers" not in client.calls[0]


async def test_jev_box_alone_turns_on_images_header() -> None:
    client = _KwargsJevClient(
        {"b": _FakeJevAnswer(box={"top": 0, "left": 0, "bottom": 1, "right": 1},
                             confidence=None)}
    )
    d = await JevModel("jev-preview", client=client).decide(
        "describe", questions={"b": Box("the logo")}
    )
    assert client.calls[0]["extra_headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    assert d.boxes["b"].confidence is None


async def test_jev_long_context_header() -> None:
    client = _KwargsJevClient({"q": _FakeJevAnswer(noul=0.5)})
    await JevModel(client=client, long_context=True).decide(
        "long doc", questions={"q": Noul("?")}
    )
    assert client.calls[0]["extra_headers"] == {
        "TypeSafe-Danger-Zone-Long-Context": "true"
    }


async def test_jev_rejects_formats_the_api_refuses_before_calling() -> None:
    client = _KwargsJevClient({"q": _FakeJevAnswer(noul=0.5)})
    with pytest.raises(DecisionError, match="PNG or JPEG"):
        await JevModel(client=client).decide(
            {"photo": _GIF}, questions={"q": Noul("?")}
        )
    # A mislabelled unknown format is reported as unknown, not as the
    # (accepted) PNG its media_type claims.
    heic = Image(data=base64.b64encode(b"\x00\x00\x00\x18ftypheic").decode())
    with pytest.raises(DecisionError, match="unrecognised format labelled image/png"):
        await JevModel(client=client).decide(heic, questions={"q": Noul("?")})
    assert client.calls == []


async def test_jev_accepts_line_wrapped_base64() -> None:
    raw = _PNG.to_bytes() * 4
    wrapped = Image(data=base64.encodebytes(raw).decode())
    assert "\n" in wrapped.data.strip()
    client = _KwargsJevClient({"q": _FakeJevAnswer(noul=0.5)})
    await JevModel(client=client).decide(
        {"photo": wrapped}, questions={"q": Noul("?")}
    )
    assert _raw(client.calls[0]["state"]["photo"]) == raw


async def test_jev_box_on_pinned_pre_1_14_fails_fast() -> None:
    client = _KwargsJevClient({})
    with pytest.raises(DecisionError, match="jev-preview"):
        await JevModel("jev-1.13.0", client=client).decide(
            _PNG, questions={"b": Box("the dog")}
        )
    assert client.calls == []


async def test_jev_wraps_sdk_errors_with_actionable_hint() -> None:
    class TypeSafeBadRequestError(Exception):
        status = 400

    # Pretend the class came from the SDK (detection is by module).
    TypeSafeBadRequestError.__module__ = "typesafe_sdk._core.errors"

    class _FailingClient:
        async def system_one(self, **kwargs: Any) -> Any:
            raise TypeSafeBadRequestError("400 unsupported question type")

    with pytest.raises(DecisionError, match="Jev 1.14") as info:
        await JevModel(client=_FailingClient()).decide(
            _PNG, questions={"b": Box("the dog")}
        )
    assert isinstance(info.value.__cause__, TypeSafeBadRequestError)
    # Models known to serve Box get the server's reason, not a hint to
    # switch to the model they already use.
    for model in ("jev-preview", "jev-1.14.0", "~typesafe/jev-preview"):
        with pytest.raises(DecisionError) as info:
            await JevModel(model, client=_FailingClient()).decide(
                _PNG, questions={"b": Box("the dog")}
            )
        assert "unsupported question type" in str(info.value)
        assert "Jev 1.14+" not in str(info.value), model


class _StubSdkImage:
    def __init__(self, data: bytes) -> None:
        self.data = data


def _stub_sdk(**overrides: Any) -> Any:
    """A ``typesafe_sdk`` stand-in: question classes return their kwargs."""
    attrs: dict[str, Any] = {
        "Noul": lambda **kw: kw,
        "Choice": lambda **kw: kw,
        "Score": lambda **kw: kw,
        **overrides,
    }
    return types.SimpleNamespace(**attrs)


async def test_jev_on_an_sdk_without_images_or_box_says_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jev_module, "_load_sdk", lambda: _stub_sdk())
    client = _KwargsJevClient({})
    with pytest.raises(DecisionError, match=r"Images need typesafe-sdk>=0\.7\.5"):
        await JevModel(client=client).decide(
            {"photo": _PNG}, questions={"q": Noul("?")}
        )
    with pytest.raises(DecisionError, match=r"Box questions need typesafe-sdk>=0\.7\.5"):
        await JevModel("jev-preview", client=client).decide(
            "s", questions={"b": Box("the logo")}
        )
    assert client.calls == []


async def test_jev_wraps_sdk_question_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StrictChoice(BaseModel):
        model_config = ConfigDict(extra="forbid")
        instructions: str
        criteria: dict[str, str]

    monkeypatch.setattr(
        jev_module, "_load_sdk", lambda: _stub_sdk(Choice=_StrictChoice)
    )
    client = _KwargsJevClient({})
    with pytest.raises(DecisionError, match="question 'pick' is not valid") as info:
        await JevModel(client=client).decide(
            "s", questions={"pick": Choice("which?", {"a": {"nested": 1}})}
        )
    assert isinstance(info.value.__cause__, ValidationError)
    assert client.calls == []


async def test_jev_image_uploads_keyed_by_value_not_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state that builds Images on access frees each one mid-walk;
    an id()-keyed cache then handed later frames an earlier frame's
    upload. Equal images still share one upload."""
    monkeypatch.setattr(
        jev_module, "_load_sdk", lambda: _stub_sdk(Image=_StubSdkImage)
    )
    blobs = {f"f{i}": b"\x89PNG\r\n\x1a\n" + bytes([i]) * 32 for i in range(20)}

    class _LazyFrames(Mapping[str, Image]):
        def __getitem__(self, key: str) -> Image:
            return Image.from_bytes(blobs[key])

        def __iter__(self) -> Iterator[str]:
            return iter(blobs)

        def __len__(self) -> int:
            return len(blobs)

    client = _KwargsJevClient({"q": _FakeJevAnswer(noul=0.5)})
    await JevModel(client=client).decide(_LazyFrames(), questions={"q": Noul("?")})
    sent = client.calls[0]["state"]
    assert {k: _raw(v) for k, v in sent.items()} == blobs

    twins = [Image.from_bytes(blobs["f0"]), Image.from_bytes(blobs["f0"])]
    await JevModel(client=client).decide(twins, questions={"q": Noul("?")})
    first, second = client.calls[1]["state"]
    assert first is second


async def test_jev_prices_every_version_and_gateway_ids() -> None:
    for name in ("jev-1.14.0", "jev-preview", "~typesafe/jev-latest", "typesafe-ai/jev"):
        client = _KwargsJevClient({"q": _FakeJevAnswer(noul=0.5)})
        d = await JevModel(name, client=client).decide("s", questions={"q": Noul("?")})
        assert d.usage.cost_usd == pytest.approx(1000 * 0.042 / 1_000_000), name


def test_jev_default_model_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-preview")
    assert JevModel(client=object()).name == "jev-preview"
    assert JevModel("jev-1.13.0", client=object()).name == "jev-1.13.0"
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL")
    assert JevModel(client=object()).name == "jev-latest"


async def test_jev_aclose_leaves_injected_client_open() -> None:
    client = _KwargsJevClient({})
    async with JevModel(client=client):
        pass
    assert client.closed is False


def test_box_decision_helpers() -> None:
    b = BoxDecision(top=0.25, left=0.1, bottom=0.75, right=0.5, confidence=0.4)
    assert b.to_pixels(200, 100) == (20, 25, 100, 75)
    assert b.width == pytest.approx(0.4) and b.height == pytest.approx(0.5)
    p = b.padded(0.2)
    assert (p.top, p.left, p.bottom, p.right) == pytest.approx((0.05, 0.0, 0.95, 0.7))
    # A box found in the right half's crop maps back to full-image fractions.
    crop = BoxDecision(top=0.0, left=0.5, bottom=1.0, right=1.0)
    inner = BoxDecision(top=0.5, left=0.5, bottom=1.0, right=1.0)
    full = inner.to_full_image(crop)
    assert (full.top, full.left, full.bottom, full.right) == pytest.approx(
        (0.5, 0.75, 1.0, 1.0)
    )


class _CapturingModel:
    """Minimal Model exposing ``complete`` — records what it was sent."""

    name = "capture"

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.sent: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> Any:
        self.sent.append(messages)
        return self._texts.pop(0), [], Usage(), "stop"


async def test_llm_decisions_attach_images_with_placeholders_and_box() -> None:
    model = _CapturingModel(json.dumps({"answers": {
        "where": {"box": {"top": 0.7, "left": 1.4, "bottom": 0.1, "right": 0.2}},
        "same": {"yes_probability": 0.8},
        "level": {"level_probabilities": [0.0, 1.0]},
    }}))
    d = await LLMDecisionModel(model).decide(  # type: ignore[arg-type]
        {"photo": _PNG},
        questions={
            "where": Box("the dog in `photo`"),
            "same": Noul(["Is `photo` the same dog as", _JPEG]),
            "level": Score("How sharp?", criteria=["blurry", _PNG]),
        },
    )
    system, user = model.sent[0]
    # _PNG is reused (state + Score level) → one image, one placeholder.
    assert user.images == (_PNG, _JPEG)
    assert "<image 1>" in user.content and "2 attached" in user.content
    assert "<image 2>" in system.content and "(box)" in system.content
    # Edges clamped to [0, 1] and swapped back into order.
    where = d.boxes["where"]
    assert (where.top, where.left, where.bottom, where.right) == (0.1, 0.2, 0.7, 1.0)
    assert where.confidence is None
    assert d.scores["level"].legend == ["blurry", "<image>"]


async def test_llm_decisions_render_null_state_and_undescribed_outcomes() -> None:
    model = _CapturingModel(json.dumps({"answers": {
        "q": {"yes_probability": 0.5},
        "c": {"probabilities": {"x": 1.0}},
    }}))
    await LLMDecisionModel(model).decide(  # type: ignore[arg-type]
        ["a", None, 3],
        questions={
            "q": Noul("Is it?", criteria={"true": None, "false": None}),
            "c": Choice("Which?", criteria={"x": None}),
        },
    )
    system, user = model.sent[0]
    # A missing state value reads as null, not as a placeholder label.
    assert user.content == "STATE:\na\nnull\n3"
    # No empty "()" when no outcome is described.
    assert "- q (noul): Is it?\n" in system.content
    assert "x: (no description)" in system.content


async def test_llm_decisions_reject_unknown_question_types() -> None:
    class _Alien:
        instructions = "?"

    model = _CapturingModel()
    with pytest.raises(DecisionError, match="unknown question type: _Alien"):
        await LLMDecisionModel(model).decide(  # type: ignore[arg-type]
            "s", questions={"q": _Alien()}  # type: ignore[dict-item]
        )
    assert model.sent == []


async def test_jev_with_real_sdk_builds_multipart_request() -> None:
    """End-to-end through the REAL typesafe-sdk (skipped when the
    extra isn't installed): our Image → SDK Image → multipart upload
    with the opt-in header; the SDK's dict-keyed Score/Box answers
    parse into our types."""
    sdk = pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        return httpx2.Response(
            200,
            headers={"x-typesafe-request-id": "req_e2e"},
            json={
                "model": "jev-1.14.0",
                "usage": {"input_tokens": 2000, "output_tokens": 0},
                "answers": {
                    "dog": {"type": "noul", "noul": 0.9},
                    "where": {
                        "type": "box",
                        "box": {"top": 0.1, "left": 0.2, "bottom": 0.8, "right": 0.9},
                        "confidence": 0.7,
                    },
                    "sharp": {
                        "type": "score", "score": 1.6,
                        "legend": {"0": "blurry", "1": "ok", "2": "sharp"},
                        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                        "confidence": 0.7,
                    },
                },
            },
        )

    client = sdk.AsyncTypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handler),
        compression=False,
    )
    async with JevModel("jev-preview", client=client) as jev:
        d = await jev.decide(
            {"photo": _PNG},
            questions={
                "dog": Noul("Is there a dog in `photo`?", criteria={"no": "no dog"}),
                "where": Box("the dog in `photo`"),
                "sharp": Score("How sharp is `photo`?", criteria=["blurry", "ok", "sharp"]),
            },
        )
    await client.aclose()
    headers = {k.lower(): v for k, v in seen["headers"].items()}
    assert headers["typesafe-danger-zone-images"] == "true"
    assert headers["content-type"].startswith("multipart/form-data")
    body: bytes = seen["body"]
    assert _PNG.to_bytes() in body  # raw bytes uploaded, not base64
    start = body.index(b"{")
    sent = json.loads(body[start : body.index(b"\r\n--", start)])
    ref = sent["state"]["photo"]  # the image became an asset reference
    assert sent["assets"] == {ref: {"type": "image", "part": "asset-0"}}
    assert sent["questions"]["dog"]["criteria"] == {"false": "no dog"}
    assert sent["questions"]["where"] == {
        "type": "box", "instructions": "the dog in `photo`"
    }
    assert sent["model"] == "jev-preview"
    assert d.boxes["where"].right == pytest.approx(0.9)
    assert d.scores["sharp"].probabilities == pytest.approx([0.1, 0.2, 0.7])
    assert d.nouls["dog"].noul == pytest.approx(0.9)
    assert d.model == "jev-1.14.0" and d.request_id == "req_e2e"


async def test_jev_with_real_sdk_wraps_invalid_question_content() -> None:
    """The real SDK's pydantic question models reject non-JSON content
    at construction — that must surface as DecisionError, pre-send."""
    pytest.importorskip("typesafe_sdk")
    client = _KwargsJevClient({})
    with pytest.raises(DecisionError, match="question 'pick' is not valid"):
        await JevModel(client=client).decide(
            "s", questions={"pick": Choice("which?", {"a": object()})}  # type: ignore[dict-item]
        )
    assert client.calls == []
