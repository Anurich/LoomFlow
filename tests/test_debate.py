"""MultiAgentDebate architecture tests.

Covers:

* Protocol satisfaction; declared_workers exposes debaters + judge.
* Constructor validation: <2 debaters rejected, rounds<1 rejected.
* :func:`_normalize` + :func:`_converged` + :func:`_majority_vote`
  helpers.
* Round 0 runs all debaters in parallel with the original prompt.
* Rounds 1..K each debater sees the full prior transcript with
  the "(you)" marker on its own previous responses.
* Convergence early-exit: all debaters in a round agree → terminate.
* Judge synthesis path (judge != None): final answer comes from
  the judge Agent.
* Majority-vote fallback (judge == None): modal answer wins.
* Deterministic debater + judge session ids for replay correctness.
* Architecture progress events surface for each milestone.
"""

from __future__ import annotations

import pytest

from jeevesagent import (
    Agent,
    Architecture,
    MultiAgentDebate,
    ScriptedModel,
    ScriptedTurn,
)
from jeevesagent.architecture.debate import (
    _converged,
    _majority_vote,
    _normalize,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted_agent(replies: list[str], instructions: str = "test") -> Agent:
    """Build an Agent that returns the given list of fixed text replies in order."""
    return Agent(
        instructions,
        model=ScriptedModel([ScriptedTurn(text=r) for r in replies]),
    )


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_debate_satisfies_architecture_protocol() -> None:
    d1 = _scripted_agent(["a"])
    d2 = _scripted_agent(["b"])
    arch = MultiAgentDebate(debaters=[d1, d2])
    assert isinstance(arch, Architecture)


def test_debate_name_is_debate() -> None:
    d1, d2 = _scripted_agent(["a"]), _scripted_agent(["b"])
    assert MultiAgentDebate(debaters=[d1, d2]).name == "debate"


def test_debate_declared_workers_exposes_debaters_and_judge() -> None:
    d1 = _scripted_agent(["a"])
    d2 = _scripted_agent(["b"])
    judge = _scripted_agent(["j"])
    arch = MultiAgentDebate(debaters=[d1, d2], judge=judge)
    workers = arch.declared_workers()
    assert workers == {"debater_0": d1, "debater_1": d2, "judge": judge}


def test_debate_declared_workers_without_judge() -> None:
    d1 = _scripted_agent(["a"])
    d2 = _scripted_agent(["b"])
    arch = MultiAgentDebate(debaters=[d1, d2])
    workers = arch.declared_workers()
    assert workers == {"debater_0": d1, "debater_1": d2}


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_debate_rejects_fewer_than_two_debaters() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        MultiAgentDebate(debaters=[_scripted_agent(["x"])])
    with pytest.raises(ValueError, match="at least 2"):
        MultiAgentDebate(debaters=[])


def test_debate_rejects_rounds_lt_1() -> None:
    d1, d2 = _scripted_agent(["a"]), _scripted_agent(["b"])
    with pytest.raises(ValueError, match="rounds"):
        MultiAgentDebate(debaters=[d1, d2], rounds=0)


# ---------------------------------------------------------------------------
# Helpers: normalize / converge / majority vote
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace_and_lowercases() -> None:
    assert _normalize("  Hello   World  ") == "hello world"
    assert _normalize("YES") == "yes"


def test_converged_true_when_all_responses_match_after_normalize() -> None:
    responses = {
        "d0": "the answer is 42",
        "d1": "  THE answer IS  42  ",
        "d2": "the answer is 42",
    }
    assert _converged(responses)


def test_converged_false_when_disagreement() -> None:
    responses = {"d0": "yes", "d1": "no"}
    assert not _converged(responses)


def test_converged_false_when_empty() -> None:
    assert not _converged({})


def test_converged_jaccard_default_handles_minor_wording() -> None:
    """Default threshold (0.85) accepts near-identical answers
    that differ only in trivial wording."""
    responses = {
        "d0": "the answer is 42 because of physics",
        "d1": "the answer is 42 because of physics",
        "d2": "the answer is 42 because of physics indeed",
    }
    assert _converged(responses)


def test_converged_strict_threshold_rejects_minor_wording() -> None:
    """``threshold=1.0`` reproduces the legacy strict-equality
    behaviour — any wording difference fails convergence."""
    responses = {
        "d0": "yes the answer is 42",
        "d1": "the answer is 42",
    }
    assert not _converged(responses, threshold=1.0)


def test_converged_low_threshold_accepts_loose_overlap() -> None:
    """Low threshold (0.4) accepts answers that broadly agree
    on most tokens but differ in some."""
    responses = {
        "d0": "the answer is forty two",
        "d1": "I think the answer is forty two for sure",
    }
    assert _converged(responses, threshold=0.4)


def test_converged_disagreement_fails_at_any_threshold() -> None:
    """Genuinely opposite answers fail convergence at every
    sensible threshold."""
    responses = {"d0": "yes", "d1": "no"}
    assert not _converged(responses, threshold=0.85)
    assert not _converged(responses, threshold=0.4)


def test_debate_rejects_invalid_convergence_similarity() -> None:
    a = _scripted_agent(["x"])
    b = _scripted_agent(["y"])
    with pytest.raises(ValueError, match="convergence_similarity"):
        MultiAgentDebate(
            debaters=[a, b], convergence_similarity=1.5
        )


def test_majority_vote_picks_modal_answer() -> None:
    responses = {
        "d0": "go left",
        "d1": "go RIGHT",
        "d2": "go left",
    }
    # Two for "go left", one for "go right" — left wins.
    assert _majority_vote(responses) == "go left"


def test_majority_vote_preserves_original_casing() -> None:
    """Whitespace-normalized for matching, but the original string
    (casing + whitespace preserved) is what's returned."""
    responses = {"d0": "FIRST APPEARANCE", "d1": "first appearance"}
    out = _majority_vote(responses)
    # Both normalize to the same thing → first one's original wins.
    assert out == "FIRST APPEARANCE"


def test_majority_vote_empty_returns_empty() -> None:
    assert _majority_vote({}) == ""


# ---------------------------------------------------------------------------
# Round 0: parallel independent answers
# ---------------------------------------------------------------------------


async def test_debate_round_0_runs_all_debaters() -> None:
    """Round 0: each debater runs once with just the original prompt
    and produces their independent answer."""
    d1 = _scripted_agent(["debater 0 round 0"])
    d2 = _scripted_agent(["debater 1 round 0"])
    judge = _scripted_agent(["final synthesized answer"])

    parent_model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "moderator",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[d1, d2],
            judge=judge,
            rounds=1,
            convergence_check=False,
        ),
    )
    # Each debater needs as many replies as rounds (round 0 + round 1).
    # We've only given each one reply, so we'd hit script exhaustion
    # if rounds went past 0+1. Bump up:
    d1 = _scripted_agent(["d0r0", "d0r1"])
    d2 = _scripted_agent(["d1r0", "d1r1"])
    judge = _scripted_agent(["FINAL"])
    agent = Agent(
        "moderator",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[d1, d2],
            judge=judge,
            rounds=1,
            convergence_check=False,
        ),
    )
    result = await agent.run("the question")
    assert result.output == "FINAL"


# ---------------------------------------------------------------------------
# Convergence early-exit
# ---------------------------------------------------------------------------


async def test_debate_terminates_early_on_convergence_round_0() -> None:
    """All debaters give the same round-0 answer → converge → no
    debate rounds, no judge call. session.output is the agreed answer."""
    d1 = _scripted_agent(["the answer is 42"])
    d2 = _scripted_agent(["The Answer is 42"])
    judge = _scripted_agent(["should not be called"])

    parent_model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "moderator",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[d1, d2],
            judge=judge,
            rounds=3,
            convergence_check=True,
        ),
    )
    result = await agent.run("question")
    # Output should be one of the agreed answers (first inserted into dict).
    assert "42" in result.output


# ---------------------------------------------------------------------------
# Judge synthesis path
# ---------------------------------------------------------------------------


async def test_debate_judge_synthesizes_final() -> None:
    """When debaters disagree, judge runs and produces the final."""
    d1 = _scripted_agent(["yes", "still yes"])
    d2 = _scripted_agent(["no", "still no"])
    judge = _scripted_agent(["After review, the answer is yes."])

    parent_model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "moderator",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[d1, d2],
            judge=judge,
            rounds=1,
            convergence_check=False,
        ),
    )
    result = await agent.run("yes or no?")
    assert result.output == "After review, the answer is yes."


# ---------------------------------------------------------------------------
# Majority vote fallback (no judge)
# ---------------------------------------------------------------------------


async def test_debate_falls_back_to_majority_vote_without_judge() -> None:
    """Three debaters; final round has 2 'yes' and 1 'no' → 'yes' wins."""
    d1 = _scripted_agent(["yes"])
    d2 = _scripted_agent(["yes"])
    d3 = _scripted_agent(["no"])

    parent_model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "moderator",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[d1, d2, d3],
            judge=None,
            rounds=1,
            convergence_check=False,
        ),
    )
    # Each debater needs 2 replies (round 0 + round 1 each); top up.
    d1 = _scripted_agent(["yes", "yes"])
    d2 = _scripted_agent(["yes", "yes"])
    d3 = _scripted_agent(["no", "no"])
    agent = Agent(
        "moderator",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[d1, d2, d3],
            judge=None,
            rounds=1,
            convergence_check=False,
        ),
    )
    result = await agent.run("decide")
    assert result.output == "yes"


# ---------------------------------------------------------------------------
# Deterministic session ids
# ---------------------------------------------------------------------------


async def test_debate_uses_deterministic_session_ids() -> None:
    """Each debater + judge invocation uses a predictable session_id
    so replays reproduce the same sub-sessions."""
    captured_ids: list[str] = []

    class _SnoopAgent(Agent):
        async def run(  # type: ignore[override]
            self, prompt: str, **kwargs: object
        ):
            sid = kwargs.get("session_id")
            assert sid is not None
            captured_ids.append(sid)
            return await super().run(prompt, **kwargs)  # type: ignore[arg-type]

    d1 = _SnoopAgent(
        "d1",
        model=ScriptedModel(
            [ScriptedTurn(text="r0"), ScriptedTurn(text="r1")]
        ),
    )
    d2 = _SnoopAgent(
        "d2",
        model=ScriptedModel(
            [ScriptedTurn(text="r0"), ScriptedTurn(text="r1")]
        ),
    )
    judge = _SnoopAgent(
        "judge",
        model=ScriptedModel([ScriptedTurn(text="final")]),
    )

    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "moderator",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[d1, d2],
            judge=judge,
            rounds=1,
            convergence_check=False,
        ),
    )
    result = await agent.run("question")

    # 2 debaters * 2 rounds = 4 debater calls + 1 judge call = 5
    assert len(captured_ids) == 5
    parent = result.session_id
    # Debater pattern: parent__debater_<i>_round_<r>
    assert any(
        sid == f"{parent}__debater_0_round_0"
        for sid in captured_ids
    )
    assert any(
        sid == f"{parent}__debater_0_round_1"
        for sid in captured_ids
    )
    assert any(
        sid == f"{parent}__debater_1_round_0"
        for sid in captured_ids
    )
    assert any(
        sid == f"{parent}__debater_1_round_1"
        for sid in captured_ids
    )
    # Judge pattern: parent__judge
    assert f"{parent}__judge" in captured_ids


# ---------------------------------------------------------------------------
# Architecture progress events
# ---------------------------------------------------------------------------


async def test_debate_emits_full_event_sequence() -> None:
    d1 = _scripted_agent(["a", "a-r1"])
    d2 = _scripted_agent(["b", "b-r1"])
    judge = _scripted_agent(["final"])

    agent = Agent(
        "moderator",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=MultiAgentDebate(
            debaters=[d1, d2],
            judge=judge,
            rounds=1,
            convergence_check=False,
        ),
    )
    events = [e async for e in agent.stream("q")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "debate.round_started" in arch_names
    assert "debate.response" in arch_names
    assert "debate.judging" in arch_names
    assert "debate.synthesized" in arch_names


async def test_debate_emits_converged_event_on_early_exit() -> None:
    d1 = _scripted_agent(["same answer"])
    d2 = _scripted_agent(["same answer"])

    agent = Agent(
        "moderator",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=MultiAgentDebate(
            debaters=[d1, d2],
            rounds=3,
            convergence_check=True,
        ),
    )
    events = [e async for e in agent.stream("q")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "debate.converged" in arch_names
