"""ActorCritic architecture tests.

Covers:

* Protocol satisfaction; ``declared_workers`` exposes both actor
  and critic.
* Constructor validation: invalid ``max_rounds`` / threshold.
* :func:`_parse_critique`: raw JSON, JSON inside markdown fences,
  regex-only fallback (score extracted, full text as single issue),
  unparseable text → 0.0 score with text-as-issue.
* Single-round approval (critic returns score >= threshold on round
  1 → no refine pass).
* Multi-round refinement (critic finds issues, actor refines, then
  critic approves).
* ``max_rounds`` enforcement (never converges, returns latest output).
* Architecture progress events surface (actor_started /
  actor_completed / critic_started / critique / approved /
  max_rounds_reached).
* Deterministic actor / critic session ids per round (replay-safe).
* Actor / critic interruption propagates to parent.
"""

from __future__ import annotations

import pytest

from jeevesagent import (
    ActorCritic,
    Agent,
    Architecture,
    ScriptedModel,
    ScriptedTurn,
)
from jeevesagent.architecture.actor_critic import (
    CriticOutput,
    _parse_critique,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted_agent(replies: list[str], instructions: str = "test") -> Agent:
    """Build an Agent that returns the given list of fixed text replies in order."""
    turns = [ScriptedTurn(text=r) for r in replies]
    return Agent(instructions, model=ScriptedModel(turns))


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_actor_critic_satisfies_architecture_protocol() -> None:
    actor = _scripted_agent(["x"])
    critic = _scripted_agent(['{"issues": [], "score": 1.0, "summary": ""}'])
    arch = ActorCritic(actor=actor, critic=critic)
    assert isinstance(arch, Architecture)


def test_actor_critic_name() -> None:
    actor = _scripted_agent(["x"])
    critic = _scripted_agent(["x"])
    assert ActorCritic(actor=actor, critic=critic).name == "actor-critic"


def test_actor_critic_declared_workers_exposes_both() -> None:
    actor = _scripted_agent(["x"])
    critic = _scripted_agent(["x"])
    arch = ActorCritic(actor=actor, critic=critic)
    workers = arch.declared_workers()
    assert workers == {"actor": actor, "critic": critic}


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_actor_critic_rejects_max_rounds_lt_1() -> None:
    actor = _scripted_agent(["x"])
    critic = _scripted_agent(["x"])
    with pytest.raises(ValueError, match="max_rounds"):
        ActorCritic(actor=actor, critic=critic, max_rounds=0)


def test_actor_critic_rejects_threshold_outside_unit_interval() -> None:
    actor = _scripted_agent(["x"])
    critic = _scripted_agent(["x"])
    with pytest.raises(ValueError, match="approval_threshold"):
        ActorCritic(actor=actor, critic=critic, approval_threshold=1.5)
    with pytest.raises(ValueError, match="approval_threshold"):
        ActorCritic(actor=actor, critic=critic, approval_threshold=-0.1)


# ---------------------------------------------------------------------------
# Critique parser
# ---------------------------------------------------------------------------


def test_parse_critique_pure_json() -> None:
    text = '{"issues": ["a", "b"], "score": 0.6, "summary": "needs work"}'
    out = _parse_critique(text)
    assert out.issues == ["a", "b"]
    assert out.score == 0.6
    assert out.summary == "needs work"


def test_parse_critique_json_in_markdown_fences() -> None:
    text = '```json\n{"issues": ["a"], "score": 0.4, "summary": "x"}\n```'
    out = _parse_critique(text)
    assert out.issues == ["a"]
    assert out.score == 0.4


def test_parse_critique_clamps_oob_score_in_json() -> None:
    text = '{"issues": [], "score": 1.5, "summary": ""}'
    out = _parse_critique(text)
    assert out.score == 1.0


def test_parse_critique_regex_fallback_for_loose_text() -> None:
    text = "I find some issues. score: 0.42"
    out = _parse_critique(text)
    assert out.score == 0.42
    # Whole text becomes one issue when JSON parse fails.
    assert len(out.issues) == 1
    assert "issues" in out.issues[0]


def test_parse_critique_no_score_returns_zero() -> None:
    text = "no number anywhere here"
    out = _parse_critique(text)
    assert out.score == 0.0
    assert out.issues == [text]


def test_parse_critique_empty_text_returns_default() -> None:
    out = _parse_critique("")
    assert out.score == 0.0
    assert out.issues == []
    assert isinstance(out, CriticOutput)


# ---------------------------------------------------------------------------
# Single-round approval (no refinement)
# ---------------------------------------------------------------------------


async def test_actor_critic_approves_round_1_no_refine_call() -> None:
    """Critic on round 1 returns score 0.95 ≥ threshold 0.9 →
    terminate. Actor's round-0 output is the final answer; the actor
    is NOT called a second time. We verify by giving the actor only
    one scripted reply — a second call would fail."""
    actor = _scripted_agent(["round 0 output"])
    critic = _scripted_agent(
        ['{"issues": [], "score": 0.95, "summary": "looks good"}']
    )
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never reached")]),
        architecture=ActorCritic(
            actor=actor, critic=critic, max_rounds=3
        ),
    )
    result = await agent.run("a task")
    assert result.output == "round 0 output"


# ---------------------------------------------------------------------------
# Multi-round refinement
# ---------------------------------------------------------------------------


async def test_actor_critic_refines_on_low_score_then_approves() -> None:
    """Round 0 actor → low score from critic → actor refines →
    critic round 2 approves. Final output is the refined version."""
    actor = _scripted_agent(["draft v0", "polished v1"])
    critic = _scripted_agent(
        [
            '{"issues": ["too short"], "score": 0.4, "summary": "needs work"}',
            '{"issues": [], "score": 0.95, "summary": "good"}',
        ]
    )
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never reached")]),
        architecture=ActorCritic(
            actor=actor, critic=critic, max_rounds=3
        ),
    )
    result = await agent.run("write a thing")
    assert result.output == "polished v1"


# ---------------------------------------------------------------------------
# Max rounds enforcement
# ---------------------------------------------------------------------------


async def test_actor_critic_returns_latest_output_at_max_rounds() -> None:
    """Critic never approves → after max_rounds, return the latest
    actor output. The session is not interrupted; just gave up."""
    actor = _scripted_agent(["v0", "v1", "v2"])
    # max_rounds=2 means: critic R1 + actor refine R1 + critic R2 +
    # actor refine R2 (but critic R3 doesn't run; max_rounds_reached
    # fires after critic R2 before refining again). Need 2 critic calls.
    critic = _scripted_agent(
        [
            '{"issues": ["i1"], "score": 0.3, "summary": ""}',
            '{"issues": ["i2"], "score": 0.4, "summary": ""}',
        ]
    )
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never reached")]),
        architecture=ActorCritic(
            actor=actor, critic=critic, max_rounds=2
        ),
    )
    result = await agent.run("task")
    # v0 (R0) + critic R1 → refine to v1, critic R2 → max_rounds hit.
    # So latest output is v1, NOT v2 (refiner not called after R2's
    # critique because we hit max_rounds first).
    assert result.output == "v1"
    assert not result.interrupted


# ---------------------------------------------------------------------------
# Architecture progress events
# ---------------------------------------------------------------------------


async def test_actor_critic_emits_full_event_sequence() -> None:
    actor = _scripted_agent(["v0", "v1"])
    critic = _scripted_agent(
        [
            '{"issues": ["x"], "score": 0.5, "summary": ""}',
            '{"issues": [], "score": 0.95, "summary": ""}',
        ]
    )
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=ActorCritic(
            actor=actor, critic=critic, max_rounds=3
        ),
    )
    events = [e async for e in agent.stream("t")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    # Expected sequence:
    # actor_started (round 0 generate)
    # actor_completed (round 0 generate)
    # critic_started (round 1)
    # critique (round 1, score 0.5)
    # actor_started (round 1 refine)
    # actor_completed (round 1 refine)
    # critic_started (round 2)
    # critique (round 2, score 0.95)
    # approved (round 2)
    assert "actor_critic.actor_started" in arch_names
    assert "actor_critic.actor_completed" in arch_names
    assert "actor_critic.critic_started" in arch_names
    assert "actor_critic.critique" in arch_names
    assert "actor_critic.approved" in arch_names


async def test_actor_critic_emits_max_rounds_reached_event() -> None:
    actor = _scripted_agent(["v0", "v1"])
    critic = _scripted_agent(
        [
            '{"issues": ["i"], "score": 0.3, "summary": ""}',
            '{"issues": ["i"], "score": 0.3, "summary": ""}',
        ]
    )
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=ActorCritic(
            actor=actor, critic=critic, max_rounds=2
        ),
    )
    events = [e async for e in agent.stream("t")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "actor_critic.max_rounds_reached" in arch_names
    assert "actor_critic.approved" not in arch_names


# ---------------------------------------------------------------------------
# Deterministic session ids
# ---------------------------------------------------------------------------


async def test_actor_critic_uses_deterministic_session_ids() -> None:
    """Each actor / critic invocation should land on a session id of
    the form ``{parent}__actor_<round>`` / ``{parent}__critic_<round>``
    so replays reproduce the same sub-sessions."""
    captured_ids: list[tuple[str, str]] = []

    class _SnoopAgent(Agent):
        def __init__(self, role: str, replies: list[str]) -> None:
            super().__init__(
                role, model=ScriptedModel([ScriptedTurn(text=r) for r in replies])
            )
            self._role = role

        async def run(  # type: ignore[override]
            self, prompt: str, *, session_id: str | None = None
        ):
            assert session_id is not None
            captured_ids.append((self._role, session_id))
            return await super().run(prompt, session_id=session_id)

    actor = _SnoopAgent("actor", ["v0", "v1"])
    critic = _SnoopAgent(
        "critic",
        [
            '{"issues": ["i"], "score": 0.4, "summary": ""}',
            '{"issues": [], "score": 0.95, "summary": ""}',
        ],
    )
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=ActorCritic(
            actor=actor, critic=critic, max_rounds=3
        ),
    )
    result = await agent.run("t")

    # Three sub-sessions: actor R0, critic R1, actor R1, critic R2.
    assert len(captured_ids) == 4
    actor_sids = [sid for role, sid in captured_ids if role == "actor"]
    critic_sids = [sid for role, sid in captured_ids if role == "critic"]
    # Actor R0 and R1 should differ.
    assert len(set(actor_sids)) == 2
    assert len(set(critic_sids)) == 2
    # All should start with the parent's session id.
    parent = result.session_id
    for _role, sid in captured_ids:
        assert sid.startswith(parent + "__")
    # And include the right role label.
    assert any("__actor_0" in sid for sid in actor_sids)
    assert any("__actor_1" in sid for sid in actor_sids)
    assert any("__critic_1" in sid for sid in critic_sids)
    assert any("__critic_2" in sid for sid in critic_sids)


# ---------------------------------------------------------------------------
# Interruption propagation
# ---------------------------------------------------------------------------


async def test_actor_critic_propagates_actor_interruption() -> None:
    """Actor with max_turns=0 interrupts on first turn → ActorCritic
    reports the parent session as interrupted with a namespaced
    reason."""
    actor = Agent(
        "actor",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        max_turns=0,
    )
    critic = _scripted_agent(["never"])
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=ActorCritic(actor=actor, critic=critic),
    )
    result = await agent.run("anything")
    assert result.interrupted
    assert result.interruption_reason is not None
    assert result.interruption_reason.startswith("actor:round_0:")


async def test_actor_critic_propagates_critic_interruption() -> None:
    """Critic interrupted on round 1 → parent reports the
    namespaced reason."""
    actor = _scripted_agent(["v0"])
    critic = Agent(
        "critic",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        max_turns=0,
    )
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=ActorCritic(actor=actor, critic=critic),
    )
    result = await agent.run("anything")
    assert result.interrupted
    assert result.interruption_reason is not None
    assert result.interruption_reason.startswith("critic:round_1:")
