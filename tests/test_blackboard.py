"""Blackboard architecture tests.

Covers:

* Protocol satisfaction; declared_workers exposes agents +
  optional coordinator + decider.
* Constructor validation: empty agents, max_rounds<1.
* :class:`Blackboard` state: post / render_for / public + private
  partitions.
* :func:`_parse_coordinator_decision` JSON parsing — clean JSON,
  markdown fences, malformed → safe default.
* Round-robin fallback (coordinator=None).
* LLM coordinator path (terminate / pick agent / unknown agent).
* Decider synthesis path (decider != None).
* Decider=None fallback: last "answer" entry, then last
  contribution, then empty.
* Architecture progress events.
"""

from __future__ import annotations

import pytest

from jeevesagent import (
    Agent,
    Architecture,
    Blackboard,
    BlackboardArchitecture,
    BlackboardEntry,
    ScriptedModel,
    ScriptedTurn,
)
from jeevesagent.architecture.blackboard import (
    _parse_coordinator_decision,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted_agent(replies: list[str], instructions: str = "test") -> Agent:
    return Agent(
        instructions,
        model=ScriptedModel([ScriptedTurn(text=r) for r in replies]),
    )


# ---------------------------------------------------------------------------
# Blackboard state object
# ---------------------------------------------------------------------------


def test_blackboard_post_appends_to_public() -> None:
    bb = Blackboard()
    bb.post("user", "the question")
    assert len(bb.public) == 1
    entry = bb.public[0]
    assert isinstance(entry, BlackboardEntry)
    assert entry.author == "user"
    assert entry.content == "the question"


def test_blackboard_post_private_routes_to_per_agent() -> None:
    bb = Blackboard()
    bb.post("alpha", "scratch", private_to="alpha")
    assert "alpha" in bb.private
    assert len(bb.public) == 0


def test_blackboard_render_for_includes_public_and_private() -> None:
    bb = Blackboard()
    bb.post("user", "Q1", kind="problem")
    bb.post("alpha", "draft thought", private_to="alpha")
    out = bb.render_for("alpha")
    assert "Public board" in out
    assert "Q1" in out
    assert "Your private notes" in out
    assert "draft thought" in out


def test_blackboard_render_for_other_agent_omits_others_private() -> None:
    bb = Blackboard()
    bb.post("alpha", "alpha-scratch", private_to="alpha")
    out = bb.render_for("beta")
    assert "alpha-scratch" not in out


def test_blackboard_render_empty_returns_empty_marker() -> None:
    assert Blackboard().render_for("anyone") == "(empty)"


# ---------------------------------------------------------------------------
# Coordinator parser
# ---------------------------------------------------------------------------


def test_parse_coordinator_decision_strict_json() -> None:
    text = (
        '{"terminate": false, "next_agent": "alpha", '
        '"instruction": "do the thing"}'
    )
    out = _parse_coordinator_decision(text)
    assert not out.terminate
    assert out.next_agent == "alpha"
    assert out.instruction == "do the thing"


def test_parse_coordinator_decision_with_markdown_fences() -> None:
    text = '```json\n{"terminate": true, "next_agent": null, "instruction": null}\n```'
    out = _parse_coordinator_decision(text)
    assert out.terminate
    assert out.next_agent is None


def test_parse_coordinator_decision_malformed_returns_safe_default() -> None:
    out = _parse_coordinator_decision("not json at all")
    assert not out.terminate
    assert out.next_agent is None  # round skipped


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_blackboard_satisfies_architecture_protocol() -> None:
    arch = BlackboardArchitecture(
        agents={"a": _scripted_agent(["x"])}
    )
    assert isinstance(arch, Architecture)


def test_blackboard_name_is_blackboard() -> None:
    arch = BlackboardArchitecture(
        agents={"a": _scripted_agent(["x"])}
    )
    assert arch.name == "blackboard"


def test_blackboard_declared_workers_exposes_all() -> None:
    a = _scripted_agent(["x"])
    coord = _scripted_agent(["coord"])
    decider = _scripted_agent(["decider"])
    arch = BlackboardArchitecture(
        agents={"a": a},
        coordinator=coord,
        decider=decider,
    )
    workers = arch.declared_workers()
    assert workers["a"] is a
    assert workers["__coordinator"] is coord
    assert workers["__decider"] is decider


def test_blackboard_rejects_empty_agents() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BlackboardArchitecture(agents={})


def test_blackboard_rejects_max_rounds_lt_1() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        BlackboardArchitecture(
            agents={"a": _scripted_agent(["x"])},
            max_rounds=0,
        )


# ---------------------------------------------------------------------------
# Round-robin fallback (no coordinator)
# ---------------------------------------------------------------------------


async def test_blackboard_round_robin_fallback() -> None:
    """Without a coordinator, agents take turns in dict-iteration
    order; max_rounds caps the loop."""
    a = _scripted_agent(["A's contribution"])
    b = _scripted_agent(["B's contribution"])

    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=BlackboardArchitecture(
            agents={"a": a, "b": b},
            max_rounds=2,  # one round each
        ),
    )
    result = await agent.run("the question")
    # No decider → falls back to the latest contribution.
    assert result.output == "B's contribution"


# ---------------------------------------------------------------------------
# LLM coordinator path
# ---------------------------------------------------------------------------


async def test_blackboard_coordinator_picks_agent_then_terminates() -> None:
    """Coordinator round 1 picks 'alpha', round 2 terminates.
    Decider produces final answer."""
    alpha = _scripted_agent(["alpha says: the answer is X"])
    coord = _scripted_agent(
        [
            '{"terminate": false, "next_agent": "alpha", "instruction": null}',
            '{"terminate": true, "next_agent": null, "instruction": null}',
        ]
    )
    decider = _scripted_agent(["FINAL: X (synthesized)"])

    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=BlackboardArchitecture(
            agents={"alpha": alpha},
            coordinator=coord,
            decider=decider,
            max_rounds=5,
        ),
    )
    result = await agent.run("question")
    assert result.output == "FINAL: X (synthesized)"


async def test_blackboard_coordinator_unknown_agent_skips_round() -> None:
    """Coordinator picks an agent name that doesn't exist; the round
    is skipped, an error gets posted to the board, and we move on
    (no crash)."""
    alpha = _scripted_agent(["alpha contribution"])
    coord = _scripted_agent(
        [
            '{"terminate": false, "next_agent": "ghost", "instruction": null}',
            '{"terminate": false, "next_agent": "alpha", "instruction": null}',
            '{"terminate": true, "next_agent": null, "instruction": null}',
        ]
    )

    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=BlackboardArchitecture(
            agents={"alpha": alpha},
            coordinator=coord,
            max_rounds=5,
        ),
    )
    events = [e async for e in agent.stream("q")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "blackboard.unknown_agent" in arch_names
    # alpha still ran (round 2 picked it correctly).
    assert "blackboard.contribution" in arch_names


# ---------------------------------------------------------------------------
# Decider fallback
# ---------------------------------------------------------------------------


async def test_blackboard_decider_none_returns_last_contribution() -> None:
    a = _scripted_agent(["my contribution"])
    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=BlackboardArchitecture(
            agents={"a": a},
            max_rounds=1,
        ),
    )
    result = await agent.run("q")
    assert result.output == "my contribution"


async def test_blackboard_decider_none_with_no_contributions_returns_empty() -> None:
    """Coordinator immediately terminates → no contributions → no
    answer-kind entries → empty output."""
    coord = _scripted_agent(
        ['{"terminate": true, "next_agent": null, "instruction": null}']
    )
    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=BlackboardArchitecture(
            agents={"a": _scripted_agent(["never"])},
            coordinator=coord,
        ),
    )
    result = await agent.run("q")
    assert result.output == ""


# ---------------------------------------------------------------------------
# Architecture events
# ---------------------------------------------------------------------------


async def test_blackboard_emits_full_event_sequence() -> None:
    a = _scripted_agent(["A's bit"])

    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=BlackboardArchitecture(
            agents={"a": a},
            max_rounds=1,
        ),
    )
    events = [e async for e in agent.stream("q")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "blackboard.started" in arch_names
    assert "blackboard.coordinator_decided" in arch_names
    assert "blackboard.invoking" in arch_names
    assert "blackboard.contribution" in arch_names
    assert "blackboard.completed" in arch_names
