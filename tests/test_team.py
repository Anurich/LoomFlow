"""Tests for the Team facade + run_architecture standalone helper.

Covers:

* Each Team builder produces an Agent whose architecture is the
  expected concrete class.
* Builders forward Agent kwargs (memory, permissions, audit_log,
  budget, hooks, tools, runtime) through to the constructed Agent.
* Team.supervisor's worker registry is mutable post-construction
  via Supervisor.add_worker / remove_worker.
* run_architecture builds a minimal shell and runs an Architecture
  end-to-end without the caller writing Agent boilerplate.
* Recursive composition still works:
  Reflexion(base=Supervisor(workers=...)) wrapped in an Agent runs
  exactly the same way as Team.supervisor — both are equivalent.
"""

from __future__ import annotations

import pytest

from jeevesagent import (
    ActorCritic,
    Agent,
    BlackboardArchitecture,
    InMemoryAuditLog,
    InMemoryMemory,
    MultiAgentDebate,
    Reflexion,
    Router,
    RouterRoute,
    ScriptedModel,
    ScriptedTurn,
    StandardPermissions,
    Supervisor,
    Swarm,
    Team,
    run_architecture,
)
from jeevesagent.core.types import ToolCall

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scripted(text: str = "ok", instructions: str = "test") -> Agent:
    return Agent(
        instructions,
        model=ScriptedModel([ScriptedTurn(text=text)]),
    )


# ---------------------------------------------------------------------------
# Team builders return Agent with the expected architecture
# ---------------------------------------------------------------------------


def test_team_supervisor_returns_agent_with_supervisor_arch() -> None:
    a, b = _scripted("a"), _scripted("b")
    team = Team.supervisor(workers={"a": a, "b": b}, model="echo")
    assert isinstance(team, Agent)
    assert isinstance(team.architecture, Supervisor)
    assert set(team.architecture.declared_workers().keys()) == {"a", "b"}


def test_team_swarm_returns_agent_with_swarm_arch() -> None:
    a, b = _scripted("a"), _scripted("b")
    team = Team.swarm(
        agents={"a": a, "b": b}, entry_agent="a", model="echo"
    )
    assert isinstance(team.architecture, Swarm)


def test_team_router_returns_agent_with_router_arch() -> None:
    a = _scripted("a")
    routes = [RouterRoute(name="r1", description="x", agent=a)]
    team = Team.router(routes=routes, model="echo")
    assert isinstance(team.architecture, Router)


def test_team_debate_returns_agent_with_debate_arch() -> None:
    a, b = _scripted("yes"), _scripted("no")
    team = Team.debate(debaters=[a, b], rounds=1, model="echo")
    assert isinstance(team.architecture, MultiAgentDebate)


def test_team_actor_critic_returns_agent_with_actor_critic_arch() -> None:
    actor = _scripted("draft")
    critic = _scripted('{"score": 0.95, "issues": [], "summary": "ok"}')
    team = Team.actor_critic(actor=actor, critic=critic, model="echo")
    assert isinstance(team.architecture, ActorCritic)


def test_team_blackboard_returns_agent_with_blackboard_arch() -> None:
    a = _scripted("a")
    team = Team.blackboard(agents={"a": a}, model="echo")
    assert isinstance(team.architecture, BlackboardArchitecture)


# ---------------------------------------------------------------------------
# Team forwards Agent kwargs through (smoke check on a few)
# ---------------------------------------------------------------------------


def test_team_supervisor_forwards_agent_kwargs() -> None:
    a = _scripted("a")
    audit = InMemoryAuditLog()
    memory = InMemoryMemory()
    permissions = StandardPermissions()
    team = Team.supervisor(
        workers={"a": a},
        instructions="manage",
        model="echo",
        audit_log=audit,
        memory=memory,
        permissions=permissions,
    )
    assert team.instructions == "manage"
    assert team.memory is memory
    assert team.permissions is permissions


# ---------------------------------------------------------------------------
# Equivalence: Team.supervisor == Agent(architecture=Supervisor(...))
# ---------------------------------------------------------------------------


async def test_team_supervisor_equivalent_to_nested_form() -> None:
    """End-to-end: a Team-built supervisor and the explicit nested
    form produce equivalent runs given identical inputs."""
    worker_a = _scripted("worker A says hello")

    parent_model_args = [
        ScriptedTurn(
            tool_calls=[
                ToolCall(
                    id="c1",
                    tool="delegate",
                    args={
                        "worker": "a",
                        "instructions": "say hi",
                    },
                )
            ]
        ),
        ScriptedTurn(text="manager wraps up"),
    ]

    team = Team.supervisor(
        workers={"a": worker_a},
        instructions="coordinator",
        model=ScriptedModel(list(parent_model_args)),
    )
    nested = Agent(
        "coordinator",
        model=ScriptedModel(list(parent_model_args)),
        architecture=Supervisor(workers={"a": worker_a}),
    )
    r_team = await team.run("task")
    r_nested = await nested.run("task")
    assert r_team.output == r_nested.output


# ---------------------------------------------------------------------------
# Worker registry mutation
# ---------------------------------------------------------------------------


def test_supervisor_add_worker_after_construction() -> None:
    a = _scripted("a")
    sup = Supervisor(workers={"a": a})
    assert set(sup.declared_workers()) == {"a"}

    b = _scripted("b")
    sup.add_worker("b", b)
    assert set(sup.declared_workers()) == {"a", "b"}


def test_supervisor_remove_worker_returns_agent() -> None:
    a, b = _scripted("a"), _scripted("b")
    sup = Supervisor(workers={"a": a, "b": b})
    removed = sup.remove_worker("a")
    assert removed is a
    assert set(sup.declared_workers()) == {"b"}


def test_supervisor_remove_unknown_worker_returns_none() -> None:
    a = _scripted("a")
    sup = Supervisor(workers={"a": a})
    assert sup.remove_worker("ghost") is None


def test_supervisor_add_worker_rejects_invalid_name() -> None:
    a, b = _scripted("a"), _scripted("b")
    sup = Supervisor(workers={"a": a})
    with pytest.raises(ValueError, match="identifier"):
        sup.add_worker("not a name", b)


async def test_supervisor_added_worker_is_callable_in_next_run() -> None:
    """add_worker between runs must register the new worker so the
    next run's delegate(<name>, ...) succeeds."""
    a = _scripted("a does it")
    sup = Supervisor(workers={"a": a})

    # First run delegates to "a".
    coordinator_model_1 = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={"worker": "a", "instructions": "go"},
                    )
                ]
            ),
            ScriptedTurn(text="done with a"),
        ]
    )
    agent_1 = Agent(
        "manager", model=coordinator_model_1, architecture=sup
    )
    result_1 = await agent_1.run("first")
    assert "done with a" in result_1.output

    # Add a new worker between runs, then delegate to it.
    b = _scripted("b does it")
    sup.add_worker("b", b)

    coordinator_model_2 = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        tool="delegate",
                        args={"worker": "b", "instructions": "go"},
                    )
                ]
            ),
            ScriptedTurn(text="done with b"),
        ]
    )
    agent_2 = Agent(
        "manager", model=coordinator_model_2, architecture=sup
    )
    result_2 = await agent_2.run("second")
    assert "done with b" in result_2.output


# ---------------------------------------------------------------------------
# run_architecture standalone helper
# ---------------------------------------------------------------------------


async def test_run_architecture_runs_orchestrator_with_minimal_shell() -> None:
    """run_architecture builds a minimal Agent shell so users can
    test/run an architecture without writing the Agent themselves."""
    worker = _scripted("worker output")
    sup = Supervisor(workers={"a": worker})
    coordinator_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={"worker": "a", "instructions": "go"},
                    )
                ]
            ),
            ScriptedTurn(text="forwarded"),
        ]
    )
    result = await run_architecture(
        sup,
        "do it",
        instructions="manage",
        model=coordinator_model,
    )
    assert "forwarded" in result.output


async def test_run_architecture_with_react_default() -> None:
    """The standalone helper works with any architecture, not just
    multi-agent ones — defaults compose normally."""
    from jeevesagent import ReAct

    model = ScriptedModel([ScriptedTurn(text="hi from solo agent")])
    result = await run_architecture(
        ReAct(),
        "say hi",
        instructions="solo",
        model=model,
    )
    assert "hi from solo agent" in result.output


# ---------------------------------------------------------------------------
# Recursive composition still works (the value prop our nested
# design retains over the sibling-only frameworks)
# ---------------------------------------------------------------------------


async def test_reflexion_can_wrap_supervisor() -> None:
    """The point of keeping the nested design: Reflexion(base=
    Supervisor(...)) is a one-liner. This test verifies the
    composition still constructs and the resulting Agent is
    runnable via the standard Agent.run() interface."""
    worker = _scripted("worker says X")

    # Reflexion wraps Supervisor; one bad attempt + one good.
    coordinator_model = ScriptedModel(
        [
            # Attempt 1: delegate, then text response
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={"worker": "a", "instructions": "go"},
                    )
                ]
            ),
            ScriptedTurn(text="bad first attempt"),
            # Evaluator + reflector
            ScriptedTurn(text="score: 0.3"),
            ScriptedTurn(text="lesson: do better"),
            # Attempt 2: another delegate, better response
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        tool="delegate",
                        args={"worker": "a", "instructions": "go"},
                    )
                ]
            ),
            ScriptedTurn(text="good second attempt"),
            # Evaluator
            ScriptedTurn(text="score: 0.95"),
        ]
    )

    agent = Agent(
        "manager",
        model=coordinator_model,
        architecture=Reflexion(
            base=Supervisor(workers={"a": worker}),
            max_attempts=2,
            threshold=0.8,
        ),
    )
    result = await agent.run("hard task")
    assert "good second attempt" in result.output
