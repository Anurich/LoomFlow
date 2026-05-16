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

from loomflow import (
    Agent,
    InMemoryAuditLog,
    InMemoryMemory,
    ScriptedModel,
    ScriptedTurn,
    StandardPermissions,
)
from loomflow.architecture import (
    ActorCritic,
    BlackboardArchitecture,
    MultiAgentDebate,
    Reflexion,
    Router,
    RouterRoute,
    Supervisor,
    Swarm,
)
from loomflow.core.types import ToolCall
from loomflow.team import Team, run_architecture

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
# prompt_caching= forwarding through every Team.* builder (0.10.12)
# ---------------------------------------------------------------------------
#
# Until 0.10.12 the coordinator Agent built by ``Team.*`` could not have
# prompt caching enabled — only workers (constructed as plain Agents
# upstream) could. Mirrors the ``stop_hooks=`` forwarding added in
# 0.10.10. The check below confirms the kwarg lands on the coordinator's
# ``_prompt_caching`` and resolves to ``enabled=True``.


def _assert_caching_enabled(coord: Agent) -> None:
    cfg = coord._prompt_caching
    assert cfg.enabled is True, (
        f"coordinator prompt_caching did not propagate: {cfg!r}"
    )


def test_team_supervisor_forwards_prompt_caching() -> None:
    team = Team.supervisor(
        workers={"a": _scripted("a")}, model="echo", prompt_caching=True
    )
    _assert_caching_enabled(team)


def test_team_swarm_forwards_prompt_caching() -> None:
    team = Team.swarm(
        agents={"a": _scripted("a"), "b": _scripted("b")},
        entry_agent="a",
        model="echo",
        prompt_caching=True,
    )
    _assert_caching_enabled(team)


def test_team_router_forwards_prompt_caching() -> None:
    team = Team.router(
        routes=[RouterRoute(name="r1", description="x", agent=_scripted("a"))],
        model="echo",
        prompt_caching=True,
    )
    _assert_caching_enabled(team)


def test_team_debate_forwards_prompt_caching() -> None:
    team = Team.debate(
        debaters=[_scripted("a"), _scripted("b")],
        rounds=1,
        model="echo",
        prompt_caching=True,
    )
    _assert_caching_enabled(team)


def test_team_actor_critic_forwards_prompt_caching() -> None:
    team = Team.actor_critic(
        actor=_scripted("draft"),
        critic=_scripted('{"score": 1.0, "issues": [], "summary": "ok"}'),
        model="echo",
        prompt_caching=True,
    )
    _assert_caching_enabled(team)


def test_team_blackboard_forwards_prompt_caching() -> None:
    team = Team.blackboard(
        agents={"a": _scripted("a")}, model="echo", prompt_caching=True
    )
    _assert_caching_enabled(team)


def test_team_supervisor_accepts_dict_form_prompt_caching() -> None:
    """Dict shape (e.g. ``{"enabled": True, "ttl": "1h"}``) propagates
    through and resolves to a populated PromptCacheConfig."""
    team = Team.supervisor(
        workers={"a": _scripted("a")},
        model="echo",
        prompt_caching={"enabled": True, "ttl": "1h"},
    )
    cfg = team._prompt_caching
    assert cfg.enabled is True
    assert cfg.ttl == "1h"


# ---------------------------------------------------------------------------
# tool_result_summarizer= forwarding through every Team.* builder (0.10.15)
# ---------------------------------------------------------------------------
#
# Same papercut pattern as prompt_caching= (0.10.12) and stop_hooks=
# (0.10.10). The 0.10.14 feature lands as ``Agent`` kwargs; 0.10.15
# plumbs them through the Team.* builders so loom-code (and any
# other Team.supervisor consumer) can wire the summariser onto its
# coordinator without monkey-patching ``coord._tool_result_summarizer``
# post-construction.


def _assert_summarizer_wired(coord: Agent, *, threshold: int) -> None:
    assert coord._tool_result_summarizer is not None, (
        "tool_result_summarizer did not propagate to coordinator"
    )
    assert coord._tool_result_summary_threshold == threshold


def test_team_supervisor_forwards_tool_result_summarizer() -> None:
    summariser = _scripted("compressed")
    team = Team.supervisor(
        workers={"a": _scripted("a")},
        model="echo",
        tool_result_summarizer=summariser,
        tool_result_summary_threshold=250,
    )
    _assert_summarizer_wired(team, threshold=250)


def test_team_swarm_forwards_tool_result_summarizer() -> None:
    team = Team.swarm(
        agents={"a": _scripted("a"), "b": _scripted("b")},
        entry_agent="a",
        model="echo",
        tool_result_summarizer="echo",
    )
    _assert_summarizer_wired(team, threshold=500)


def test_team_router_forwards_tool_result_summarizer() -> None:
    team = Team.router(
        routes=[RouterRoute(name="r1", description="x", agent=_scripted("a"))],
        model="echo",
        tool_result_summarizer="echo",
    )
    _assert_summarizer_wired(team, threshold=500)


def test_team_debate_forwards_tool_result_summarizer() -> None:
    team = Team.debate(
        debaters=[_scripted("a"), _scripted("b")],
        rounds=1,
        model="echo",
        tool_result_summarizer="echo",
    )
    _assert_summarizer_wired(team, threshold=500)


def test_team_actor_critic_forwards_tool_result_summarizer() -> None:
    team = Team.actor_critic(
        actor=_scripted("draft"),
        critic=_scripted('{"score": 1.0, "issues": [], "summary": "ok"}'),
        model="echo",
        tool_result_summarizer="echo",
    )
    _assert_summarizer_wired(team, threshold=500)


def test_team_blackboard_forwards_tool_result_summarizer() -> None:
    team = Team.blackboard(
        agents={"a": _scripted("a")},
        model="echo",
        tool_result_summarizer="echo",
    )
    _assert_summarizer_wired(team, threshold=500)


def test_team_supervisor_summarizer_default_none() -> None:
    """No kwarg → no summariser wired (back-compat). Confirms the
    new kwarg doesn't accidentally default to anything truthy."""
    team = Team.supervisor(workers={"a": _scripted("a")}, model="echo")
    assert team._tool_result_summarizer is None
    assert team._tool_result_summary_threshold == 500


# ---------------------------------------------------------------------------
# Comprehensive Agent-kwarg forwarding (catch-all)
# ---------------------------------------------------------------------------
#
# Anything Agent accepts as a kwarg SHOULD round-trip through every
# Team.* builder. Historical pattern: each new Agent kwarg has shipped
# without Team forwarding, then a follow-up release patched it
# (0.10.10 stop_hooks=, 0.10.12 prompt_caching=, 0.10.13/15
# tool_result_summarizer=). This test file enumerates the FULL set so
# regressions surface in CI.
#
# Tested via direct introspection of the private ``_*`` attributes
# the kwargs land on — that's the load-bearing contract; the kwargs
# themselves are just developer ergonomics.


def _scripted_for_tests() -> Agent:
    return _scripted("ok")


@pytest.fixture
def supervisor_with_all_kwargs() -> Agent:
    """A coordinator built with every currently-forwardable Agent
    kwarg set to a non-default value. Tests below read the
    landed attribute to confirm propagation."""
    cheap = _scripted("compressed")
    return Team.supervisor(
        workers={"a": _scripted_for_tests()},
        model="echo",
        # token-opt knobs (0.10.13 — the prompted fix)
        snip_window=8,
        auto_compact_at_tokens=50_000,
        auto_compact_summariser=cheap,
        auto_compact_keep_recent_turns=2,
        tool_result_summarizer=cheap,
        tool_result_summary_threshold=250,
        # historical-gap knobs being closed in this same release
        retry_policy=None,  # accepts the kwarg shape; None is valid
        auto_extract=False,
        response_tone="terse",
        effort="medium",
        strict_effort=False,
    )


def test_supervisor_forwards_snip_window(
    supervisor_with_all_kwargs: Agent,
) -> None:
    assert supervisor_with_all_kwargs._snip_window == 8


def test_supervisor_forwards_auto_compact_at_tokens(
    supervisor_with_all_kwargs: Agent,
) -> None:
    assert (
        supervisor_with_all_kwargs._auto_compact_at_tokens == 50_000
    )


def test_supervisor_forwards_auto_compact_summariser(
    supervisor_with_all_kwargs: Agent,
) -> None:
    assert (
        supervisor_with_all_kwargs._auto_compact_summariser is not None
    )


def test_supervisor_forwards_auto_compact_keep_recent_turns(
    supervisor_with_all_kwargs: Agent,
) -> None:
    assert (
        supervisor_with_all_kwargs._auto_compact_keep_recent_turns
        == 2
    )


def test_supervisor_forwards_response_tone(
    supervisor_with_all_kwargs: Agent,
) -> None:
    assert (
        supervisor_with_all_kwargs._default_response_tone == "terse"
    )


def test_supervisor_forwards_effort(
    supervisor_with_all_kwargs: Agent,
) -> None:
    assert supervisor_with_all_kwargs._default_effort == "medium"


def test_supervisor_forwards_strict_effort(
    supervisor_with_all_kwargs: Agent,
) -> None:
    assert supervisor_with_all_kwargs._strict_effort is False


def test_supervisor_accepts_auto_extract_without_raise() -> None:
    """``auto_extract`` gates memory-wrapper construction; it
    doesn't land on a stored attribute we can introspect post-
    facto. Smoke-test: the kwarg is ACCEPTED by the builder
    (no TypeError), which is the actual gap this release closes."""
    team = Team.supervisor(
        workers={"a": _scripted("a")},
        model="echo",
        auto_extract=False,
    )
    assert isinstance(team, Agent)


def test_supervisor_accepts_retry_policy_without_raise() -> None:
    """Same shape — ``retry_policy=`` is accepted; passing None
    is valid (means "default policy"). Verifies the kwarg is
    plumbed through, not its post-resolution semantics."""
    team = Team.supervisor(
        workers={"a": _scripted("a")},
        model="echo",
        retry_policy=None,
    )
    assert isinstance(team, Agent)


def test_supervisor_accepts_approval_handler_and_secrets() -> None:
    """The remaining two historical-gap kwargs."""

    async def _handler(call, user_id):  # type: ignore[no-untyped-def]
        return True

    team = Team.supervisor(
        workers={"a": _scripted("a")},
        model="echo",
        approval_handler=_handler,
        secrets=None,
    )
    assert isinstance(team, Agent)


def test_supervisor_default_snip_disabled() -> None:
    """Back-compat: omitting ``snip_window`` keeps the default 0
    (disabled). Mirrors the back-compat assertions for the other
    optional knobs."""
    team = Team.supervisor(workers={"a": _scripted("a")}, model="echo")
    assert team._snip_window == 0
    assert team._auto_compact_at_tokens is None


# Spot-check the OTHER 5 builders accept the same new kwargs without
# raising — full attribute introspection on each would be 5x the
# above; the round-trip on supervisor + this surface test catches
# any silent gap in the replace_all forwarding pattern.


def test_swarm_accepts_snip_and_compact_kwargs() -> None:
    team = Team.swarm(
        agents={"a": _scripted("a"), "b": _scripted("b")},
        entry_agent="a",
        model="echo",
        snip_window=5,
        auto_compact_at_tokens=10_000,
    )
    assert team._snip_window == 5
    assert team._auto_compact_at_tokens == 10_000


def test_router_accepts_snip_and_compact_kwargs() -> None:
    team = Team.router(
        routes=[RouterRoute(name="r1", description="x", agent=_scripted("a"))],
        model="echo",
        snip_window=5,
        auto_compact_at_tokens=10_000,
    )
    assert team._snip_window == 5
    assert team._auto_compact_at_tokens == 10_000


def test_debate_accepts_snip_and_compact_kwargs() -> None:
    team = Team.debate(
        debaters=[_scripted("a"), _scripted("b")],
        rounds=1,
        model="echo",
        snip_window=5,
        auto_compact_at_tokens=10_000,
    )
    assert team._snip_window == 5
    assert team._auto_compact_at_tokens == 10_000


def test_actor_critic_accepts_snip_and_compact_kwargs() -> None:
    team = Team.actor_critic(
        actor=_scripted("draft"),
        critic=_scripted('{"score": 1.0, "issues": [], "summary": "ok"}'),
        model="echo",
        snip_window=5,
        auto_compact_at_tokens=10_000,
    )
    assert team._snip_window == 5
    assert team._auto_compact_at_tokens == 10_000


def test_blackboard_accepts_snip_and_compact_kwargs() -> None:
    team = Team.blackboard(
        agents={"a": _scripted("a")},
        model="echo",
        snip_window=5,
        auto_compact_at_tokens=10_000,
    )
    assert team._snip_window == 5
    assert team._auto_compact_at_tokens == 10_000


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
# Sub-agent cost rollup — without this the parent's RunResult.cost_usd
# silently under-counts every architecture that uses SubagentInvocation
# (Supervisor, Swarm, Router, ActorCritic, Debate, Blackboard). Regression
# guard so the rollup helper stays wired.
# ---------------------------------------------------------------------------


async def test_supervisor_rolls_up_worker_costs_into_parent_result() -> None:
    """When the coordinator delegates to a worker, the worker's
    tokens + cost must be added to the team's RunResult — otherwise
    every consumer of ``Team.supervisor`` is shown a number that
    only reflects the coordinator's own model calls and silently
    omits the workers'."""
    from loomflow.core.types import Usage

    # Worker emits one turn worth ~$0.05 / 100 in / 20 out.
    worker = Agent(
        "worker",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    text="worker did it",
                    usage=Usage(
                        input_tokens=100, output_tokens=20, cost_usd=0.05
                    ),
                ),
            ]
        ),
    )
    # Coordinator: turn 1 delegates (50/10/$0.02), turn 2 finishes
    # (80/15/$0.01). Expected team total = 230/45/$0.08.
    coordinator_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={"worker": "w", "instructions": "go"},
                    )
                ],
                usage=Usage(
                    input_tokens=50, output_tokens=10, cost_usd=0.02
                ),
            ),
            ScriptedTurn(
                text="done",
                usage=Usage(
                    input_tokens=80, output_tokens=15, cost_usd=0.01
                ),
            ),
        ]
    )
    team = Team.supervisor(
        workers={"w": worker},
        instructions="manager",
        model=coordinator_model,
    )
    result = await team.run("kick off")

    # Coordinator's two turns + worker's one turn must all roll up.
    assert result.tokens_in == 50 + 80 + 100, (
        f"tokens_in={result.tokens_in} — worker tokens missing"
    )
    assert result.tokens_out == 10 + 15 + 20, (
        f"tokens_out={result.tokens_out} — worker tokens missing"
    )
    assert result.cost_usd == pytest.approx(0.08), (
        f"cost_usd={result.cost_usd} — worker cost missing"
    )


async def test_swarm_rolls_up_active_agent_costs_into_parent_result() -> None:
    """Same invariant for Swarm — its active-agent invocation also
    goes through SubagentInvocation and must roll up usage."""
    from loomflow.core.types import Usage

    # Swarm entry agent emits one turn with usage; no handoff.
    entry = Agent(
        "entry",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    text="ok",
                    usage=Usage(
                        input_tokens=70, output_tokens=30, cost_usd=0.04
                    ),
                ),
            ]
        ),
    )
    team = Team.swarm(
        agents={"entry": entry},
        entry_agent="entry",
        model="echo",  # swarm's own coordinator model — unused after
                       # the entry agent runs to completion
    )
    result = await team.run("hello")

    assert result.tokens_in >= 70, (
        f"tokens_in={result.tokens_in} — entry-agent tokens missing"
    )
    assert result.tokens_out >= 30, (
        f"tokens_out={result.tokens_out} — entry-agent tokens missing"
    )
    assert result.cost_usd >= 0.04, (
        f"cost_usd={result.cost_usd} — entry-agent cost missing"
    )


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
    from loomflow import ReAct

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
