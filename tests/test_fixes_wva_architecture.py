"""Regression tests for the WVA architecture-layer fixes
(REVIEW_2026-07 Part A).

Pins:

1. Cross-tenant worker reuse — every persistent-worker spawn site
   (Supervisor, Swarm, Router, Blackboard, Debate) goes through the
   shared :func:`acquire_worker_session` helper: cross-user mismatch
   is rejected, the handle lock is held for the worker's run, and
   the run's REAL user_id is pinned (Debate previously touched with
   a hard-coded ``None``).
2. Fail-closed destructive backstop — when ``list_tools()`` raises
   while stamping the destructive flag, the call is treated as
   destructive (routed through the ask gate) instead of sailing
   through as destructive=False.
3. Reflexion gates each attempt on the budget (evaluator/reflector
   calls no longer burn tokens after the budget is exhausted).
4. Swarm gates each handoff iteration on the budget.
6. ExtendedToolHost: extras SHADOW same-named base defs (no
   duplicate ToolDefs → no API 400), and extra fns that return a
   ToolResult are passed through instead of double-nested.
7. Worker spend is charged against the PARENT budget after each
   sub-agent invocation (budget-only — cumulative_usage is rolled
   up once, never twice; workers sharing the parent budget instance
   are skipped).
8. ReAct only records SUCCESSFUL tool calls for plan_write's
   strong-mode verification.
9. ``parse_score`` no longer treats a bare 0/1 inside prose
   ("1 issue remains") as a score.
10. Router: a classification with NO confidence line counts as
    below-threshold when ``require_confidence_above`` is set.
11. Reflexion does not wipe ``session.messages`` on attempt 1, so
    stop-hook re-entry / follow-up runs keep their conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from loomflow import (
    Agent,
    InMemoryMemory,
    ReAct,
    ScriptedModel,
    ScriptedTurn,
    StandardPermissions,
    tool,
)
from loomflow.agent.worker_registry import (
    CrossUserWorkerError,
    acquire_worker_session,
    build_worker_registry,
)
from loomflow.architecture import (
    AgentSession,
    BlackboardArchitecture,
    Dependencies,
    MultiAgentDebate,
    Reflexion,
    Router,
    RouterRoute,
    Supervisor,
    Swarm,
)
from loomflow.architecture.helpers import (
    consume_worker_usage,
    parse_score,
    run_single_tool,
)
from loomflow.architecture.tool_host_wrappers import ExtendedToolHost
from loomflow.core.types import (
    BudgetStatus,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from loomflow.governance.budget import NoBudget
from loomflow.observability.tracing import NoTelemetry
from loomflow.runtime.inproc import InProcRuntime
from loomflow.security.hooks import HookRegistry
from loomflow.security.permissions import AllowAll
from loomflow.tools.registry import InProcessToolHost, Tool

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _RecordingBudget:
    """Always-OK budget that records every consume call."""

    allows_calls: int = 0
    consume_records: list[tuple[int, int, float, str | None]] = field(
        default_factory=list
    )

    async def allows_step(
        self, *, user_id: str | None = None
    ) -> BudgetStatus:
        self.allows_calls += 1
        return BudgetStatus.ok_()

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        user_id: str | None = None,
    ) -> None:
        self.consume_records.append(
            (tokens_in, tokens_out, cost_usd, user_id)
        )


class _BlockAfterBudget:
    """Allows the first ``n`` gate checks, blocks from n+1 on."""

    def __init__(self, n: int) -> None:
        self._n = n
        self.allows_calls = 0

    async def allows_step(
        self, *, user_id: str | None = None
    ) -> BudgetStatus:
        self.allows_calls += 1
        if self.allows_calls > self._n:
            return BudgetStatus.blocked_("cap reached")
        return BudgetStatus.ok_()

    async def consume(self, **kwargs: object) -> None:
        return None


class _EmptyToolHost:
    async def list_tools(self) -> list[object]:
        return []

    async def call(
        self, tool_name: str, args: dict, *, call_id: str = ""
    ) -> ToolResult:
        return ToolResult.error_(call_id or "none", "no tools")


class _FailingListToolHost:
    """``list_tools`` raises; records whether ``call`` was reached."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self) -> list[object]:
        raise RuntimeError("tool host unavailable")

    async def call(
        self, tool_name: str, args: dict, *, call_id: str = ""
    ) -> ToolResult:
        self.calls.append(tool_name)
        return ToolResult.success(call_id or "none", "ran")


def _make_deps(model: ScriptedModel, **overrides: Any) -> Dependencies:
    kwargs: dict[str, Any] = dict(
        model=model,
        memory=InMemoryMemory(),
        runtime=InProcRuntime(),
        tools=_EmptyToolHost(),
        budget=NoBudget(),
        permissions=AllowAll(),
        hooks=HookRegistry(),
        telemetry=NoTelemetry(),
        audit_log=None,
        max_turns=10,
    )
    kwargs.update(overrides)
    return Dependencies(**kwargs)


# ---------------------------------------------------------------------------
# 1. Cross-tenant worker reuse
# ---------------------------------------------------------------------------


async def test_acquire_worker_session_pins_locks_and_rejects() -> None:
    registry, role_map = build_worker_registry(
        {"coder": Agent("coder", model="echo")}
    )
    handle = registry[role_map["coder"]]
    assert handle.user_id is None

    # First touch pins the caller's user_id (under the lock).
    async with acquire_worker_session(handle, "alice"):
        assert handle.user_id == "alice"
        assert handle.lock.locked()
    assert not handle.lock.locked()
    assert handle.last_used_at is not None

    # Same user again is fine; a DIFFERENT user is rejected before
    # the lock is taken.
    async with acquire_worker_session(handle, "alice"):
        pass
    with pytest.raises(CrossUserWorkerError, match="Cross-tenant"):
        async with acquire_worker_session(handle, "bob"):
            pass
    assert not handle.lock.locked()
    # Anonymous callers (None) are not a mismatch.
    async with acquire_worker_session(handle, None):
        pass


async def test_swarm_rejects_cross_user_worker() -> None:
    peer_model = ScriptedModel([ScriptedTurn(text="peer answer")])
    peer = Agent("a", model=peer_model)
    registry, role_map = build_worker_registry({"a": peer})
    registry[role_map["a"]].touch(user_id="alice")

    agent = Agent(
        "host",
        model="echo",
        architecture=Swarm(
            agents={"a": peer},
            entry_agent="a",
            worker_registry=registry,
            role_to_worker_id=role_map,
        ),
    )
    events = [e async for e in agent.stream("go", user_id="bob")]
    rejected = [
        e
        for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "swarm.cross_user_rejected"
    ]
    assert len(rejected) == 1
    assert "Cross-tenant" in rejected[0].payload["error"]
    # The peer never ran — its script is untouched.
    assert peer_model.remaining == 1


async def test_router_rejects_cross_user_specialist() -> None:
    spec_model = ScriptedModel([ScriptedTurn(text="specialist answer")])
    spec = Agent("s", model=spec_model)
    registry, role_map = build_worker_registry({"s": spec})
    registry[role_map["s"]].touch(user_id="alice")

    agent = Agent(
        "router-host",
        model=ScriptedModel(
            [ScriptedTurn(text="route: s\nconfidence: 0.9")]
        ),
        architecture=Router(
            routes=[RouterRoute(name="s", agent=spec)],
            worker_registry=registry,
            role_to_worker_id=role_map,
        ),
    )
    result = await agent.run("go", user_id="bob")
    assert result.interrupted
    assert "cross_user_worker" in (result.interruption_reason or "")
    assert "Could not route" in result.output
    assert spec_model.remaining == 1


async def test_blackboard_skips_cross_user_contributor() -> None:
    writer_model = ScriptedModel([ScriptedTurn(text="a contribution")])
    writer = Agent("writer", model=writer_model)
    registry, role_map = build_worker_registry({"writer": writer})
    registry[role_map["writer"]].touch(user_id="alice")

    agent = Agent(
        "host",
        model="echo",
        architecture=BlackboardArchitecture(
            agents={"writer": writer},
            coordinator=None,  # round-robin picks writer
            decider=None,
            max_rounds=1,
            worker_registry=registry,
            role_to_worker_id=role_map,
        ),
    )
    events = [e async for e in agent.stream("solve", user_id="bob")]
    rejected = [
        e
        for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "blackboard.cross_user_rejected"
    ]
    assert len(rejected) == 1
    assert writer_model.remaining == 1  # contributor never ran


async def test_debate_pins_real_user_id_on_handles() -> None:
    """Pre-fix, debaters touched with ``user_id=None`` — handles were
    never pinned, so the cross-tenant guard could never fire."""
    d0 = Agent(
        "d0", model=ScriptedModel([ScriptedTurn(text="same answer")])
    )
    d1 = Agent(
        "d1", model=ScriptedModel([ScriptedTurn(text="same answer")])
    )
    registry, role_map = build_worker_registry(
        {"debater_0": d0, "debater_1": d1}
    )
    agent = Agent(
        "moderator",
        model="echo",
        architecture=MultiAgentDebate(
            debaters=[d0, d1],
            rounds=1,
            worker_registry=registry,
            role_to_worker_id=role_map,
        ),
    )
    await agent.run("question", user_id="carol")
    assert registry[role_map["debater_0"]].user_id == "carol"
    assert registry[role_map["debater_1"]].user_id == "carol"


async def test_debate_rejects_cross_user_debater_gracefully() -> None:
    d0_model = ScriptedModel([ScriptedTurn(text="pinned answer")])
    d0 = Agent("d0", model=d0_model)
    d1 = Agent(
        "d1", model=ScriptedModel([ScriptedTurn(text="free answer")])
    )
    registry, role_map = build_worker_registry(
        {"debater_0": d0, "debater_1": d1}
    )
    registry[role_map["debater_0"]].touch(user_id="alice")

    agent = Agent(
        "moderator",
        model="echo",
        architecture=MultiAgentDebate(
            debaters=[d0, d1],
            rounds=1,
            convergence_check=False,
            worker_registry=registry,
            role_to_worker_id=role_map,
        ),
    )
    events = [e async for e in agent.stream("question", user_id="bob")]
    rejected = [
        e
        for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "debate.cross_user_rejected"
    ]
    assert rejected  # d0 rejected each round, run didn't crash
    assert d0_model.remaining == 1


async def test_supervisor_cross_user_delegate_returns_error_string() -> None:
    """Supervisor's pre-existing behaviour — the model sees an error
    string — is preserved through the shared helper."""
    worker_model = ScriptedModel([ScriptedTurn(text="worker out")])
    worker = Agent("w", model=worker_model)
    registry, role_map = build_worker_registry({"w": worker})
    registry[role_map["w"]].touch(user_id="alice")

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={"worker": "w", "instructions": "go"},
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        "boss",
        model=model,
        architecture=Supervisor(
            workers={"w": worker},
            worker_registry=registry,
            role_to_worker_id=role_map,
        ),
    )
    events = [e async for e in agent.stream("task", user_id="bob")]
    delegate_results = [
        str(e.payload["result"].get("output", ""))
        for e in events
        if e.kind == "tool_result"
    ]
    assert any(
        out.startswith("Error:")
        and "Cross-tenant delegation is rejected" in out
        for out in delegate_results
    )
    assert worker_model.remaining == 1  # worker never ran


# ---------------------------------------------------------------------------
# 2. Fail-closed destructive backstop
# ---------------------------------------------------------------------------


async def test_destructive_backstop_fails_closed_on_list_tools_error() -> None:
    """``list_tools`` failure → the call is stamped destructive and
    routed through the ask gate; with no approver it is DENIED and
    never reaches the tool host."""
    host = _FailingListToolHost()
    deps = _make_deps(
        ScriptedModel([]),
        tools=host,
        permissions=StandardPermissions(),
        fast_permissions=False,
    )
    result = await run_single_tool(
        deps,
        ToolCall(id="c1", tool="wipe_db", args={}),
        turn=1,
        slot=0,
    )
    assert result.denied
    assert host.calls == []


async def test_destructive_backstop_consults_approver_on_error() -> None:
    """The fail-closed stamp routes through the approval handler —
    an approving handler still lets the call proceed."""
    host = _FailingListToolHost()
    consulted: list[ToolCall] = []

    async def approve(call: ToolCall, user_id: str | None = None) -> bool:
        consulted.append(call)
        return True

    deps = _make_deps(
        ScriptedModel([]),
        tools=host,
        permissions=StandardPermissions(),
        fast_permissions=False,
        approval_handler=approve,
    )
    result = await run_single_tool(
        deps,
        ToolCall(id="c1", tool="wipe_db", args={}),
        turn=1,
        slot=0,
    )
    assert len(consulted) == 1
    assert consulted[0].destructive  # the fail-closed stamp
    assert result.ok
    assert host.calls == ["wipe_db"]


# ---------------------------------------------------------------------------
# 3. Reflexion budget gate
# ---------------------------------------------------------------------------


async def test_reflexion_gates_attempts_on_budget() -> None:
    model = ScriptedModel([ScriptedTurn(text="never runs")])
    agent = Agent(
        "test",
        model=model,
        architecture=Reflexion(base=ReAct(), max_attempts=2),
        budget=_BlockAfterBudget(0),  # type: ignore[arg-type]
    )
    events = [e async for e in agent.stream("task")]
    assert any(e.kind == "budget_exceeded" for e in events)
    completed = next(e for e in events if e.kind == "completed")
    result = completed.payload["result"]
    assert result["interrupted"]
    assert str(result["interruption_reason"]).startswith("budget:")
    # Neither the base nor the evaluator/reflector ever called the
    # model — pre-fix, only the base's own loop was gated.
    assert model.remaining == 1


# ---------------------------------------------------------------------------
# 4. Swarm budget gate
# ---------------------------------------------------------------------------


async def test_swarm_gates_each_handoff_iteration_on_budget() -> None:
    model_a = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="h1",
                        tool="handoff",
                        args={"target": "b", "message": "yours"},
                    )
                ]
            ),
            ScriptedTurn(text="a is done"),
        ]
    )
    model_b = ScriptedModel([ScriptedTurn(text="b never runs")])
    budget = _BlockAfterBudget(1)
    agent = Agent(
        "host",
        model="echo",
        architecture=Swarm(
            agents={
                "a": Agent("a", model=model_a),
                "b": Agent("b", model=model_b),
            },
            entry_agent="a",
        ),
        budget=budget,  # type: ignore[arg-type]
    )
    result = await agent.run("go")
    # Iteration 1 (peer a) was allowed; iteration 2 (peer b) blocked.
    assert budget.allows_calls == 2
    assert result.interrupted
    assert (result.interruption_reason or "").startswith("budget:")
    assert result.output == "a is done"
    assert model_b.remaining == 1  # peer b never ran


# ---------------------------------------------------------------------------
# 6. ExtendedToolHost — shadowing + ToolResult pass-through
# ---------------------------------------------------------------------------


async def test_extended_tool_host_extras_shadow_base_defs() -> None:
    async def base_fn() -> str:
        return "base"

    async def extra_fn() -> str:
        return "extra"

    base = InProcessToolHost(
        [Tool(name="dup", description="base version", fn=base_fn)]
    )
    host = ExtendedToolHost(
        base, [Tool(name="dup", description="extra version", fn=extra_fn)]
    )
    defs = await host.list_tools()
    dup_defs = [d for d in defs if d.name == "dup"]
    assert len(dup_defs) == 1  # no duplicate ToolDefs (API 400)
    assert dup_defs[0].description == "extra version"  # extras win
    # Dispatch matches listing: the extra executes.
    result = await host.call("dup", {}, call_id="c1")
    assert result.output == "extra"


async def test_extended_tool_host_passes_through_tool_result() -> None:
    async def returns_result() -> ToolResult:
        return ToolResult.error_("inner", "already a result")

    host = ExtendedToolHost(
        InProcessToolHost([]),
        [Tool(name="raw", description="", fn=returns_result)],
    )
    result = await host.call("raw", {}, call_id="c9")
    # NOT double-nested: the returned ToolResult IS the result.
    assert not isinstance(result.output, ToolResult)
    assert not result.ok
    assert result.error == "already a result"
    assert result.call_id == "c9"  # re-stamped to the caller's id


# ---------------------------------------------------------------------------
# 7. Worker spend charged against the parent budget
# ---------------------------------------------------------------------------


async def test_consume_worker_usage_unit_semantics() -> None:
    budget = _RecordingBudget()
    deps = _make_deps(
        ScriptedModel([]), budget=budget, fast_budget=False
    )

    class _Worker:
        def __init__(self, b: object) -> None:
            self.budget = b

    usage = Usage(input_tokens=5, output_tokens=2, cost_usd=0.1)
    # Distinct budget instance → charged.
    await consume_worker_usage(deps, _Worker(NoBudget()), usage)
    assert budget.consume_records == [(5, 2, 0.1, None)]
    # SAME budget instance → skipped (worker already consumed).
    await consume_worker_usage(deps, _Worker(budget), usage)
    assert len(budget.consume_records) == 1
    # All-zero usage → skipped.
    await consume_worker_usage(deps, _Worker(NoBudget()), Usage())
    assert len(budget.consume_records) == 1


async def test_router_charges_specialist_spend_to_parent_budget() -> None:
    budget = _RecordingBudget()
    specialist = Agent(
        "s",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    text="answer",
                    usage=Usage(
                        input_tokens=7, output_tokens=3, cost_usd=0.5
                    ),
                )
            ]
        ),
    )
    agent = Agent(
        "router-host",
        model=ScriptedModel(
            [ScriptedTurn(text="route: s\nconfidence: 0.9")]
        ),
        architecture=Router(routes=[RouterRoute(name="s", agent=specialist)]),
        budget=budget,  # type: ignore[arg-type]
    )
    result = await agent.run("go", user_id="alice")
    # The specialist's spend hit the PARENT budget...
    assert (7, 3, 0.5, "alice") in budget.consume_records
    # ...and cumulative usage is NOT double-counted (rollup once).
    assert result.tokens_in == 7
    assert result.tokens_out == 3


async def test_supervisor_charges_worker_spend_to_parent_budget() -> None:
    budget = _RecordingBudget()
    worker = Agent(
        "w",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    text="worker out",
                    usage=Usage(
                        input_tokens=11, output_tokens=4, cost_usd=0.2
                    ),
                )
            ]
        ),
    )
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={"worker": "w", "instructions": "go"},
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        "boss",
        model=model,
        architecture=Supervisor(workers={"w": worker}),
        budget=budget,  # type: ignore[arg-type]
    )
    result = await agent.run("task", user_id="alice")
    assert (11, 4, 0.2, "alice") in budget.consume_records
    # Rolled up exactly once into the parent's cumulative usage.
    assert result.tokens_in == 11


# ---------------------------------------------------------------------------
# 8. record_tool_call gated on final.ok
# ---------------------------------------------------------------------------


async def test_failed_tool_calls_are_not_recorded_for_plan_verification() -> None:
    from loomflow.core.context import _ambient_living_plan_var
    from loomflow.tools.plan import _LivingPlanState

    @tool
    def works(value: str) -> str:
        """Succeeds."""
        return f"ok:{value}"

    @tool
    def breaks(value: str) -> str:
        """Raises."""
        raise RuntimeError("boom")

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="ok1", tool="works", args={"value": "x"}),
                    ToolCall(id="bad1", tool="breaks", args={"value": "x"}),
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    state = _LivingPlanState()
    token = _ambient_living_plan_var.set(state)
    try:
        agent = Agent("test", model=model, tools=[works, breaks])
        await agent.run("go")
    finally:
        _ambient_living_plan_var.reset(token)
    # Only the SUCCESSFUL call is verification evidence.
    assert state.observed_tool_call_ids == {"ok1"}


# ---------------------------------------------------------------------------
# 9. parse_score prose fallback
# ---------------------------------------------------------------------------


def test_parse_score_ignores_bare_integers_in_prose() -> None:
    # Pre-fix: "1" matched → 1.0 (a passing score for a critique
    # that reports an unresolved issue!). "0 errors" matched → 0.0.
    assert parse_score("1 issue remains in the code") == 0.0
    assert parse_score("Found 0 problems, but incomplete") == 0.0


def test_parse_score_accepts_whole_line_bare_numbers() -> None:
    assert parse_score("0.7") == 0.7
    assert parse_score("1") == 1.0
    assert parse_score("Reasoning above.\n0.4") == 0.4


def test_parse_score_keeps_decimal_in_prose_and_score_prefix() -> None:
    assert parse_score("the answer scored 0.6 overall") == 0.6
    assert parse_score("score: 0.85\njustification") == 0.85
    assert parse_score("no number anywhere here") == 0.0


# ---------------------------------------------------------------------------
# 10. Router missing-confidence semantics
# ---------------------------------------------------------------------------


async def test_missing_confidence_falls_back_when_threshold_set() -> None:
    billing = Agent(
        "billing", model=ScriptedModel([ScriptedTurn(text="billing ran")])
    )
    general = Agent(
        "general", model=ScriptedModel([ScriptedTurn(text="general ran")])
    )
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="route: billing")]),
        architecture=Router(
            routes=[
                RouterRoute(name="billing", agent=billing),
                RouterRoute(name="general", agent=general),
            ],
            fallback_route="general",
            require_confidence_above=0.7,
        ),
    )
    result = await agent.run("query")
    # Pre-fix: missing confidence defaulted to 1.0 → billing ran,
    # defeating require_confidence_above entirely.
    assert result.output == "general ran"


async def test_missing_confidence_dispatches_when_no_threshold() -> None:
    billing = Agent(
        "billing", model=ScriptedModel([ScriptedTurn(text="billing ran")])
    )
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="route: billing")]),
        architecture=Router(
            routes=[RouterRoute(name="billing", agent=billing)],
        ),
    )
    result = await agent.run("query")
    assert result.output == "billing ran"


# ---------------------------------------------------------------------------
# 11. Reflexion keeps conversation on re-entry
# ---------------------------------------------------------------------------


async def test_reflexion_reinvocation_preserves_prior_messages() -> None:
    """Mirrors the ReAct re-invocation pin (wsb1 #3): a second
    ``run()`` on the same session must append, not wipe — pre-fix,
    Reflexion's attempt-1 ``session.messages = []`` destroyed the
    conversation a stop-hook re-entry depends on."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="first answer"),
            ScriptedTurn(text="score: 0.9"),
            ScriptedTurn(text="second answer"),
            ScriptedTurn(text="score: 0.9"),
        ]
    )
    deps = _make_deps(model)
    session = AgentSession(id="sess_reflex", instructions="be helpful")
    arch = Reflexion(base=ReAct(), max_attempts=2, threshold=0.8)

    async for _ in arch.run(session, deps, "first prompt"):
        pass
    async for _ in arch.run(session, deps, "second prompt"):
        pass

    user_contents = [
        m.content for m in session.messages if m.role is Role.USER
    ]
    assert user_contents == ["first prompt", "second prompt"]
    system_msgs = [
        m
        for m in session.messages
        if m.role is Role.SYSTEM and m.content == "be helpful"
    ]
    assert len(system_msgs) == 1
    assert session.output == "second answer"
