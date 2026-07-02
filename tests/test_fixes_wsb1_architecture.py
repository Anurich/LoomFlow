"""Regression tests for the WSB1 architecture-layer fixes.

Pins:

1. Per-user budget forwarding — every architecture (not just ReAct)
   passes ``user_id`` into ``budget.allows_step`` / ``budget.consume``
   via the shared :func:`budget_gate` / :func:`consume_usage`
   helpers. Pre-fix, per-user caps were silently bypassed outside
   ReAct.
2. ReWOO tool execution is GATED — a destructive tool in a ReWOO
   plan routes through permissions → approval handler and leaves
   tool_call / tool_result audit entries, exactly like ReAct.
   Pre-fix, ReWOO called ``deps.tools.call`` directly (no gates,
   no audit).
3. ReAct re-invocation (stop-hook Ralph loop) does NOT re-seed the
   full context — the second ``run()`` on the same session appends
   only the new user prompt.
4. Per-tool-call timeout backstop (``Dependencies.tool_timeout_s``)
   turns a stuck tool into a ``ToolResult.error_``.
5. Unconditional tool-result truncation
   (``Dependencies.tool_result_max_chars``) hard-caps oversized
   tool output entering conversation history.
6. Blackboard budget WARN events surface (the pre-fix copy dropped
   the warn branch).
7. Reflexion attempts each get a fresh turn budget (attempt 2+ no
   longer starves at the base architecture's ``max_turns``).
8. TreeOfThoughts synthesizes a real final answer by default.
9. Supervisor's delegate marks interrupted worker output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anyio
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
from loomflow.architecture import (
    AgentSession,
    BlackboardArchitecture,
    Dependencies,
    MultiAgentDebate,
    Reflexion,
    ReWOO,
    Supervisor,
    TreeOfThoughts,
)
from loomflow.architecture.helpers import run_single_tool
from loomflow.core.types import BudgetStatus, Role, ToolCall, ToolResult
from loomflow.governance.budget import NoBudget
from loomflow.observability.tracing import NoTelemetry
from loomflow.runtime.inproc import InProcRuntime
from loomflow.security.audit import InMemoryAuditLog
from loomflow.security.hooks import HookRegistry
from loomflow.security.permissions import AllowAll

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _RecordingBudget:
    """Budget fake that records the user_id passed to each call."""

    allows_user_ids: list[str | None] = field(default_factory=list)
    consume_user_ids: list[str | None] = field(default_factory=list)

    async def allows_step(
        self, *, user_id: str | None = None
    ) -> BudgetStatus:
        self.allows_user_ids.append(user_id)
        return BudgetStatus.ok_()

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        user_id: str | None = None,
    ) -> None:
        self.consume_user_ids.append(user_id)


class _LegacyBudget:
    """Budget fake WITHOUT the user_id kwarg — the legacy protocol
    shape. The shared helpers must fall back gracefully."""

    def __init__(self) -> None:
        self.allows_calls = 0
        self.consume_calls = 0

    async def allows_step(self) -> BudgetStatus:
        self.allows_calls += 1
        return BudgetStatus.ok_()

    async def consume(
        self, *, tokens_in: int, tokens_out: int, cost_usd: float
    ) -> None:
        self.consume_calls += 1


class _AlwaysWarnBudget:
    async def allows_step(
        self, *, user_id: str | None = None
    ) -> BudgetStatus:
        return BudgetStatus.warn_("nearly out")

    async def consume(self, **kwargs: object) -> None:
        return None


class _EmptyToolHost:
    async def list_tools(self) -> list[object]:
        return []

    async def call(
        self, tool_name: str, args: dict, *, call_id: str | None = None
    ) -> ToolResult:
        return ToolResult.error_(call_id or "none", "no tools registered")


class _SlowToolHost:
    """Tool host whose call blocks far longer than any test timeout."""

    async def list_tools(self) -> list[object]:
        return []

    async def call(
        self, tool_name: str, args: dict, *, call_id: str | None = None
    ) -> ToolResult:
        await anyio.sleep(30)
        return ToolResult.success(call_id or "none", "too late")


def _make_deps(
    model: ScriptedModel,
    *,
    tools: object | None = None,
    tool_timeout_s: float | None = 120.0,
) -> Dependencies:
    return Dependencies(
        model=model,  # type: ignore[arg-type]
        memory=InMemoryMemory(),
        runtime=InProcRuntime(),
        tools=tools if tools is not None else _EmptyToolHost(),  # type: ignore[arg-type]
        budget=NoBudget(),
        permissions=AllowAll(),
        hooks=HookRegistry(),
        telemetry=NoTelemetry(),
        audit_log=None,
        max_turns=10,
        tool_timeout_s=tool_timeout_s,
    )


# ---------------------------------------------------------------------------
# 1. Per-user budget forwarding (ReWOO / ToT / Debate)
# ---------------------------------------------------------------------------


@tool
def echo(value: str) -> str:
    """Return the value unchanged."""
    return f"echoed:{value}"


async def test_rewoo_forwards_user_id_to_budget() -> None:
    budget = _RecordingBudget()
    plan_json = '[{"id": "E1", "tool": "echo", "args": {"value": "x"}}]'
    model = ScriptedModel(
        [ScriptedTurn(text=plan_json), ScriptedTurn(text="done")]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[echo],
        architecture=ReWOO(),
        budget=budget,  # type: ignore[arg-type]
    )
    await agent.run("go", user_id="alice")
    # Level gate ran at least once, planner + solver consumed.
    assert budget.allows_user_ids and all(
        u == "alice" for u in budget.allows_user_ids
    )
    assert len(budget.consume_user_ids) == 2  # planner + solver
    assert all(u == "alice" for u in budget.consume_user_ids)


async def test_tree_of_thoughts_forwards_user_id_to_budget() -> None:
    budget = _RecordingBudget()
    model = ScriptedModel(
        [
            ScriptedTurn(text="a thought"),
            ScriptedTurn(text="score: 0.5"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=1, max_depth=1, beam_width=1,
            synthesize_final=False,
        ),
        budget=budget,  # type: ignore[arg-type]
    )
    await agent.run("task", user_id="bob")
    assert budget.allows_user_ids and all(
        u == "bob" for u in budget.allows_user_ids
    )
    assert len(budget.consume_user_ids) == 2  # propose + eval
    assert all(u == "bob" for u in budget.consume_user_ids)


async def test_debate_forwards_user_id_to_budget() -> None:
    budget = _RecordingBudget()
    d1 = Agent(
        "d1",
        model=ScriptedModel(
            [
                ScriptedTurn(text="apples bananas cherries oranges"),
                ScriptedTurn(text="apples bananas cherries oranges"),
            ]
        ),
    )
    d2 = Agent(
        "d2",
        model=ScriptedModel(
            [
                ScriptedTurn(text="rockets planets stars galaxies"),
                ScriptedTurn(text="rockets planets stars galaxies"),
            ]
        ),
    )
    agent = Agent(
        "moderator",
        model="echo",
        architecture=MultiAgentDebate(debaters=[d1, d2], rounds=1),
        budget=budget,  # type: ignore[arg-type]
    )
    await agent.run("contested question", user_id="carol")
    # The round-1 gate must have seen carol's user_id.
    assert budget.allows_user_ids == ["carol"]


async def test_legacy_budget_without_user_id_kwarg_still_works() -> None:
    """Legacy budgets (no user_id kwarg) fall back via TypeError."""
    budget = _LegacyBudget()
    plan_json = '[{"id": "E1", "tool": "echo", "args": {"value": "x"}}]'
    model = ScriptedModel(
        [ScriptedTurn(text=plan_json), ScriptedTurn(text="done")]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[echo],
        architecture=ReWOO(),
        budget=budget,  # type: ignore[arg-type]
    )
    result = await agent.run("go", user_id="alice")
    assert "done" in result.output
    assert budget.allows_calls >= 1
    assert budget.consume_calls == 2


# ---------------------------------------------------------------------------
# 2. ReWOO destructive tools hit approval + audit gates
# ---------------------------------------------------------------------------


async def test_rewoo_destructive_tool_routes_through_approval_and_audit() -> None:
    executed: list[str] = []

    @tool(destructive=True)
    def wipe(target: str) -> str:
        """Destructively wipe a target."""
        executed.append(target)
        return f"wiped:{target}"

    handler_calls: list[ToolCall] = []

    async def approve(call: ToolCall, user_id: str | None = None) -> bool:
        handler_calls.append(call)
        return True

    audit = InMemoryAuditLog(secret="test-secret")
    plan_json = '[{"id": "E1", "tool": "wipe", "args": {"target": "db"}}]'
    model = ScriptedModel(
        [ScriptedTurn(text=plan_json), ScriptedTurn(text="wiped ok")]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[wipe],
        architecture=ReWOO(),
        permissions=StandardPermissions(),
        approval_handler=approve,
        audit_log=audit,
    )
    result = await agent.run("wipe the db")

    # Approval handler consulted exactly once, for the wipe call.
    assert len(handler_calls) == 1
    assert handler_calls[0].tool == "wipe"
    # Handler approved → the tool actually ran.
    assert executed == ["db"]
    assert "wiped ok" in result.output

    # The same audit trail ReAct leaves: tool_call + tool_result.
    tool_call_entries = await audit.query(action="tool_call")
    tool_result_entries = await audit.query(action="tool_result")
    assert any(
        e.payload.get("tool") == "wipe" for e in tool_call_entries
    )
    assert any(
        e.payload.get("tool") == "wipe" and e.payload.get("ok")
        for e in tool_result_entries
    )


async def test_rewoo_destructive_tool_denied_when_approval_declines() -> None:
    executed: list[str] = []

    @tool(destructive=True)
    def wipe(target: str) -> str:
        """Destructively wipe a target."""
        executed.append(target)
        return f"wiped:{target}"

    async def deny(call: ToolCall, user_id: str | None = None) -> bool:
        return False

    plan_json = '[{"id": "E1", "tool": "wipe", "args": {"target": "db"}}]'
    model = ScriptedModel(
        [ScriptedTurn(text=plan_json), ScriptedTurn(text="could not wipe")]
    )
    agent = Agent(
        "test",
        model=model,
        tools=[wipe],
        architecture=ReWOO(),
        permissions=StandardPermissions(),
        approval_handler=deny,
    )
    result = await agent.run("wipe the db")
    # The gate has teeth: the tool must NOT have executed.
    assert executed == []
    # The architecture kept going — the solver saw the denial.
    assert "could not wipe" in result.output


# ---------------------------------------------------------------------------
# 3. ReAct re-invocation does not re-seed the context
# ---------------------------------------------------------------------------


async def test_react_reinvocation_seeds_once_and_appends_user_prompt() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(text="first answer"),
            ScriptedTurn(text="second answer"),
        ]
    )
    deps = _make_deps(model)
    session = AgentSession(id="sess_test", instructions="be helpful")
    arch = ReAct()

    async for _ in arch.run(session, deps, "first prompt"):
        pass
    async for _ in arch.run(session, deps, "second prompt"):
        pass

    # Exactly ONE copy of the system prompt — no duplicated seed.
    system_msgs = [
        m for m in session.messages
        if m.role is Role.SYSTEM and m.content == "be helpful"
    ]
    assert len(system_msgs) == 1
    # Both prompts present as separate USER turns, in order.
    user_contents = [
        m.content for m in session.messages if m.role is Role.USER
    ]
    assert user_contents == ["first prompt", "second prompt"]
    assert session.output == "second answer"


# ---------------------------------------------------------------------------
# 4. Per-tool-call timeout backstop
# ---------------------------------------------------------------------------


async def test_tool_timeout_returns_error_result() -> None:
    deps = _make_deps(
        ScriptedModel([]), tools=_SlowToolHost(), tool_timeout_s=0.05
    )
    result = await run_single_tool(
        deps,
        ToolCall(id="c1", tool="slow", args={}),
        turn=1,
        slot=0,
    )
    assert not result.ok
    assert result.error is not None
    assert "timed out" in result.error


async def test_tool_timeout_none_disables_backstop() -> None:
    class _QuickHost:
        async def list_tools(self) -> list[object]:
            return []

        async def call(
            self, tool_name: str, args: dict, *, call_id: str | None = None
        ) -> ToolResult:
            await anyio.sleep(0.01)
            return ToolResult.success(call_id or "c", "fine")

    deps = _make_deps(
        ScriptedModel([]), tools=_QuickHost(), tool_timeout_s=None
    )
    result = await run_single_tool(
        deps,
        ToolCall(id="c1", tool="quick", args={}),
        turn=1,
        slot=0,
    )
    assert result.ok
    assert result.output == "fine"


# ---------------------------------------------------------------------------
# 5. Unconditional tool-result truncation
# ---------------------------------------------------------------------------


async def test_oversized_tool_result_is_truncated() -> None:
    @tool
    def firehose(n: int) -> str:
        """Return an n-char blob."""
        return "x" * n

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="firehose", args={"n": 60_000})
                ]
            ),
            ScriptedTurn(text="handled"),
        ]
    )
    agent = Agent("test", model=model, tools=[firehose])
    events = [e async for e in agent.stream("flood me")]
    truncated = [
        e for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "tool_result_truncated"
    ]
    assert len(truncated) == 1
    assert truncated[0].payload["truncated_chars"] == 10_000
    assert truncated[0].payload["kept_chars"] == 50_000


# ---------------------------------------------------------------------------
# 6. Blackboard surfaces budget WARN events (previously dropped)
# ---------------------------------------------------------------------------


async def test_blackboard_emits_budget_warning() -> None:
    writer = Agent(
        "writer",
        model=ScriptedModel([ScriptedTurn(text="a contribution")]),
    )
    agent = Agent(
        "host",
        model="echo",
        architecture=BlackboardArchitecture(
            agents={"writer": writer},
            coordinator=None,  # round-robin
            decider=None,
            max_rounds=1,
        ),
        budget=_AlwaysWarnBudget(),  # type: ignore[arg-type]
    )
    events = [e async for e in agent.stream("solve it")]
    kinds = [e.kind for e in events]
    assert "budget_warning" in kinds


# ---------------------------------------------------------------------------
# 7. Reflexion: fresh per-attempt turn budget
# ---------------------------------------------------------------------------


async def test_reflexion_attempt_two_gets_fresh_turn_budget() -> None:
    """With Agent(max_turns=3), attempt 1 consumes 3 turns (base +
    eval + reflect). Pre-fix, attempt 2's base ReAct saw
    ``session.turns >= max_turns`` immediately and starved. Post-fix
    it runs with a fresh budget and succeeds; RunResult.turns still
    reports the true cumulative total."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="v1"),          # attempt 1: base
            ScriptedTurn(text="score: 0.2"),  # attempt 1: eval
            ScriptedTurn(text="a lesson"),    # attempt 1: reflect
            ScriptedTurn(text="v2"),          # attempt 2: base
            ScriptedTurn(text="score: 0.9"),  # attempt 2: eval
        ]
    )
    agent = Agent(
        "test",
        model=model,
        max_turns=3,
        architecture=Reflexion(
            base=ReAct(), threshold=0.8, max_attempts=2
        ),
    )
    result = await agent.run("task")
    assert not result.interrupted
    assert result.output == "v2"
    # Cumulative across attempts: 3 (attempt 1) + 2 (attempt 2).
    assert result.turns == 5


# ---------------------------------------------------------------------------
# 8. TreeOfThoughts synthesizes a final answer by default
# ---------------------------------------------------------------------------


async def test_tot_synthesizes_final_answer_by_default() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(text="1. compute 6*7"),   # propose
            ScriptedTurn(text="score: 0.6"),        # eval
            ScriptedTurn(text="The answer is 42."),  # synthesis
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=1, max_depth=1, beam_width=1
        ),
    )
    result = await agent.run("what is 6*7?")
    # The output is the synthesized ANSWER, not the raw thought.
    assert result.output == "The answer is 42."
    assert result.turns == 3


async def test_tot_batched_proposer_issues_one_call_per_parent() -> None:
    """branch_factor=3 must NOT mean 3 identical proposer calls —
    one batched call returns all three thoughts."""
    model = ScriptedModel(
        [
            ScriptedTurn(text="1. A\n2. B\n3. C"),  # ONE propose call
            ScriptedTurn(text="score: 0.9"),
            ScriptedTurn(text="score: 0.5"),
            ScriptedTurn(text="score: 0.1"),
        ]
    )
    agent = Agent(
        "solver",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=3, max_depth=1, beam_width=1,
            synthesize_final=False,
        ),
    )
    events = [e async for e in agent.stream("task")]
    proposed = [
        e for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "tot.proposed"
    ]
    assert [e.payload["content"] for e in proposed] == ["A", "B", "C"]
    # Script had exactly 4 turns — if the proposer had issued 3
    # calls, the evaluators would have consumed empty turns and no
    # candidate could score 0.9.
    completed = next(
        e for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "tot.completed"
    )
    assert completed.payload["winner_score"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 9. Supervisor delegate marks interrupted worker output
# ---------------------------------------------------------------------------


async def test_supervisor_delegate_marks_interrupted_worker() -> None:
    worker = Agent(
        "worker",
        model=ScriptedModel([ScriptedTurn(text="partial work")]),
        max_turns=0,  # interrupts immediately: max_turns_exceeded
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
            ScriptedTurn(text="synthesized"),
        ]
    )
    agent = Agent(
        "boss",
        model=model,
        architecture=Supervisor(workers={"w": worker}),
    )
    events = [e async for e in agent.stream("do the thing")]
    delegate_results = [
        e for e in events
        if e.kind == "tool_result"
        and str(e.payload["result"].get("output", "")).startswith(
            "[interrupted:"
        )
    ]
    assert len(delegate_results) == 1
    assert "max_turns_exceeded" in str(
        delegate_results[0].payload["result"]["output"]
    )
