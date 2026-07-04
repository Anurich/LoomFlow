"""G15 — dynamic model-driven agent spawning (``spawn_worker``).

Covers:

* Disabled by default: no ``spawn_worker`` in the tool defs the
  model sees; ``allow_spawn=True`` injects it.
* End-to-end: coordinator spawns a specialist mid-run, delegates to
  it, and forwards its output verbatim.
* Roster propagation: after a spawn, the delegate tool's enum +
  description (re-registered on the ExtendedToolHost, re-fetched by
  ReAct each turn) include the spawned role; the spawn confirmation
  and delegate error strings carry the roster.
* ``max_spawned`` cap → error string to the model, not an exception.
* Role validation (Python-identifier rule, duplicate roles, empty
  instructions) → error strings.
* Ephemeral lifecycle: spawned workers die with the run — the
  agent-level registry is clean after the run and a second run
  cannot delegate to the previously spawned role.
* Tenant isolation: the spawned handle pins the spawning run's
  ``user_id``; a cross-user delegate to a spawned worker is
  rejected (mirrors the ``acquire_worker_session`` tests).
* ``Team.supervisor(allow_spawn=..., max_spawned=...,
  spawn_template=...)`` passthrough.

All tests are zero-dep (ScriptedModel / EchoModel only).
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, EchoModel, ScriptedModel, ScriptedTurn
from loomflow.architecture import Supervisor
from loomflow.architecture.base import AgentSession
from loomflow.architecture.supervisor import (
    _make_delegate_tool,
    _make_spawn_worker_tool,
)
from loomflow.core.context import RunContext, set_run_context
from loomflow.core.types import ToolCall
from loomflow.memory.inmemory import InMemoryMemory
from loomflow.team import Team

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _worker(reply: str, instructions: str = "I am a specialist worker.") -> Agent:
    return Agent(instructions, model=ScriptedModel([ScriptedTurn(text=reply)]))


def _spawn_call(call_id: str, role: str, instructions: str = "You are new.") -> ToolCall:
    return ToolCall(
        id=call_id,
        tool="spawn_worker",
        args={"role": role, "instructions": instructions},
    )


def _delegate_call(call_id: str, worker: str, instructions: str = "do it") -> ToolCall:
    return ToolCall(
        id=call_id,
        tool="delegate",
        args={"worker": worker, "instructions": instructions},
    )


class _CaptureToolsModel:
    """Scripted model that records the tool defs offered each turn."""

    name = "capture-tools"

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        self._inner = ScriptedModel(turns)
        self.tool_lists: list[list[Any]] = []

    async def complete(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        self.tool_lists.append(list(tools or []))
        return await self._inner.complete(messages, tools=tools, **kwargs)

    async def stream(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        self.tool_lists.append(list(tools or []))
        async for chunk in self._inner.stream(messages, tools=tools, **kwargs):
            yield chunk


def _make_spawn_tool_direct(
    *,
    run_workers: dict[str, Agent] | None = None,
    spawn_overlay: dict[str, Any] | None = None,
    spawned_ids: list[str] | None = None,
    max_spawned: int = 5,
    template: Agent | None = None,
    shared_registry: dict[str, Any] | None = None,
) -> Any:
    """Unit-level spawn tool with all collaborators defaulted."""
    return _make_spawn_worker_tool(
        run_workers if run_workers is not None else {},
        spawn_overlay if spawn_overlay is not None else {},
        session=AgentSession(id="parent", instructions=""),
        template=template,
        fallback_model=EchoModel(),
        max_spawned=max_spawned,
        shared_registry=shared_registry,
        spawned_ids=spawned_ids if spawned_ids is not None else [],
        rebuild_worker_tools=lambda: None,
        event_sink=None,
    )


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------


async def test_spawn_worker_tool_absent_by_default() -> None:
    """Without ``allow_spawn=True`` the model never sees
    ``spawn_worker`` — zero behavior change for existing teams."""
    model = _CaptureToolsModel([ScriptedTurn(text="direct answer")])
    agent = Agent(
        "manager",
        model=model,  # type: ignore[arg-type]
        architecture=Supervisor(workers={"seed": _worker("ok")}),
    )
    await agent.run("hello")
    names = {d.name for d in model.tool_lists[0]}
    assert "spawn_worker" not in names
    assert "delegate" in names


async def test_spawn_worker_tool_present_when_enabled() -> None:
    model = _CaptureToolsModel([ScriptedTurn(text="direct answer")])
    agent = Agent(
        "manager",
        model=model,  # type: ignore[arg-type]
        architecture=Supervisor(
            workers={"seed": _worker("ok")}, allow_spawn=True
        ),
    )
    await agent.run("hello")
    names = {d.name for d in model.tool_lists[0]}
    assert "spawn_worker" in names


async def test_spawn_instructions_section_only_when_enabled() -> None:
    """The system-prompt spawn section appears only with
    ``allow_spawn=True``."""
    captured: list[str] = []

    class _CaptureSystemModel:
        name = "capture"

        async def stream(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            from loomflow.core.types import ModelChunk, Usage

            for m in messages:
                if m.role == "system":
                    captured.append(m.content)
            yield ModelChunk(kind="text", text="ok")
            yield ModelChunk(kind="finish", usage=Usage())

    agent = Agent(
        "manager",
        model=_CaptureSystemModel(),  # type: ignore[arg-type]
        architecture=Supervisor(
            workers={"seed": _worker("ok")},
            allow_spawn=True,
            max_spawned=3,
        ),
    )
    await agent.run("x")
    joined = "\n".join(captured)
    assert "spawn_worker(role, instructions)" in joined
    assert "At most 3 workers" in joined

    captured.clear()
    agent2 = Agent(
        "manager",
        model=_CaptureSystemModel(),  # type: ignore[arg-type]
        architecture=Supervisor(workers={"seed": _worker("ok")}),
    )
    await agent2.run("x")
    assert "spawn_worker" not in "\n".join(captured)


# ---------------------------------------------------------------------------
# End-to-end: spawn → delegate → forward
# ---------------------------------------------------------------------------


async def test_spawn_then_delegate_end_to_end() -> None:
    """Coordinator spawns a specialist mid-run, delegates to it, and
    forwards its output verbatim as the final answer."""
    template = Agent(
        "template",
        model=ScriptedModel([ScriptedTurn(text="SPAWNED_ANSWER")]),
    )
    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    _spawn_call("c1", "summarizer", "You summarize text.")
                ]
            ),
            ScriptedTurn(
                tool_calls=[_delegate_call("c2", "summarizer", "summarize X")]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c3",
                        tool="forward_message",
                        args={"worker": "summarizer"},
                    )
                ]
            ),
            ScriptedTurn(text="[done]"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(
            workers={"seed": _worker("seed output")},
            allow_spawn=True,
            spawn_template=template,
        ),
    )
    result = await agent.run("go")
    assert result.output == "SPAWNED_ANSWER"


async def test_spawn_emits_worker_spawned_event() -> None:
    template = Agent(
        "template", model=ScriptedModel([ScriptedTurn(text="out")])
    )
    parent_model = ScriptedModel(
        [
            ScriptedTurn(tool_calls=[_spawn_call("c1", "helper")]),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(
            workers={"seed": _worker("s")},
            allow_spawn=True,
            spawn_template=template,
        ),
    )
    events = [e async for e in agent.stream("go")]
    arch = [
        e.payload
        for e in events
        if e.kind == "architecture_event"
        and e.payload.get("name") == "supervisor.worker_spawned"
    ]
    assert len(arch) == 1
    assert arch[0]["role"] == "helper"
    assert arch[0]["worker_id"].startswith("worker_helper_")


# ---------------------------------------------------------------------------
# Roster propagation into delegate enum / description
# ---------------------------------------------------------------------------


async def test_delegate_defs_reflect_spawned_worker_next_turn() -> None:
    """ReAct re-fetches ``list_tools()`` each turn and the spawn tool
    re-registers delegate/forward — so the turn AFTER a spawn carries
    the new role in the delegate enum and description."""
    template = Agent(
        "template", model=ScriptedModel([ScriptedTurn(text="out")])
    )
    model = _CaptureToolsModel(
        [
            ScriptedTurn(
                tool_calls=[_spawn_call("c1", "helper", "You help.")]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        "manager",
        model=model,  # type: ignore[arg-type]
        architecture=Supervisor(
            workers={"seed": _worker("s")},
            allow_spawn=True,
            spawn_template=template,
        ),
    )
    await agent.run("go")
    assert len(model.tool_lists) == 2

    def _delegate_def(defs: list[Any]) -> Any:
        return next(d for d in defs if d.name == "delegate")

    before = _delegate_def(model.tool_lists[0])
    after = _delegate_def(model.tool_lists[1])
    assert before.input_schema["properties"]["worker"]["enum"] == ["seed"]
    assert set(after.input_schema["properties"]["worker"]["enum"]) == {
        "seed",
        "helper",
    }
    assert "helper" in after.description
    # forward_message enum updates too.
    fwd = next(d for d in model.tool_lists[1] if d.name == "forward_message")
    assert "helper" in fwd.input_schema["properties"]["worker"]["enum"]


async def test_spawn_confirmation_contains_roster_and_worker_id() -> None:
    run_workers: dict[str, Agent] = {"seed": _worker("s")}
    t = _make_spawn_tool_direct(run_workers=run_workers)
    out = await t.fn(role="analyst", instructions="You analyse.")
    assert "Spawned worker 'analyst'" in out
    assert "worker_id: worker_analyst_" in out
    assert "analyst" in out and "seed" in out  # roster echoed
    assert "delegate(worker='analyst'" in out


async def test_delegate_error_lists_spawned_roster() -> None:
    """The delegate unknown-worker error names the merged roster,
    including workers spawned this run."""
    run_workers: dict[str, Agent] = {"seed": _worker("s")}
    overlay: dict[str, Any] = {}
    spawn = _make_spawn_tool_direct(
        run_workers=run_workers, spawn_overlay=overlay
    )
    await spawn.fn(role="helper", instructions="You help.")

    delegate = _make_delegate_tool(
        run_workers,
        AgentSession(id="parent", instructions=""),
        tool_name="delegate",
        memory=InMemoryMemory(),
        spawn_overlay=overlay,
    )
    out = await delegate.fn(worker="ghost", instructions="x")
    assert "unknown worker 'ghost'" in out
    assert "helper" in out and "seed" in out


# ---------------------------------------------------------------------------
# max_spawned cap + validation → error strings, not exceptions
# ---------------------------------------------------------------------------


async def test_max_spawned_cap_returns_error_string() -> None:
    t = _make_spawn_tool_direct(max_spawned=1)
    ok = await t.fn(role="first", instructions="You are first.")
    assert ok.startswith("Spawned worker")
    blocked = await t.fn(role="second", instructions="You are second.")
    assert blocked.startswith("Error: spawn limit reached")
    assert "1" in blocked


async def test_max_spawned_zero_blocks_all_spawns() -> None:
    t = _make_spawn_tool_direct(max_spawned=0)
    out = await t.fn(role="any", instructions="x")
    assert out.startswith("Error: spawn limit reached")


async def test_spawn_rejects_non_identifier_role() -> None:
    t = _make_spawn_tool_direct()
    for bad in ("not a role", "123", "hyphen-name", ""):
        out = await t.fn(role=bad, instructions="x")
        assert out.startswith("Error: invalid role")
        assert "Python identifier" in out


async def test_spawn_rejects_duplicate_role() -> None:
    run_workers: dict[str, Agent] = {"seed": _worker("s")}
    t = _make_spawn_tool_direct(run_workers=run_workers)
    out = await t.fn(role="seed", instructions="clone of seed")
    assert out.startswith("Error: a worker named 'seed' already exists")


async def test_spawn_rejects_empty_instructions() -> None:
    t = _make_spawn_tool_direct()
    out = await t.fn(role="ok_role", instructions="   ")
    assert out.startswith("Error: instructions must be a non-empty")


async def test_max_spawned_cap_end_to_end() -> None:
    """Cap enforced through the real supervisor loop: the second
    spawn's tool result is the error string, and the run completes."""
    template = Agent(
        "template",
        model=ScriptedModel(
            [ScriptedTurn(text="a"), ScriptedTurn(text="b")]
        ),
    )
    parent_model = ScriptedModel(
        [
            ScriptedTurn(tool_calls=[_spawn_call("c1", "one")]),
            ScriptedTurn(tool_calls=[_spawn_call("c2", "two")]),
            ScriptedTurn(text="finished"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(
            workers={"seed": _worker("s")},
            allow_spawn=True,
            max_spawned=1,
            spawn_template=template,
        ),
    )
    events = [e async for e in agent.stream("go")]
    dumped = "\n".join(str(e.payload) for e in events)
    assert "spawn limit reached" in dumped
    final = [e for e in events if e.kind == "run_completed"]
    # Run reached its final text regardless of the blocked spawn.
    assert "finished" in dumped or final


def test_supervisor_rejects_negative_max_spawned() -> None:
    with pytest.raises(ValueError, match="max_spawned"):
        Supervisor(
            workers={"a": _worker("x")}, allow_spawn=True, max_spawned=-1
        )


# ---------------------------------------------------------------------------
# Ephemeral lifecycle — spawned workers die with the run
# ---------------------------------------------------------------------------


async def test_spawned_worker_gone_on_second_run() -> None:
    """Run 1 spawns + delegates. Run 2 delegating to the same role
    gets the unknown-worker error, and the agent-level registry holds
    no spawned handles after either run."""
    template = Agent(
        "template", model=ScriptedModel([ScriptedTurn(text="helper out")])
    )
    parent_model = ScriptedModel(
        [
            # run 1
            ScriptedTurn(tool_calls=[_spawn_call("c1", "helper")]),
            ScriptedTurn(tool_calls=[_delegate_call("c2", "helper")]),
            ScriptedTurn(text="run1 done"),
            # run 2
            ScriptedTurn(tool_calls=[_delegate_call("c3", "helper")]),
            ScriptedTurn(text="run2 done"),
        ]
    )
    coord = Team.supervisor(
        workers={"seed": _worker("s")},
        model=parent_model,
        allow_spawn=True,
        spawn_template=template,
    )
    r1 = await coord.run("first")
    assert "run1 done" in r1.output
    # Agent-level registry cleaned: only the fixed roster remains.
    assert {h.role for h in coord._worker_registry.values()} == {"seed"}

    events = [e async for e in coord.stream("second")]
    dumped = "\n".join(str(e.payload) for e in events)
    assert "unknown worker 'helper'" in dumped
    assert {h.role for h in coord._worker_registry.values()} == {"seed"}


async def test_spawned_handle_mirrored_into_shared_registry_during_run() -> None:
    """Mid-run the spawned handle lives in the shared registry (so
    ``send_message(to=<id>)`` works); the unit-level tool mirrors it."""
    shared: dict[str, Any] = {}
    overlay: dict[str, Any] = {}
    spawned_ids: list[str] = []
    t = _make_spawn_tool_direct(
        spawn_overlay=overlay,
        shared_registry=shared,
        spawned_ids=spawned_ids,
    )
    await t.fn(role="helper", instructions="You help.")
    assert len(spawned_ids) == 1
    assert spawned_ids[0] in shared
    assert shared[spawned_ids[0]] is overlay["helper"]


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


async def test_spawned_handle_pins_spawning_users_id() -> None:
    overlay: dict[str, Any] = {}
    t = _make_spawn_tool_direct(spawn_overlay=overlay)
    ctx = RunContext(
        user_id="alice", session_id="s1", run_id="r1", metadata={}
    )
    async with set_run_context(ctx):
        await t.fn(role="helper", instructions="You help.")
    assert overlay["helper"].user_id == "alice"


async def test_cross_user_delegate_to_spawned_worker_rejected() -> None:
    """Delegating to a spawned worker from a different user's run
    returns the cross-tenant refusal string (mirrors the
    ``acquire_worker_session`` discipline for persistent workers)."""
    run_workers: dict[str, Agent] = {}
    overlay: dict[str, Any] = {}
    spawn = _make_spawn_tool_direct(
        run_workers=run_workers, spawn_overlay=overlay
    )
    alice_ctx = RunContext(
        user_id="alice", session_id="s1", run_id="r1", metadata={}
    )
    async with set_run_context(alice_ctx):
        await spawn.fn(role="helper", instructions="You help.")

    delegate = _make_delegate_tool(
        run_workers,
        AgentSession(id="parent", instructions=""),
        tool_name="delegate",
        memory=InMemoryMemory(),
        spawn_overlay=overlay,
    )
    bob_ctx = RunContext(
        user_id="bob", session_id="s2", run_id="r2", metadata={}
    )
    async with set_run_context(bob_ctx):
        out = await delegate.fn(worker="helper", instructions="hi")
    assert out.startswith("Error:")
    assert "alice" in out and "bob" in out
    assert "cross-tenant" in out.lower() or "rejected" in out.lower()


async def test_same_user_delegate_to_spawned_worker_succeeds() -> None:
    run_workers: dict[str, Agent] = {}
    overlay: dict[str, Any] = {}
    template = Agent(
        "template", model=ScriptedModel([ScriptedTurn(text="helper says hi")])
    )
    spawn = _make_spawn_tool_direct(
        run_workers=run_workers, spawn_overlay=overlay, template=template
    )
    ctx = RunContext(
        user_id="alice", session_id="s1", run_id="r1", metadata={}
    )
    async with set_run_context(ctx):
        await spawn.fn(role="helper", instructions="You help.")
        delegate = _make_delegate_tool(
            run_workers,
            AgentSession(id="parent", instructions=""),
            tool_name="delegate",
            memory=InMemoryMemory(),
            spawn_overlay=overlay,
        )
        out = await delegate.fn(worker="helper", instructions="hi")
    assert "helper says hi" in out
    assert out.startswith("[worker_id: worker_helper_")


# ---------------------------------------------------------------------------
# Template inheritance
# ---------------------------------------------------------------------------


async def test_spawned_worker_inherits_template_model_and_tools() -> None:
    template = Agent(
        "template",
        model=ScriptedModel([ScriptedTurn(text="unused")]),
    )
    run_workers: dict[str, Agent] = {}
    overlay: dict[str, Any] = {}
    t = _make_spawn_tool_direct(
        run_workers=run_workers, spawn_overlay=overlay, template=template
    )
    await t.fn(role="clone", instructions="You are cloned.")
    spawned = run_workers["clone"]
    assert spawned.model is template.model
    assert spawned.tool_host is template.tool_host
    assert spawned.instructions == "You are cloned."


async def test_spawned_worker_without_template_uses_fallback_model() -> None:
    run_workers: dict[str, Agent] = {}
    t = _make_spawn_tool_direct(run_workers=run_workers)
    await t.fn(role="bare", instructions="You are bare.")
    assert isinstance(run_workers["bare"].model, EchoModel)


# ---------------------------------------------------------------------------
# Team.supervisor passthrough
# ---------------------------------------------------------------------------


def test_team_supervisor_spawn_kwargs_passthrough() -> None:
    template = _worker("t")
    coord = Team.supervisor(
        workers={"seed": _worker("s")},
        model="echo",
        allow_spawn=True,
        max_spawned=2,
        spawn_template=template,
    )
    arch = coord.architecture
    assert isinstance(arch, Supervisor)
    assert arch._allow_spawn is True
    assert arch._max_spawned == 2
    assert arch._spawn_template is template


def test_team_supervisor_spawn_disabled_by_default() -> None:
    coord = Team.supervisor(
        workers={"seed": _worker("s")}, model="echo"
    )
    arch = coord.architecture
    assert isinstance(arch, Supervisor)
    assert arch._allow_spawn is False
    assert arch._max_spawned == 5
    assert arch._spawn_template is None
