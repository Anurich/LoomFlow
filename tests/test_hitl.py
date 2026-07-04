"""G8 — rich HITL: edit-args / remember-decision / interrupt-park.

Contract under test:

* ``bool`` approval handlers keep working exactly as before (allow +
  deny) — 100% back-compat.
* ``ApprovalDecision(action="edit")`` runs the tool with the edited
  args; the audit trail carries a ``tool_call_edited`` entry with both
  the edited and the original args.
* ``remember_allow`` / ``remember_deny`` cache the decision per
  ``(user_id, tool)`` for the run — the handler is consulted at most
  once per key; deny stays fail-closed.
* The remember cache is keyed per user (unit-level on the resolver).
* A raising handler is still a deny (fail-closed).
* Interrupt/park: an approval handler can park on the runtime signal
  channel; ``Agent.signal`` unblocks it — proven through a full
  ``agent.run``. ``wait_for_user_signal`` emits ``interrupt.waiting``
  and returns the delivered payload; runtimes without signal support
  warn and no-op instead of deadlocking.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from loomflow import (
    Agent,
    ApprovalDecision,
    InMemoryAuditLog,
    InMemoryMemory,
    InProcRuntime,
    Mode,
    StandardPermissions,
    get_run_context,
)
from loomflow.architecture.helpers import (
    _resolve_ask_decision,
    wait_for_user_signal,
)
from loomflow.core.types import Event, ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _destructive_turns(
    tool: str, args: dict[str, Any], n: int = 1, final: str = "done"
) -> list[ScriptedTurn]:
    """N turns each planning one destructive call, then a text turn."""
    return [
        ScriptedTurn(
            tool_calls=[ToolCall(tool=tool, args=dict(args), destructive=True)]
        )
        for _ in range(n)
    ] + [ScriptedTurn(text=final)]


def _gated_agent(
    model: ScriptedModel,
    tools: list[Any],
    handler: Any,
    **kwargs: Any,
) -> Agent:
    return Agent(
        "act on request",
        model=model,
        memory=InMemoryMemory(),
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        tools=tools,
        approval_handler=handler,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# bool back-compat
# ---------------------------------------------------------------------------


async def test_bool_true_still_allows() -> None:
    executed: list[str] = []

    def _rm(_path: str) -> str:
        executed.append(_path)
        return "removed"

    async def approve(_call: ToolCall, _uid: str | None) -> bool:
        return True

    agent = _gated_agent(
        ScriptedModel(_destructive_turns("_rm", {"_path": "/tmp/x"})),
        [_rm],
        approve,
    )
    result = await agent.run("go", user_id="alice")
    assert result.output == "done"
    assert executed == ["/tmp/x"]


async def test_bool_false_still_denies() -> None:
    executed: list[str] = []

    def _rm(_path: str) -> str:
        executed.append(_path)
        return "removed"

    async def decline(_call: ToolCall, _uid: str | None) -> bool:
        return False

    agent = _gated_agent(
        ScriptedModel(
            _destructive_turns("_rm", {"_path": "/tmp/x"}, final="blocked")
        ),
        [_rm],
        decline,
    )
    result = await agent.run("go", user_id="alice")
    assert result.output == "blocked"
    assert executed == []


# ---------------------------------------------------------------------------
# edit-args
# ---------------------------------------------------------------------------


async def test_edit_args_executes_tool_with_edited_args() -> None:
    executed: list[str] = []

    def _rm(_path: str) -> str:
        executed.append(_path)
        return f"removed {_path}"

    async def edit(call: ToolCall, _uid: str | None) -> ApprovalDecision:
        assert call.args == {"_path": "/etc/passwd"}
        return ApprovalDecision(
            action="edit",
            edited_args={"_path": "/tmp/safe"},
            reason="redirected to a safe path",
        )

    audit_log = InMemoryAuditLog(secret="s3")
    agent = _gated_agent(
        ScriptedModel(_destructive_turns("_rm", {"_path": "/etc/passwd"})),
        [_rm],
        edit,
        audit_log=audit_log,
    )
    result = await agent.run("go", user_id="alice", session_id="edit-sess")
    assert result.output == "done"
    # The tool body ran with the EDITED args, not the model-planned ones.
    assert executed == ["/tmp/safe"]

    # The audit trail reflects the edited call.
    entries = await audit_log.all_entries()
    edited = [e for e in entries if e.action == "tool_call_edited"]
    assert len(edited) == 1
    assert edited[0].payload["args"] == {"_path": "/tmp/safe"}
    assert edited[0].payload["original_args"] == {"_path": "/etc/passwd"}
    assert edited[0].session_id == "edit-sess"


async def test_edit_without_edited_args_is_plain_allow() -> None:
    executed: list[str] = []

    def _rm(_path: str) -> str:
        executed.append(_path)
        return "removed"

    async def edit_noop(_call: ToolCall, _uid: str | None) -> ApprovalDecision:
        return ApprovalDecision(action="edit")  # edited_args=None → keep

    agent = _gated_agent(
        ScriptedModel(_destructive_turns("_rm", {"_path": "/tmp/x"})),
        [_rm],
        edit_noop,
    )
    result = await agent.run("go", user_id="alice")
    assert result.output == "done"
    assert executed == ["/tmp/x"]


# ---------------------------------------------------------------------------
# remember_allow / remember_deny
# ---------------------------------------------------------------------------


async def test_remember_allow_asks_once_across_two_calls() -> None:
    executed: list[str] = []
    handler_calls: list[str] = []

    def _rm(_path: str) -> str:
        executed.append(_path)
        return "removed"

    async def remember(call: ToolCall, _uid: str | None) -> ApprovalDecision:
        handler_calls.append(call.tool)
        return ApprovalDecision(action="remember_allow")

    agent = _gated_agent(
        ScriptedModel(_destructive_turns("_rm", {"_path": "/tmp/x"}, n=2)),
        [_rm],
        remember,
    )
    result = await agent.run("go", user_id="alice")
    assert result.output == "done"
    # Two gated calls executed...
    assert executed == ["/tmp/x", "/tmp/x"]
    # ...but the human was asked exactly once.
    assert handler_calls == ["_rm"]


async def test_remember_deny_denies_without_reasking() -> None:
    executed: list[str] = []
    handler_calls: list[str] = []

    def _rm(_path: str) -> str:
        executed.append(_path)
        return "removed"

    async def remember_no(
        call: ToolCall, _uid: str | None
    ) -> ApprovalDecision:
        handler_calls.append(call.tool)
        return ApprovalDecision(action="remember_deny", reason="never")

    agent = _gated_agent(
        ScriptedModel(
            _destructive_turns("_rm", {"_path": "/tmp/x"}, n=2, final="no")
        ),
        [_rm],
        remember_no,
    )
    result = await agent.run("go", user_id="alice")
    assert result.output == "no"
    # Fail-closed on every call; handler consulted only once.
    assert executed == []
    assert handler_calls == ["_rm"]


async def test_remember_cache_does_not_leak_across_runs() -> None:
    """approval_memory is re-initialised per run — a remembered allow
    from run 1 must not auto-approve run 2."""
    handler_calls: list[str] = []

    def _rm(_path: str) -> str:
        return "removed"

    async def remember(call: ToolCall, _uid: str | None) -> ApprovalDecision:
        handler_calls.append(call.tool)
        return ApprovalDecision(action="remember_allow")

    model = ScriptedModel(
        _destructive_turns("_rm", {"_path": "/a"})
        + _destructive_turns("_rm", {"_path": "/b"})
    )
    agent = _gated_agent(model, [_rm], remember)
    await agent.run("one", user_id="alice")
    await agent.run("two", user_id="alice")
    # Asked once per RUN, not once per agent lifetime.
    assert handler_calls == ["_rm", "_rm"]


async def test_remember_cache_is_keyed_per_user() -> None:
    """Unit-level: the (user_id, tool) key partitions decisions."""
    memory: dict[tuple[str | None, str], bool] = {}
    handler_users: list[str | None] = []

    async def remember(
        _call: ToolCall, user_id: str | None
    ) -> ApprovalDecision:
        handler_users.append(user_id)
        return ApprovalDecision(action="remember_allow")

    call = ToolCall(tool="rm", args={"path": "/x"}, destructive=True)

    ok1, _, _ = await _resolve_ask_decision(
        call, remember, "alice", approval_memory=memory
    )
    ok2, _, _ = await _resolve_ask_decision(
        call, remember, "alice", approval_memory=memory
    )
    ok3, _, _ = await _resolve_ask_decision(
        call, remember, "bob", approval_memory=memory
    )
    assert ok1 and ok2 and ok3
    # alice cached after the first ask; bob is a distinct key.
    assert handler_users == ["alice", "bob"]
    assert memory == {("alice", "rm"): True, ("bob", "rm"): True}


async def test_remembered_deny_reason_surfaces() -> None:
    memory: dict[tuple[str | None, str], bool] = {("alice", "rm"): False}
    call = ToolCall(tool="rm", args={}, destructive=True)

    async def never_called(
        _call: ToolCall, _uid: str | None
    ) -> ApprovalDecision:
        raise AssertionError("handler must not be consulted")

    ok, _, reason = await _resolve_ask_decision(
        call, never_called, "alice", approval_memory=memory
    )
    assert ok is False
    assert reason == "denied by remembered decision"


# ---------------------------------------------------------------------------
# fail-closed
# ---------------------------------------------------------------------------


async def test_handler_raising_is_deny() -> None:
    executed: list[str] = []

    def _rm(_path: str) -> str:
        executed.append(_path)
        return "removed"

    async def boom(_call: ToolCall, _uid: str | None) -> ApprovalDecision:
        raise RuntimeError("approval UI down")

    agent = _gated_agent(
        ScriptedModel(
            _destructive_turns("_rm", {"_path": "/tmp/x"}, final="aborted")
        ),
        [_rm],
        boom,
    )
    result = await agent.run("go", user_id="alice")
    assert result.output == "aborted"
    assert executed == []


# ---------------------------------------------------------------------------
# interrupt / park — signal round-trip through a real run
# ---------------------------------------------------------------------------


async def test_approval_handler_parks_and_agent_signal_resumes() -> None:
    runtime = InProcRuntime()
    parked = anyio.Event()
    executed: list[str] = []

    def _deploy(_target: str) -> str:
        executed.append(_target)
        return "deployed"

    async def parking_handler(
        _call: ToolCall, user_id: str | None
    ) -> ApprovalDecision:
        # The handler asks out-of-band (here: the test task) and parks
        # on the runtime signal channel until the caller decides.
        ctx = get_run_context()
        assert ctx.session_id == "hitl-park"
        assert user_id == "alice"
        parked.set()
        payload = await runtime.wait_for_signal(ctx.session_id, "approval")
        return ApprovalDecision(action=payload["action"])

    agent = Agent(
        "deploy when approved",
        model=ScriptedModel(
            _destructive_turns("_deploy", {"_target": "prod"})
        ),
        memory=InMemoryMemory(),
        runtime=runtime,
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        tools=[_deploy],
        approval_handler=parking_handler,
    )

    result_box: dict[str, Any] = {}

    async def _run() -> None:
        result_box["result"] = await agent.run(
            "ship it", session_id="hitl-park", user_id="alice"
        )

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_run)
            await parked.wait()  # run is genuinely parked mid-approval
            await agent.signal(
                "hitl-park", "approval", {"action": "allow"}
            )

    assert result_box["result"].output == "done"
    assert executed == ["prod"]


async def test_wait_for_user_signal_emits_event_and_returns_payload() -> None:
    runtime = InProcRuntime()
    deps = SimpleNamespace(runtime=runtime)
    events: list[Event] = []

    async def emit(ev: Event) -> None:
        events.append(ev)

    got: dict[str, Any] = {}

    async def waiter() -> None:
        got["payload"] = await wait_for_user_signal(
            deps,  # type: ignore[arg-type]
            "sig-sess",
            "resume",
            emit=emit,
        )

    with anyio.fail_after(5):
        async with runtime.session("sig-sess"):
            async with anyio.create_task_group() as tg:
                tg.start_soon(waiter)
                await anyio.sleep(0.01)
                await runtime.signal("sig-sess", "resume", {"go": True})

    assert got["payload"] == {"go": True}
    assert len(events) == 1
    assert events[0].payload["name"] == "interrupt.waiting"
    assert events[0].payload["signal"] == "resume"
    assert events[0].session_id == "sig-sess"


async def test_wait_for_user_signal_unsupported_runtime_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    deps = SimpleNamespace(runtime=SimpleNamespace(name="bare"))
    with caplog.at_level(
        logging.WARNING, logger="loomflow.architecture.helpers"
    ):
        with anyio.fail_after(1):  # must NOT park
            payload = await wait_for_user_signal(
                deps,  # type: ignore[arg-type]
                "s1",
            )
    assert payload is None
    assert any(
        "does not support signals" in r.message for r in caplog.records
    )


async def test_agent_signal_unsupported_runtime_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class NoSignalRuntime:
        name = "nosignal"

        async def step(self, name: str, fn: Any, *a: Any, **kw: Any) -> Any:
            return await fn(*a, **kw)

        def stream_step(self, name: str, fn: Any, *a: Any, **kw: Any) -> Any:
            return fn(*a, **kw)

    agent = Agent(
        "hi",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        memory=InMemoryMemory(),
        runtime=NoSignalRuntime(),  # type: ignore[arg-type]
    )
    with caplog.at_level(logging.WARNING, logger="loomflow.agent"):
        await agent.signal("s1", "approval", {"ok": True})  # no raise
    assert any(
        "does not support signals" in r.message for r in caplog.records
    )
