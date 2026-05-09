"""M10.4 — approval handler resolves Decision.ask outcomes.

Before M10, a permissions policy returning ``Decision.ask_(...)``
was silently treated as a deny — the loop never gave the
application a chance to prompt a human / Slack / ticketing
system. ``Agent(approval_handler=...)`` closes the gap.

Contract under test:

* When permissions return ``ask`` and an approval handler is
  registered, the handler is invoked with ``(call, user_id)``;
  the returned bool decides allow vs deny.
* When the handler is absent, ``ask`` falls back to deny — the
  agent never silently bypasses the gate.
* When the handler raises, the call is treated as a deny (and
  logged). A buggy approval flow must not green-light a tool the
  policy explicitly wanted gated.
* The user_id from the live RunContext flows through to the
  handler so per-user approval flows can route correctly.
"""

from __future__ import annotations

import pytest

from loomflow import Agent, InMemoryMemory, Mode, StandardPermissions
from loomflow.core.types import ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _destructive_call(tool: str = "rm", args: dict | None = None) -> ToolCall:
    return ToolCall(tool=tool, args=args or {"path": "/tmp/x"}, destructive=True)


def _delete_tool(_path: str) -> str:
    """A trivial destructive tool. ``StandardPermissions(mode=DEFAULT)``
    will issue ``Decision.ask_`` for destructive calls, which is the
    branch the approval handler needs to handle."""
    return "deleted"


# ---------------------------------------------------------------------------
# Approval handler resolution
# ---------------------------------------------------------------------------


async def test_handler_approves_lets_tool_run() -> None:
    seen: list[tuple[str, str | None]] = []

    async def approve(call: ToolCall, user_id: str | None) -> bool:
        seen.append((call.tool, user_id))
        return True

    agent = Agent(
        "delete things when asked",
        model=ScriptedModel([
            ScriptedTurn(tool_calls=[
                ToolCall(tool="_delete_tool", args={"_path": "/tmp/x"},
                         destructive=True)
            ]),
            ScriptedTurn(text="done"),
        ]),
        memory=InMemoryMemory(),
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        tools=[_delete_tool],
        approval_handler=approve,
    )

    result = await agent.run("go", user_id="alice")
    assert result.output == "done"
    # The handler was consulted with (tool, user_id) for the
    # destructive call; the run completed without a deny.
    assert seen == [("_delete_tool", "alice")]


async def test_handler_declines_blocks_tool_execution() -> None:
    """A declining handler must prevent the tool from running. The
    canary is incrementing a counter inside the tool — if the
    counter ticks, the deny gate failed."""
    invocations: list[str] = []

    def _delete_tracking(_path: str) -> str:
        invocations.append(_path)
        return "deleted"

    decline_seen: list[str] = []

    async def decline(call: ToolCall, _uid: str | None) -> bool:
        decline_seen.append(call.tool)
        return False

    agent = Agent(
        "delete things when asked",
        model=ScriptedModel([
            ScriptedTurn(tool_calls=[
                ToolCall(tool="_delete_tracking",
                         args={"_path": "/tmp/x"},
                         destructive=True)
            ]),
            ScriptedTurn(text="aborted"),
        ]),
        memory=InMemoryMemory(),
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        tools=[_delete_tracking],
        approval_handler=decline,
    )

    result = await agent.run("go", user_id="alice")
    assert result.output == "aborted"
    # The handler was consulted...
    assert decline_seen == ["_delete_tracking"]
    # ...and the tool body was NEVER executed.
    assert invocations == []


async def test_no_handler_falls_back_to_deny() -> None:
    """``ask`` without a handler is a deny — the historical
    fallback. Single-tenant code that never wanted approvals
    keeps working. The tool body must NOT execute."""
    invocations: list[str] = []

    def _delete_tracking(_path: str) -> str:
        invocations.append(_path)
        return "deleted"

    agent = Agent(
        "delete things when asked",
        model=ScriptedModel([
            ScriptedTurn(tool_calls=[
                ToolCall(tool="_delete_tracking",
                         args={"_path": "/tmp/x"},
                         destructive=True)
            ]),
            ScriptedTurn(text="ok"),
        ]),
        memory=InMemoryMemory(),
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        tools=[_delete_tracking],
        # approval_handler not set — should deny
    )

    result = await agent.run("go", user_id="alice")
    assert result.output == "ok"
    assert invocations == []


async def test_handler_raising_treated_as_deny() -> None:
    """A raising approval handler must NOT be treated as approval —
    that would let UI plumbing bugs silently bypass the gate. The
    canary is the tool's own counter."""
    invocations: list[str] = []

    def _delete_tracking(_path: str) -> str:
        invocations.append(_path)
        return "deleted"

    async def boom(_call: ToolCall, _uid: str | None) -> bool:
        raise RuntimeError("approval system down")

    agent = Agent(
        "delete things when asked",
        model=ScriptedModel([
            ScriptedTurn(tool_calls=[
                ToolCall(tool="_delete_tracking",
                         args={"_path": "/tmp/x"},
                         destructive=True)
            ]),
            ScriptedTurn(text="aborted"),
        ]),
        memory=InMemoryMemory(),
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        tools=[_delete_tracking],
        approval_handler=boom,
    )

    result = await agent.run("go", user_id="alice")
    assert result.output == "aborted"
    assert invocations == []
