"""Pin the destructive-flag propagation chain.

Background: pre-0.10.17, ``Tool.to_def()`` dropped the
``destructive`` field, and model adapters (openai, anthropic)
constructed ``ToolCall`` without setting it. That meant a call to
a ``destructive=True`` tool arrived at ``permissions.check`` with
``destructive=False``, ``StandardPermissions`` returned ``allow_``
instead of ``ask_``, and the approval handler was **never called**
— so any framework user with a wired ``ApprovalGate`` was silently
auto-approving every write/edit/bash.

Fix landed in three places:

* ``ToolDef`` carries a ``destructive`` field (``core/types.py``).
* ``Tool.to_def()`` propagates it (``tools/registry.py``).
* ``_run_single_tool`` in ReAct does a defensive lookup via
  ``deps.tools.list_tools()`` and stamps ``call.destructive`` from
  the matching def before the permissions check. Backstop against
  adapters that don't propagate the flag from ToolDef → ToolCall
  themselves.

These tests pin all three so a future edit to any of the layers
can't silently re-introduce the auto-approve bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.core.types import ToolCall
from loomflow.security import StandardPermissions
from loomflow.tools import edit_tool, read_tool, write_tool

pytestmark = pytest.mark.anyio


# --- Layer 1: Tool.to_def() carries the flag -------------------------------


def test_tool_to_def_propagates_destructive_true() -> None:
    """edit_tool/write_tool/bash_tool all set ``destructive=True``
    at the Tool level; ``to_def()`` must carry that forward so model
    adapters and the permissions stamp can read it."""
    from loomflow.tools.builtin import bash_tool

    for factory in (edit_tool, write_tool, bash_tool):
        t = factory("/tmp")
        d = t.to_def()
        assert d.destructive is True, (
            f"{factory.__name__}.to_def() dropped destructive flag"
        )


def test_tool_to_def_propagates_destructive_false() -> None:
    """Read-only tools (``destructive=False``) round-trip too."""
    t = read_tool("/tmp")
    d = t.to_def()
    assert d.destructive is False


# --- Layer 2: ReAct stamps from the host before permissions check ---------


async def test_destructive_edit_triggers_approval_handler(
    tmp_path: Path,
) -> None:
    """End-to-end: a destructive tool call routes through the
    handler even when the ScriptedModel emits a ToolCall with
    ``destructive=False`` (matching what real adapters do)."""
    target = tmp_path / "foo.py"
    target.write_text("def hello():\n    return 'world'\n")

    handler_calls: list[ToolCall] = []

    async def recording_handler(
        call: ToolCall, user_id: str | None = None
    ) -> bool:
        handler_calls.append(call)
        return True  # approve

    agent = Agent(
        instructions="You edit files.",
        model=ScriptedModel(
            turns=[
                ScriptedTurn(
                    text="",
                    tool_calls=[
                        ToolCall(
                            tool="edit",
                            args={
                                "path": "foo.py",
                                "old_string": "world",
                                "new_string": "loomflow",
                            },
                        ),
                    ],
                ),
                ScriptedTurn(text="done"),
            ]
        ),
        tools=[edit_tool(tmp_path), read_tool(tmp_path)],
        permissions=StandardPermissions(),
        approval_handler=recording_handler,
    )

    await agent.run("edit foo.py")

    # Handler MUST be called exactly once for the destructive edit.
    assert len(handler_calls) == 1
    assert handler_calls[0].tool == "edit"
    # Handler returned True → edit went through.
    assert "loomflow" in target.read_text()


async def test_denied_edit_skips_execution(tmp_path: Path) -> None:
    """Handler returning False → tool MUST NOT execute. This is the
    teeth of the gate; if it doesn't actually skip, the gate is just
    decorative."""
    target = tmp_path / "foo.py"
    original = "def hello():\n    return 'world'\n"
    target.write_text(original)

    async def deny_handler(
        call: ToolCall, user_id: str | None = None
    ) -> bool:
        return False  # deny

    agent = Agent(
        instructions="You edit files.",
        model=ScriptedModel(
            turns=[
                ScriptedTurn(
                    text="",
                    tool_calls=[
                        ToolCall(
                            tool="edit",
                            args={
                                "path": "foo.py",
                                "old_string": "world",
                                "new_string": "loomflow",
                            },
                        ),
                    ],
                ),
                ScriptedTurn(text="done"),
            ]
        ),
        tools=[edit_tool(tmp_path), read_tool(tmp_path)],
        permissions=StandardPermissions(),
        approval_handler=deny_handler,
    )

    await agent.run("edit foo.py")
    assert target.read_text() == original


async def test_read_tool_does_not_trigger_handler(tmp_path: Path) -> None:
    """Non-destructive tools bypass the handler entirely — no
    spurious prompts for read/grep/find/ls."""
    target = tmp_path / "foo.py"
    target.write_text("hello")

    handler_calls: list[ToolCall] = []

    async def recording_handler(
        call: ToolCall, user_id: str | None = None
    ) -> bool:
        handler_calls.append(call)
        return True

    agent = Agent(
        instructions="You read files.",
        model=ScriptedModel(
            turns=[
                ScriptedTurn(
                    text="",
                    tool_calls=[
                        ToolCall(tool="read", args={"path": "foo.py"}),
                    ],
                ),
                ScriptedTurn(text="done"),
            ]
        ),
        tools=[read_tool(tmp_path)],
        permissions=StandardPermissions(),
        approval_handler=recording_handler,
    )

    await agent.run("read foo.py")
    assert handler_calls == []
