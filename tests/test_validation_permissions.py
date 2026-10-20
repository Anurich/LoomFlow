"""Every path to a tool goes through the run's permission gates.

Each test here is a way a call used to slip past the deny-list, hooks
or approval handler: code mode, skill-shipped tools, ``ACCEPT_EDITS``,
approval handlers returning a truthy "no", timed-out or raising hooks,
and workers spawned from a template.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, tool
from loomflow.core.types import PermissionDecision, ToolCall
from loomflow.security.audit import InMemoryAuditLog
from loomflow.security.hooks import HookRegistry
from loomflow.security.permissions import Mode, StandardPermissions
from loomflow.security.sandbox import FilesystemSandbox
from loomflow.skills import Skill
from loomflow.team import Team
from loomflow.tools.builtin import edit_tool, write_tool
from loomflow.tools.code_mode import make_code_mode_tools
from loomflow.tools.registry import InProcessToolHost, Tool

pytestmark = pytest.mark.anyio


def _calls(*calls: ToolCall) -> ScriptedModel:
    return ScriptedModel([ScriptedTurn(tool_calls=list(calls)), ScriptedTurn(text="done")])


def _deleter(ran: list[str]) -> Tool:
    @tool(destructive=True)
    async def delete_user(user: str) -> str:
        """Delete a user account."""
        ran.append(user)
        return f"deleted {user}"

    return delete_user


async def test_accept_edits_approves_file_edits_only(tmp_path: Path) -> None:
    ran: list[str] = []
    asked: list[str] = []

    async def deny_all(call: ToolCall, user_id: str | None) -> bool:
        asked.append(call.tool)
        return False

    agent = Agent(
        "x",
        model=_calls(
            ToolCall(id="w", tool="write", args={"path": "a.txt", "content": "hi"}),
            ToolCall(id="d", tool="delete_user", args={"user": "bob"}),
        ),
        tools=[write_tool(tmp_path), edit_tool(tmp_path), _deleter(ran)],
        permissions=StandardPermissions(mode=Mode.ACCEPT_EDITS),
        approval_handler=deny_all,
    )
    await agent.run("go")
    assert (tmp_path / "a.txt").read_text() == "hi"  # edit auto-approved
    assert asked == ["delete_user"] and ran == []  # anything else still asks

    custom = StandardPermissions(mode=Mode.ACCEPT_EDITS, edit_tools=["patch"])
    call = ToolCall(tool="patch", args={}, destructive=True)
    assert (await custom.check(call, context={})).allow
    call = ToolCall(tool="write", args={}, destructive=True)
    assert (await custom.check(call, context={})).ask


@pytest.mark.parametrize(
    ("decision", "allowed"),
    [
        (True, True),
        ("allow", True),
        (False, False),
        ("deny", False),
        ("no", False),
        ({"approved": False}, False),
        ({"approved": True}, False),  # unknown shapes never allow
        (None, False),
        (1, False),
    ],
)
async def test_approval_handler_only_allows_explicit_approvals(
    decision: Any, allowed: bool
) -> None:
    ran: list[str] = []

    async def handler(call: ToolCall, user_id: str | None) -> Any:
        return decision

    agent = Agent(
        "x",
        model=_calls(ToolCall(tool="delete_user", args={"user": "bob"})),
        tools=[_deleter(ran)],
        permissions=StandardPermissions(),
        approval_handler=handler,
    )
    await agent.run("go")
    assert ran == (["bob"] if allowed else [])


async def test_pre_tool_hooks_that_time_out_or_raise_deny() -> None:
    async def slow(call: ToolCall) -> PermissionDecision | None:
        await anyio.sleep(1)
        return None

    async def broken(call: ToolCall) -> PermissionDecision | None:
        raise RuntimeError("policy service down")

    for hook, timeout_denies, expect_ran in (
        (slow, True, False),
        (slow, False, True),
        (broken, True, False),
    ):
        ran: list[str] = []
        hooks = HookRegistry(hook_timeout_s=0.05, timeout_denies=timeout_denies)
        hooks.register_pre_tool(hook)

        @tool
        async def send(to: str, _ran: list[str] = ran) -> str:
            """Send."""
            _ran.append(to)
            return "sent"

        agent = Agent(
            "x",
            model=_calls(ToolCall(tool="send", args={"to": "x"})),
            tools=[send],
            hooks=hooks,
        )
        result = await agent.run("go")  # a broken hook doesn't crash the run
        assert result.output == "done"
        assert bool(ran) is expect_ran


async def test_code_mode_tool_calls_go_through_the_gates() -> None:
    ran: list[str] = []
    asked: list[str] = []
    seen_by_hook: list[str] = []
    hooks = HookRegistry()

    @hooks.register_pre_tool
    async def watch(call: ToolCall) -> PermissionDecision | None:
        seen_by_hook.append(call.tool)
        return None

    async def approve_code_only(call: ToolCall, user_id: str | None) -> bool:
        asked.append(call.tool)
        return call.tool == "run_code"

    audit = InMemoryAuditLog(secret="k" * 32)
    run_code = ToolCall(tool="run_code", args={"code": "result = await delete_user(user='bob')"})

    # Deny-list: the inner tool is refused even though run_code ran.
    agent = Agent(
        "x",
        model=_calls(run_code),
        tools=make_code_mode_tools([_deleter(ran)]),
        permissions=StandardPermissions(denied_tools=["delete_user"]),
        approval_handler=approve_code_only,
        hooks=hooks,
        audit_log=audit,
    )
    await agent.run("go")
    assert ran == []
    assert "delete_user" in seen_by_hook
    assert [e.action for e in await audit.query() if "delete_user" in str(e.payload)].count(
        "tool_result"
    ) == 1

    # Approval: approving "run some code" doesn't approve what it calls.
    asked.clear()
    agent = Agent(
        "x",
        model=_calls(run_code),
        tools=make_code_mode_tools([_deleter(ran)]),
        permissions=StandardPermissions(),
        approval_handler=approve_code_only,
    )
    await agent.run("go")
    assert asked == ["run_code", "delete_user"] and ran == []


def _peek_skill(root: Path) -> Path:
    sk = root / "skills" / "file-peek"
    (sk / "scripts").mkdir(parents=True)
    (sk / "scripts" / "peek.py").write_text("import sys; print(open(sys.argv[1]).read())\n")
    (sk / "SKILL.md").write_text(
        "---\nname: file-peek\ndescription: Peek at a file.\ntools:\n"
        "  peek:\n    description: print a file\n    script: scripts/peek.py\n"
        "    args:\n      path:\n        type: string\n        description: file path\n"
        "---\nUse file_peek__peek.\n"
    )
    return sk


async def test_skill_tools_are_destructive_and_run_under_the_sandbox(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("OUTSIDE-SECRET")
    asked: list[str] = []

    async def approve(call: ToolCall, user_id: str | None) -> bool:
        asked.append(call.tool)
        return True

    agent = Agent(
        "x",
        model=ScriptedModel(
            [
                ScriptedTurn(tool_calls=[ToolCall(tool="load_skill", args={"name": "file-peek"})]),
                ScriptedTurn(
                    tool_calls=[ToolCall(tool="file_peek__peek", args={"path": str(secret)})]
                ),
                ScriptedTurn(text="done"),
            ]
        ),
        tools=FilesystemSandbox(InProcessToolHost([]), roots=[root]),
        skills=[Skill(_peek_skill(tmp_path))],
        permissions=StandardPermissions(),
        approval_handler=approve,
    )
    outputs = [
        ev.payload["result"]
        async for ev in agent.stream("go")
        if ev.kind == "tool_result"
    ]
    assert asked == ["file_peek__peek"]
    assert outputs[-1]["denied"] and "outside the allowed roots" in outputs[-1]["reason"]
    assert not any("OUTSIDE-SECRET" in str(o.get("output")) for o in outputs)


async def test_spawned_worker_keeps_the_templates_permissions() -> None:
    ran: list[str] = []

    @tool
    async def danger(x: str) -> str:
        """Dangerous op."""
        ran.append(x)
        return "EXECUTED"

    template = Agent(
        "tmpl",
        model=_calls(ToolCall(id="d1", tool="danger", args={"x": "spawned"})),
        tools=[danger],
        permissions=StandardPermissions(denied_tools=["danger"]),
    )
    coordinator = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="s1",
                        tool="spawn_worker",
                        args={"role": "helper", "instructions": "You help."},
                    )
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="s2",
                        tool="delegate",
                        args={"worker": "helper", "instructions": "run danger"},
                    )
                ]
            ),
            ScriptedTurn(text="fin"),
        ]
    )
    team = Team.supervisor(
        {"w": Agent("worker", model=ScriptedModel([]))},
        model=coordinator,
        allow_spawn=True,
        spawn_template=template,
    )
    await team.run("go")
    assert ran == []


class _AsyncCallable:
    async def __call__(self, x: int) -> int:
        return x * 2


async def test_async_callable_objects_and_partials_are_awaited() -> None:
    async def add(x: int, y: int) -> int:
        return x + y

    obj = Tool(name="obj", description="d", fn=_AsyncCallable())
    part = Tool(name="part", description="d", fn=functools.partial(add, y=1))
    assert await obj.execute({"x": 2}) == 4
    assert await part.execute({"x": 2}) == 3


async def test_sync_tool_timeout_returns_promptly() -> None:
    # A timeout around a blocking sync tool used to wait for the thread.
    @tool
    def stuck() -> str:
        """Blocks its thread."""
        time.sleep(1.0)
        return "late"

    started = time.monotonic()
    with anyio.move_on_after(0.1) as scope:
        await stuck.execute({})
    assert scope.cancelled_caught
    assert time.monotonic() - started < 0.6
