"""Security regression tests — audit redaction, MCP destructive
defaults, ``bash_tool`` timeouts, ``FilesystemSandbox`` around a
shell, and guardrail verdicts that fail closed.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from loomflow import Agent
from loomflow.core.types import ToolCall
from loomflow.guardrails.base import GuardVerdict, apply_guardrails
from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec
from loomflow.mcp.registry import _annotated_destructive
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.security.audit import (
    FileAuditLog,
    FullTranscriptAuditLog,
    InMemoryAuditLog,
    verify_signature,
)
from loomflow.security.sandbox.filesystem import FilesystemSandbox
from loomflow.security.secrets import DictSecrets, EnvSecrets, redact_payload
from loomflow.tools.builtin import bash_tool, read_tool
from loomflow.tools.registry import InProcessToolHost, tool

pytestmark = pytest.mark.anyio

KEY_A = "sk-" + "A" * 30
KEY_B = "sk-" + "B" * 30
KEY_C = "sk-" + "C" * 30


# ---------------------------------------------------------------------------
# 6. audit payloads are redacted before they are signed / written
# ---------------------------------------------------------------------------


async def test_inmemory_audit_redacts_key_patterns_before_signing() -> None:
    log = InMemoryAuditLog(secret="k" * 32)
    entry = await log.append(
        session_id="s",
        actor="tool",
        action="tool_call",
        payload={
            "args": {"header": f"X-Key: {KEY_A}", "auth": "Bearer abcdefghijklmnop1234"},
            "nested": [{"deep": f"OPENAI_API_KEY={KEY_B}"}, 7, None],
            "note": "nothing secret here",
        },
        user_id="u",
    )
    blob = json.dumps(entry.payload)
    assert KEY_A not in blob and KEY_B not in blob
    assert "abcdefghijklmnop1234" not in blob
    assert "[REDACTED]" in blob
    assert entry.payload["note"] == "nothing secret here"
    assert entry.payload["nested"][1:] == [7, None]
    # Signed over the redacted form: it still verifies.
    assert verify_signature(entry, "k" * 32)
    assert (await log.verify()).ok


async def test_file_audit_log_never_writes_the_secret(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = FileAuditLog(path, secret="k" * 32)
    await log.append(
        session_id="s", actor="user", action="run_started",
        payload={"prompt": f"my key is {KEY_C}"},
    )
    raw = path.read_text(encoding="utf-8")
    assert KEY_C not in raw and "[REDACTED]" in raw
    assert (await log.verify()).ok
    # The full-transcript wrapper redacts too (it wraps a shipped log).
    full = FullTranscriptAuditLog(FileAuditLog(tmp_path / "full.jsonl", secret="k" * 32))
    await full.append(
        session_id="s", actor="user", action="run_started",
        payload={"prompt": f"my key is {KEY_C}"},
    )
    assert KEY_C not in (tmp_path / "full.jsonl").read_text(encoding="utf-8")


async def test_audit_redacts_known_secret_values_without_a_key_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DictSecrets({"DB_PASSWORD": "correct-horse-battery-staple"})
    monkeypatch.setenv("LOOM_TEST_WEBHOOK_SECRET", "whsec-plain-looking-value")
    assert EnvSecrets().lookup_sync("LOOM_TEST_WEBHOOK_SECRET")
    stored = DictSecrets()
    await stored.store("LATER", "stored-after-construction-9")

    log = InMemoryAuditLog(secret="k" * 32)
    entry = await log.append(
        session_id="s", actor="tool", action="tool_result",
        payload={
            "output": (
                "dsn=postgres://svc:correct-horse-battery-staple@db/prod "
                "hook=whsec-plain-looking-value later=stored-after-construction-9"
            )
        },
    )
    text = entry.payload["output"]
    for secret in (
        "correct-horse-battery-staple",
        "whsec-plain-looking-value",
        "stored-after-construction-9",
    ):
        assert secret not in text
    assert text.startswith("dsn=postgres://svc:[REDACTED]@db/prod")
    assert DictSecrets().redact("x correct-horse-battery-staple y") == "x [REDACTED] y"


async def test_audit_redact_can_be_customised_or_disabled() -> None:
    verbatim = InMemoryAuditLog(secret="k" * 32, redact=False)
    entry = await verbatim.append(
        session_id="s", actor="a", action="x", payload={"v": KEY_A}
    )
    assert entry.payload["v"] == KEY_A

    custom = InMemoryAuditLog(secret="k" * 32, redact=lambda s: s.replace("pony", "***"))
    entry = await custom.append(
        session_id="s", actor="a", action="x", payload={"v": "my pony"}
    )
    assert entry.payload["v"] == "my ***"
    assert redact_payload({"k": [f"a {KEY_A}", 1]}) == {"k": ["a [REDACTED]", 1]}
    with pytest.raises(TypeError):
        InMemoryAuditLog(secret="k" * 32, redact="yes")  # type: ignore[arg-type]


async def test_agent_run_audit_log_has_no_raw_keys() -> None:
    """End to end: keys in the prompt, tool args and tool result."""

    @tool
    async def call_api(url: str, header: str) -> str:
        "call an api"
        return f"ok token={KEY_B}"

    audit = InMemoryAuditLog(secret="k" * 32)
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        tool="call_api",
                        args={"url": "https://x", "header": f"X-Key: {KEY_A}"},
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model, tools=[call_api], audit_log=audit)
    await agent.run(f"go, my key is {KEY_C}", user_id="u")
    entries = await audit.query()
    assert entries
    blob = json.dumps([e.payload for e in entries], default=str)
    for key in (KEY_A, KEY_B, KEY_C):
        assert key not in blob
    assert "[REDACTED]" in blob
    assert (await audit.verify()).ok


# ---------------------------------------------------------------------------
# 7. MCP: destructiveHint defaults to true
# ---------------------------------------------------------------------------


@dataclass
class _FakeMcpTool:
    name: str
    description: str = ""
    inputSchema: dict[str, Any] = field(default_factory=dict)
    annotations: Any = None


class _FakeMcpSession:
    def __init__(self, tools: list[_FakeMcpTool]) -> None:
        self._tools = tools

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=list(self._tools))


def _ann(**hints: Any) -> SimpleNamespace:
    return SimpleNamespace(
        destructiveHint=hints.get("destructiveHint"),
        readOnlyHint=hints.get("readOnlyHint"),
    )


def test_mcp_spec_default_for_destructive_hint_is_true() -> None:
    """Pin the spec text this behaviour follows (skipped without the SDK)."""
    types_mod = pytest.importorskip("mcp.types")
    import inspect
    import re

    # mcp 1.x spells the fields ``destructiveHint``, 2.x ``destructive_hint``.
    text = inspect.getsource(types_mod.ToolAnnotations)
    parts = re.split(r"destructive_?[hH]int", text, maxsplit=1)
    assert len(parts) == 2, text
    after = re.split(r"idempotent_?[hH]int", parts[1], maxsplit=1)[0]
    assert "Default: true" in after, text


@pytest.mark.parametrize(
    ("annotations", "expected"),
    [
        (None, True),  # no annotations at all
        (_ann(), True),  # annotations present, hints unset
        ({}, True),  # mapping form, empty
        (_ann(readOnlyHint=False), True),
        (_ann(readOnlyHint=True), False),
        ({"readOnlyHint": True}, False),
        (_ann(destructiveHint=False), False),  # explicit: additive only
        (_ann(destructiveHint=False, readOnlyHint=False), False),
        (_ann(destructiveHint=True), True),
        (_ann(destructiveHint=True, readOnlyHint=True), True),
    ],
)
def test_mcp_annotations_follow_spec_defaults(annotations: Any, expected: bool) -> None:
    t = SimpleNamespace(name="drop_database", annotations=annotations)
    assert _annotated_destructive(t) is expected


async def test_mcp_per_server_override() -> None:
    tools = [
        _FakeMcpTool(name="search"),
        _FakeMcpTool(name="write"),
        _FakeMcpTool(name="peek", annotations=_ann(readOnlyHint=True)),
    ]

    async def defs(spec: MCPServerSpec) -> dict[str, bool]:
        reg = MCPRegistry([MCPClient(spec, session=_FakeMcpSession(tools))])
        return {d.name: d.destructive for d in await reg.list_tools()}

    # Default: un-annotated tools need approval.
    assert await defs(MCPServerSpec.stdio("s", "noop")) == {
        "search": True, "write": True, "peek": False,
    }
    # A trusted server: everything safe.
    assert await defs(MCPServerSpec.stdio("s", "noop", destructive=False)) == {
        "search": False, "write": False, "peek": False,
    }
    # Only named tools safe.
    assert await defs(MCPServerSpec.http("s", "http://127.0.0.1:1", safe_tools=["search"])) == {
        "search": False, "write": True, "peek": False,
    }
    # Distrust the server's own hints entirely.
    assert await defs(MCPServerSpec.stdio("s", "noop", destructive=True)) == {
        "search": True, "write": True, "peek": True,
    }


# ---------------------------------------------------------------------------
# 8. bash_tool: the timeout bounds the whole process tree
# ---------------------------------------------------------------------------

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")


@posix_only
@pytest.mark.parametrize(
    "command", ["sleep 6", "sleep 6; echo x", "sleep 6 | cat", "sleep 6 & wait"]
)
async def test_bash_timeout_bounds_execution(tmp_path: Path, command: str) -> None:
    bash = bash_tool(tmp_path, timeout=0.5)
    started = time.monotonic()
    with anyio.fail_after(5):
        out = await bash.fn(command=command)
    assert time.monotonic() - started < 3.0
    assert "timed out after 0.5s" in out


@posix_only
async def test_bash_timeout_kills_backgrounded_children(tmp_path: Path) -> None:
    marker = tmp_path / "orphan.txt"
    bash = bash_tool(tmp_path, timeout=0.3)
    with anyio.fail_after(5):
        out = await bash.fn(command=f"(sleep 1; echo alive > {marker}) & wait")
        assert "timed out" in out
        # Detached-from-the-shell child that still holds the pipes.
        out = await bash.fn(command=f"(sleep 1; echo alive > {marker}) &")
        assert "timed out" in out
    await anyio.sleep(1.4)
    assert not marker.exists(), "a child outlived the killed command"


@posix_only
async def test_bash_cancellation_kills_the_process_tree(tmp_path: Path) -> None:
    marker = tmp_path / "cancelled.txt"
    bash = bash_tool(tmp_path, timeout=30)
    with anyio.move_on_after(0.3):
        await bash.fn(command=f"sleep 1; echo alive > {marker}")
    await anyio.sleep(1.2)
    assert not marker.exists()


@posix_only
async def test_bash_model_timeout_cannot_exceed_operator_timeout(tmp_path: Path) -> None:
    bash = bash_tool(tmp_path, timeout=0.5)
    for requested in (30, 1e9, float("inf"), float("nan"), -1, 0, "30"):
        started = time.monotonic()
        with anyio.fail_after(5):
            out = await bash.fn(command="sleep 2; echo survived", timeout_sec=requested)
        assert time.monotonic() - started < 1.8, requested
        assert "timed out after 0.5s" in out and "--- stdout ---" not in out
    # The model may still ask for LESS time.
    generous = bash_tool(tmp_path, timeout=20)
    started = time.monotonic()
    out = await generous.fn(command="sleep 5", timeout_sec=0.3)
    assert time.monotonic() - started < 3.0
    assert "timed out after 0.3s" in out


@posix_only
async def test_bash_normal_commands_unchanged(tmp_path: Path) -> None:
    bash = bash_tool(tmp_path, timeout=10)
    out = await bash.fn(command="echo hello; echo oops 1>&2; exit 3")
    assert "[exit=3]" in out and "hello" in out and "oops" in out
    assert "(no output)" in await bash.fn(command="true")
    assert tmp_path.name in await bash.fn(command="pwd -P")
    # stdin is not inherited: a command that reads it sees EOF.
    with anyio.fail_after(5):
        assert "[exit=0]" in await bash.fn(command="cat")


# ---------------------------------------------------------------------------
# 10. FilesystemSandbox around a shell tool
# ---------------------------------------------------------------------------


@posix_only
async def test_filesystem_sandbox_warns_that_it_does_not_contain_bash(
    tmp_path: Path,
) -> None:
    host = InProcessToolHost([bash_tool(tmp_path), read_tool(tmp_path)])
    sandbox = FilesystemSandbox(host, roots=[tmp_path])
    with pytest.warns(UserWarning, match="does not contain the shell tool 'bash'"):
        result = await sandbox.call("bash", {"command": "echo hi"}, call_id="1")
    assert result.ok  # default is warn, not refuse
    # Once per tool, and never for a plain file tool.
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await sandbox.call("bash", {"command": "echo again"}, call_id="2")
        read = await sandbox.call("read", {"path": "a.txt"}, call_id="3")
    assert read.ok


@posix_only
async def test_filesystem_sandbox_recognises_a_renamed_bash_tool(
    tmp_path: Path,
) -> None:
    host = InProcessToolHost([bash_tool(tmp_path, name="run_it")])
    sandbox = FilesystemSandbox(host, roots=[tmp_path], shell_tools="deny")
    result = await sandbox.call("run_it", {"command": "head -1 /etc/passwd"}, call_id="1")
    assert not result.ok
    assert result.denied and "cannot contain" in (result.reason or "")


async def test_filesystem_sandbox_shell_policy_options(tmp_path: Path) -> None:
    class _Host:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def list_tools(self, *, query: str | None = None) -> list[Any]:
            return []

        async def call(self, tool: str, args: Any, *, call_id: str = "") -> Any:
            self.calls.append(tool)
            from loomflow.core.types import ToolResult

            return ToolResult.success(call_id, "ran")

    host = _Host()
    denied = await FilesystemSandbox(host, roots=[tmp_path], shell_tools="deny").call(
        "shell", {"cmd": "ls"}, call_id="1"
    )
    assert not denied.ok and host.calls == []

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "allow" is silent
        allowed = await FilesystemSandbox(
            host, roots=[tmp_path], shell_tools="allow"
        ).call("shell", {"cmd": "ls"}, call_id="2")
    assert allowed.ok and host.calls == ["shell"]

    with pytest.raises(ValueError, match="shell_tools"):
        FilesystemSandbox(host, roots=[tmp_path], shell_tools="maybe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 11. guardrails: unknown verdict actions fail closed
# ---------------------------------------------------------------------------


class _Guard:
    stages = frozenset({"input", "output", "tool_result"})

    def __init__(self, name: str, action: Any, **kw: Any) -> None:
        self.name = name
        self._verdict = GuardVerdict(action=action, **kw)
        self.calls = 0

    async def check(self, text: str, *, stage: str, context: Any) -> GuardVerdict:
        self.calls += 1
        return self._verdict


@pytest.mark.parametrize("action", ["deny", "blocked", "BLOCK", "reject", "", None, 0])
async def test_unknown_guard_action_fails_closed(action: Any) -> None:
    bad = _Guard("typo-guard", action)
    later = _Guard("later", "allow")
    outcome = await apply_guardrails([bad, later], "text", stage="input", context=None)
    assert outcome.blocked is True
    assert outcome.guard == "typo-guard"
    assert "unknown verdict action" in (outcome.reason or "")
    assert repr(action) in (outcome.reason or "")
    assert [t.action for t in outcome.triggered] == ["block"]
    assert later.calls == 0  # short-circuits like a real block


async def test_known_guard_actions_unchanged() -> None:
    guards = [
        _Guard("a", "allow"),
        _Guard("b", "annotate", transformed="wrapped", reason="flagged"),
        _Guard("c", "annotate"),
    ]
    outcome = await apply_guardrails(guards, "text", stage="output", context=None)
    assert outcome.blocked is False and outcome.text == "wrapped"
    assert [t.guard for t in outcome.triggered] == ["b"]
    blocked = await apply_guardrails(
        [_Guard("x", "block", reason="no")], "text", stage="output", context=None
    )
    assert blocked.blocked and blocked.reason == "no"
