"""Tool dispatch end-to-end: registration, parallel fan-out, permissions, hooks."""

from __future__ import annotations

import time

import anyio
import pytest

from jeevesagent import (
    Agent,
    HookRegistry,
    Mode,
    StandardPermissions,
    Tool,
    tool,
)
from jeevesagent.core.types import PermissionDecision, ToolCall
from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


def _scripted(*turns: ScriptedTurn) -> ScriptedModel:
    return ScriptedModel(list(turns))


# ---------------------------------------------------------------------------
# Basic dispatch
# ---------------------------------------------------------------------------


async def test_tool_call_executes_and_result_appears_in_next_turn() -> None:
    @tool
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="add", args={"a": 2, "b": 3})]
        ),
        ScriptedTurn(text="The answer is 5."),
    )
    agent = Agent("calculator", model=model, tools=[add])

    result = await agent.run("what is 2 + 3?")

    assert result.turns == 2
    assert "answer is 5" in result.output
    assert not result.interrupted


async def test_sync_function_is_dispatched_to_thread() -> None:
    """A non-async tool runs via anyio.to_thread.run_sync without blocking."""

    @tool
    def upper(s: str) -> str:
        """Uppercase a string."""
        return s.upper()

    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="upper", args={"s": "hi"})]
        ),
        ScriptedTurn(text="done"),
    )
    agent = Agent("test", model=model, tools=[upper])
    result = await agent.run("uppercase hi")
    assert "done" in result.output


async def test_tool_decorator_with_explicit_metadata() -> None:
    @tool(name="fetch_url", description="Fetch a URL.", destructive=False)
    async def _fetch(url: str) -> str:
        return f"fetched:{url}"

    assert _fetch.name == "fetch_url"
    assert _fetch.description == "Fetch a URL."
    schema = _fetch.input_schema
    assert schema["properties"]["url"]["type"] == "string"
    assert "url" in schema["required"]


async def test_explicit_tool_object_is_accepted() -> None:
    async def echo(msg: str) -> str:
        return msg

    t = Tool(
        name="echo",
        description="echo back",
        fn=echo,
        input_schema={"type": "object"},
    )
    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="echo", args={"msg": "hi"})]
        ),
        ScriptedTurn(text="done"),
    )
    agent = Agent("test", model=model, tools=[t])
    result = await agent.run("test")
    assert "done" in result.output


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------


async def test_parallel_tool_calls_run_concurrently() -> None:
    """Two slow tools in one turn finish in ~one tool's latency, not two."""
    SLEEP = 0.10

    @tool
    async def slow_a() -> str:
        """Slow A."""
        await anyio.sleep(SLEEP)
        return "a"

    @tool
    async def slow_b() -> str:
        """Slow B."""
        await anyio.sleep(SLEEP)
        return "b"

    model = _scripted(
        ScriptedTurn(
            tool_calls=[
                ToolCall(id="ca", tool="slow_a", args={}),
                ToolCall(id="cb", tool="slow_b", args={}),
            ]
        ),
        ScriptedTurn(text="done"),
    )
    agent = Agent("test", model=model, tools=[slow_a, slow_b])

    started = time.monotonic()
    result = await agent.run("go")
    elapsed = time.monotonic() - started

    assert result.output == "done"
    # If serial, elapsed >= 2 * SLEEP. We allow generous slack to account
    # for scheduler jitter; the key check is "well below 2*SLEEP".
    assert elapsed < SLEEP * 1.8, f"expected concurrent dispatch, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Permissions and hooks
# ---------------------------------------------------------------------------


async def test_permission_deny_short_circuits_tool() -> None:
    @tool
    async def dangerous() -> str:
        """Should never run."""
        raise AssertionError("tool should have been denied")

    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="dangerous", args={})]
        ),
        ScriptedTurn(text="ok handled"),
    )
    perms = StandardPermissions(denied_tools=["dangerous"])
    agent = Agent("test", model=model, tools=[dangerous], permissions=perms)

    result = await agent.run("try")

    assert "ok handled" in result.output
    assert not result.interrupted


async def test_destructive_default_mode_requires_approval() -> None:
    """With Mode.DEFAULT, destructive calls become 'ask' and resolve to deny."""

    @tool(destructive=True)
    async def delete_all() -> str:
        """Destructive."""
        raise AssertionError("must not run")

    model = _scripted(
        ScriptedTurn(
            tool_calls=[
                ToolCall(
                    id="c1",
                    tool="delete_all",
                    args={},
                    destructive=True,
                )
            ]
        ),
        ScriptedTurn(text="acknowledged denial"),
    )
    perms = StandardPermissions(mode=Mode.DEFAULT)
    agent = Agent("test", model=model, tools=[delete_all], permissions=perms)

    result = await agent.run("delete")
    assert "acknowledged denial" in result.output


async def test_bypass_mode_allows_destructive_calls() -> None:
    """Mode.BYPASS skips the destructive-call gate."""

    @tool(destructive=True)
    async def risky() -> str:
        return "ran"

    model = _scripted(
        ScriptedTurn(
            tool_calls=[
                ToolCall(id="c1", tool="risky", args={}, destructive=True)
            ]
        ),
        ScriptedTurn(text="finished"),
    )
    perms = StandardPermissions(mode=Mode.BYPASS)
    agent = Agent("test", model=model, tools=[risky], permissions=perms)
    result = await agent.run("...")
    assert "finished" in result.output


async def test_before_tool_hook_can_deny() -> None:
    @tool
    async def whatever() -> str:
        raise AssertionError("hook must deny first")

    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="whatever", args={})]
        ),
        ScriptedTurn(text="all good"),
    )
    agent = Agent("test", model=model, tools=[whatever])

    @agent.before_tool
    async def reject(call: ToolCall) -> PermissionDecision | None:
        return PermissionDecision.deny_("hook says no")

    result = await agent.run("...")
    assert "all good" in result.output


async def test_after_tool_hook_observes_results() -> None:
    seen: list[tuple[str, str]] = []

    @tool
    async def ping() -> str:
        return "pong"

    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="ping", args={})]
        ),
        ScriptedTurn(text="acknowledged"),
    )
    agent = Agent("test", model=model, tools=[ping])

    @agent.after_tool
    async def observe(call: ToolCall, result) -> None:
        seen.append((call.tool, str(result.output)))

    await agent.run("ping?")
    assert seen == [("ping", "pong")]


async def test_buggy_post_hook_does_not_break_loop() -> None:
    @tool
    async def ping() -> str:
        return "pong"

    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="ping", args={})]
        ),
        ScriptedTurn(text="ok"),
    )
    agent = Agent("test", model=model, tools=[ping])

    @agent.after_tool
    async def explode(call: ToolCall, result) -> None:
        raise RuntimeError("boom")

    result = await agent.run("ping?")
    assert "ok" in result.output


# ---------------------------------------------------------------------------
# Loop bounds
# ---------------------------------------------------------------------------


async def test_max_turns_caps_runaway_loop() -> None:
    """A model that never stops emitting tool calls is cut off at max_turns."""

    @tool
    async def noop() -> str:
        return "ack"

    forever = [
        ScriptedTurn(tool_calls=[ToolCall(tool="noop", args={})])
        for _ in range(100)
    ]
    model = ScriptedModel(forever)
    agent = Agent("loop", model=model, tools=[noop], max_turns=3)

    result = await agent.run("go")

    assert result.interrupted
    assert result.interruption_reason == "max_turns_exceeded"
    assert result.turns == 3


async def test_unknown_tool_returns_error_result_and_loop_continues() -> None:
    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="ghost", args={})]
        ),
        ScriptedTurn(text="recovered"),
    )
    agent = Agent("test", model=model)  # no tools registered
    result = await agent.run("...")
    assert "recovered" in result.output


# ---------------------------------------------------------------------------
# Hook registry direct construction
# ---------------------------------------------------------------------------


async def test_external_hook_registry_is_used() -> None:
    @tool
    async def echo(msg: str) -> str:
        return msg

    seen_pre: list[str] = []
    hooks = HookRegistry()

    async def watch(call: ToolCall) -> PermissionDecision | None:
        seen_pre.append(call.tool)
        return None

    hooks.register_pre_tool(watch)

    model = _scripted(
        ScriptedTurn(
            tool_calls=[ToolCall(id="c1", tool="echo", args={"msg": "hi"})]
        ),
        ScriptedTurn(text="done"),
    )
    agent = Agent("test", model=model, tools=[echo], hooks=hooks)

    await agent.run("...")

    assert seen_pre == ["echo"]
