"""SubprocessSandbox tests — process isolation, timeout, error propagation.

Subprocess spawn time on macOS is ~150-300ms, so these tests are
slower than the rest of the suite. Kept short and few.
"""

from __future__ import annotations

import os

import pytest

from jeevesagent import (
    Agent,
    InProcessToolHost,
    ScriptedModel,
    ScriptedTurn,
    SubprocessSandbox,
    Tool,
)
from jeevesagent.core.errors import ConfigError
from jeevesagent.core.types import ToolCall

from . import _subprocess_tools as tools

pytestmark = pytest.mark.anyio


def _make_host(*tool_specs: tuple[str, object, str]) -> InProcessToolHost:
    """Build a host registering each ``(name, fn, description)`` spec."""
    host = InProcessToolHost()
    for name, fn, description in tool_specs:
        host.register(
            Tool(
                name=name,
                description=description,
                fn=fn,
                input_schema={"type": "object"},
            )
        )
    return host


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_non_inprocess_host() -> None:
    """The sandbox needs ``InProcessToolHost.get`` to retrieve the
    callable; can't sandbox arbitrary ToolHost implementations."""

    class _BareHost:
        async def list_tools(self, *, query=None):  # type: ignore[no-untyped-def]
            return []

        async def call(self, tool, args, *, call_id=""):  # type: ignore[no-untyped-def]
            from jeevesagent.core.types import ToolResult

            return ToolResult.error_(call_id, "noop")

        async def watch(self):  # type: ignore[no-untyped-def]
            empty: tuple = ()
            for ev in empty:
                yield ev

    with pytest.raises(ConfigError, match="InProcessToolHost"):
        SubprocessSandbox(_BareHost())  # type: ignore[arg-type]


def test_rejects_zero_or_negative_timeout() -> None:
    host = _make_host(("double", tools.double, "double"))
    with pytest.raises(ValueError):
        SubprocessSandbox(host, timeout_seconds=0)
    with pytest.raises(ValueError):
        SubprocessSandbox(host, timeout_seconds=-1.0)


# ---------------------------------------------------------------------------
# Roundtrip: tool runs in subprocess, result returns
# ---------------------------------------------------------------------------


async def test_tool_runs_in_subprocess_and_returns_result() -> None:
    host = _make_host(("double", tools.double, "double a number"))
    sandbox = SubprocessSandbox(host, timeout_seconds=15.0)
    result = await sandbox.call("double", {"x": 21}, call_id="c1")
    assert result.ok
    assert result.output == 42
    assert result.call_id == "c1"


async def test_async_tool_runs_via_asyncio_run_in_subprocess() -> None:
    host = _make_host(
        ("async_double", tools.async_double, "async double")
    )
    sandbox = SubprocessSandbox(host, timeout_seconds=15.0)
    result = await sandbox.call("async_double", {"x": 5}, call_id="c1")
    assert result.ok
    assert result.output == 10


async def test_subprocess_pid_differs_from_parent() -> None:
    """Sanity check: the tool really did execute in a different process."""
    host = _make_host(("get_pid", tools.get_pid, "return pid"))
    sandbox = SubprocessSandbox(host, timeout_seconds=15.0)
    result = await sandbox.call("get_pid", {}, call_id="c1")
    assert result.ok
    assert isinstance(result.output, int)
    assert result.output != os.getpid()  # different process


# ---------------------------------------------------------------------------
# Error / timeout paths
# ---------------------------------------------------------------------------


async def test_tool_exception_returns_error_result() -> None:
    host = _make_host(("crashy", tools.crashy_tool, "always fails"))
    sandbox = SubprocessSandbox(host, timeout_seconds=15.0)
    result = await sandbox.call(
        "crashy", {"message": "from-test"}, call_id="c1"
    )
    assert not result.ok
    assert result.error is not None
    assert "from-test" in result.error
    assert "ValueError" in result.error


async def test_timeout_kills_subprocess_and_returns_error() -> None:
    host = _make_host(("slow", tools.slow_tool, "sleep then return"))
    sandbox = SubprocessSandbox(host, timeout_seconds=0.5)
    result = await sandbox.call(
        "slow", {"delay_seconds": 5.0}, call_id="c1"
    )
    assert not result.ok
    assert result.error is not None
    assert "0.5s" in result.error or "exceeded" in result.error


async def test_unknown_tool_returns_error() -> None:
    host = _make_host(("double", tools.double, "double"))
    sandbox = SubprocessSandbox(host, timeout_seconds=15.0)
    result = await sandbox.call("ghost", {}, call_id="c1")
    assert not result.ok
    assert result.error is not None
    assert "unknown tool" in result.error


# ---------------------------------------------------------------------------
# Pass-through behaviour
# ---------------------------------------------------------------------------


async def test_list_tools_passes_through() -> None:
    host = _make_host(("double", tools.double, "double a number"))
    sandbox = SubprocessSandbox(host, timeout_seconds=15.0)
    defs = await sandbox.list_tools()
    assert {d.name for d in defs} == {"double"}


# ---------------------------------------------------------------------------
# End-to-end inside an Agent loop
# ---------------------------------------------------------------------------


async def test_agent_dispatches_tool_through_subprocess_sandbox() -> None:
    host = _make_host(("double", tools.double, "double"))
    sandbox = SubprocessSandbox(host, timeout_seconds=15.0)

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="double", args={"x": 7})
                ]
            ),
            ScriptedTurn(text="14, of course."),
        ]
    )
    agent = Agent("hi", model=model, tools=sandbox)
    result = await agent.run("double 7")
    assert "14" in result.output
