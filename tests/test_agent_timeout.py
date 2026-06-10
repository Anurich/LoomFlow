"""Tests for ``Agent(timeout=)`` — the wall-clock run ceiling.

A whole ``run()`` (setup + every model / tool call + teardown) is
wrapped in ``anyio.fail_after(timeout)``. On expiry the run is
cancelled and :class:`~loomflow.RunTimeout` is raised — directly
from ``run()``, and surfaced via the producer task in ``stream()``
(the consumer sees an ERROR event, then the task group re-raises
inside an ``ExceptionGroup``).

``timeout=None`` (the default) is a zero-overhead passthrough;
``timeout <= 0`` is rejected at construction.
"""

from __future__ import annotations

import anyio
import pytest

from loomflow import Agent, RunTimeout, ScriptedModel, ScriptedTurn
from loomflow.core.types import Event, EventKind, ToolCall
from loomflow.tools import tool

pytestmark = pytest.mark.anyio


def _scripted(*turns: ScriptedTurn) -> ScriptedModel:
    return ScriptedModel(list(turns))


@tool
async def slow() -> str:
    """A tool that hangs well past any test timeout."""
    await anyio.sleep(30)
    return "never-returns"


@tool
async def quick() -> str:
    """A fast tool."""
    return "ok"


# ---------------------------------------------------------------------------
# run() — expiry raises RunTimeout
# ---------------------------------------------------------------------------


async def test_run_times_out_on_slow_tool() -> None:
    model = _scripted(
        ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="slow", args={})]),
        ScriptedTurn(text="done"),
    )
    agent = Agent("worker", model=model, tools=[slow], timeout=0.2)

    with pytest.raises(RunTimeout) as exc_info:
        await agent.run("do the slow thing")

    assert exc_info.value.seconds == 0.2


async def test_run_timeout_message_names_the_ceiling() -> None:
    model = _scripted(
        ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="slow", args={})]),
    )
    agent = Agent("worker", model=model, tools=[slow], timeout=0.2)

    with pytest.raises(RunTimeout, match="wall-clock timeout of 0.2s"):
        await agent.run("hang")


# ---------------------------------------------------------------------------
# run() — passthrough on success
# ---------------------------------------------------------------------------


async def test_run_default_timeout_none_completes() -> None:
    """No timeout (the default) → unbounded, fast exchange completes."""
    model = _scripted(ScriptedTurn(text="hello"))
    agent = Agent("worker", model=model)

    result = await agent.run("hi")
    assert "hello" in result.output


async def test_run_generous_timeout_is_passthrough_on_success() -> None:
    """A generous timeout doesn't interfere with a fast run."""
    model = _scripted(
        ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="quick", args={})]),
        ScriptedTurn(text="finished"),
    )
    agent = Agent("worker", model=model, tools=[quick], timeout=30)

    result = await agent.run("be quick")
    assert "finished" in result.output


# ---------------------------------------------------------------------------
# construction-time validation
# ---------------------------------------------------------------------------


def test_timeout_zero_rejected() -> None:
    with pytest.raises(ValueError, match="timeout must be > 0"):
        Agent("worker", model="echo", timeout=0)


def test_timeout_negative_rejected() -> None:
    with pytest.raises(ValueError, match="timeout must be > 0"):
        Agent("worker", model="echo", timeout=-1)


def test_timeout_none_is_default() -> None:
    agent = Agent("worker", model="echo")
    assert agent._timeout is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# stream() — expiry surfaces an ERROR event then re-raises
# ---------------------------------------------------------------------------


async def test_stream_times_out_and_raises_runtimeout() -> None:
    model = _scripted(
        ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="slow", args={})]),
    )
    agent = Agent("worker", model=model, tools=[slow], timeout=0.2)

    events: list[Event] = []
    with pytest.raises(BaseException) as exc_info:  # noqa: PT011, B017
        async for event in agent.stream("hang"):
            events.append(event)

    assert _contains_runtimeout(exc_info.value)


async def test_stream_emits_error_event_before_raising() -> None:
    """The consumer gets a terminal ERROR event carrying the
    RunTimeout before the producer task re-raises it."""
    model = _scripted(
        ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="slow", args={})]),
    )
    agent = Agent("worker", model=model, tools=[slow], timeout=0.2)

    events: list[Event] = []
    with pytest.raises(BaseException):  # noqa: PT011, B017
        async for event in agent.stream("hang"):
            events.append(event)

    error_events = [e for e in events if e.kind is EventKind.ERROR]
    assert error_events, "expected a terminal ERROR event"
    assert error_events[-1].payload["type"] == "RunTimeout"


def _contains_runtimeout(exc: BaseException) -> bool:
    """Walk an ExceptionGroup (or a bare exception) for a RunTimeout.

    ``stream()`` surfaces producer-task failures via a task group,
    so the RunTimeout arrives wrapped in an ``ExceptionGroup`` on
    Python 3.11+. Accept either shape.
    """
    if isinstance(exc, RunTimeout):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_contains_runtimeout(sub) for sub in exc.exceptions)
    return False
