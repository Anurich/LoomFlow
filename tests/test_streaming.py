"""``Agent.stream()`` event-channel tests.

Verifies:

* event ordering (STARTED first, COMPLETED last)
* model chunks arrive as the model emits them
* TOOL_CALL/TOOL_RESULT pair around each dispatch
* ``stream()`` and ``run()`` produce identical final output
* breaking out of the iteration cleanly cancels the producer
* budget exhaustion surfaces a BUDGET_EXCEEDED event
"""

from __future__ import annotations

import time

import anyio
import pytest

from loomflow import Agent, tool
from loomflow.core.types import EventKind, ToolCall
from loomflow.governance.budget import BudgetConfig, StandardBudget
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Ordering and basic shape
# ---------------------------------------------------------------------------


async def test_stream_starts_with_started_and_ends_with_completed() -> None:
    agent = Agent("hi", model="echo")  # uses EchoModel by default
    events = [e async for e in agent.stream("hello")]

    assert events[0].kind == EventKind.STARTED
    assert events[-1].kind == EventKind.COMPLETED
    # The session_id should be consistent across all events.
    assert {e.session_id for e in events} == {events[0].session_id}


async def test_stream_emits_model_chunks_in_order() -> None:
    agent = Agent("hi", model="echo")
    chunks = [
        e
        async for e in agent.stream("alpha beta gamma")
        if e.kind == EventKind.MODEL_CHUNK
    ]
    # EchoModel splits per-word and emits one finish chunk.
    text_chunks = [
        e.payload["chunk"]
        for e in chunks
        if e.payload["chunk"]["kind"] == "text"
    ]
    assert len(text_chunks) >= 3
    finish_chunks = [
        e.payload["chunk"]
        for e in chunks
        if e.payload["chunk"]["kind"] == "finish"
    ]
    assert len(finish_chunks) == 1


async def test_stream_emits_tool_call_then_result_event_pair() -> None:
    @tool
    async def ping() -> str:
        """Return pong."""
        return "pong"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="ping", args={})]
            ),
            ScriptedTurn(text="all done"),
        ]
    )
    agent = Agent("test", model=model, tools=[ping])
    events = [e async for e in agent.stream("...")]

    kinds = [e.kind for e in events]
    call_idx = kinds.index(EventKind.TOOL_CALL)
    result_idx = kinds.index(EventKind.TOOL_RESULT)
    assert call_idx < result_idx
    # Both reference the same call_id.
    assert events[call_idx].payload["call"]["id"] == "c1"
    assert events[result_idx].payload["result"]["call_id"] == "c1"
    assert events[result_idx].payload["result"]["output"] == "pong"


# ---------------------------------------------------------------------------
# Parity with run()
# ---------------------------------------------------------------------------


async def test_stream_completed_payload_matches_run_output() -> None:
    a1 = Agent("hi", model="echo")
    a2 = Agent("hi", model="echo")

    via_run = await a1.run("hello world")
    streamed = [e async for e in a2.stream("hello world")]
    completed = next(e for e in streamed if e.kind == EventKind.COMPLETED)

    assert completed.payload["result"]["output"] == via_run.output
    assert completed.payload["result"]["turns"] == via_run.turns


# ---------------------------------------------------------------------------
# Backpressure and cancellation
# ---------------------------------------------------------------------------


async def test_consumer_break_cancels_producer_quickly() -> None:
    """Breaking the iteration must stop the loop without leaving tasks running."""

    @tool
    async def slow() -> str:
        await anyio.sleep(2.0)  # would block for 2s if allowed to complete
        return "done"

    model = ScriptedModel(
        [
            ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="slow", args={})]),
            ScriptedTurn(text="never reached"),
        ]
    )
    agent = Agent("test", model=model, tools=[slow])

    started = time.monotonic()
    seen: list[EventKind] = []
    async for event in agent.stream("..."):
        seen.append(event.kind)
        if event.kind == EventKind.TOOL_CALL:
            break  # bail out before the slow tool finishes
    elapsed = time.monotonic() - started

    assert EventKind.TOOL_CALL in seen
    # Cancellation must propagate fast — far below the slow tool's 2s.
    assert elapsed < 0.5, f"producer not cancelled (elapsed {elapsed:.2f}s)"


async def test_budget_exceeded_event_appears_before_completed() -> None:
    budget = StandardBudget(BudgetConfig(max_tokens=0))
    agent = Agent("hi", model="echo", budget=budget)

    events = [e async for e in agent.stream("anything")]
    kinds = [e.kind for e in events]

    assert EventKind.BUDGET_EXCEEDED in kinds
    assert kinds.index(EventKind.BUDGET_EXCEEDED) < kinds.index(EventKind.COMPLETED)
    completed = next(e for e in events if e.kind == EventKind.COMPLETED)
    assert completed.payload["result"]["interrupted"] is True
    assert completed.payload["result"]["interruption_reason"].startswith("budget:")


# ---------------------------------------------------------------------------
# Parallel tool dispatch surfaces both call/result events
# ---------------------------------------------------------------------------


async def test_parallel_tool_calls_each_emit_call_and_result_events() -> None:
    @tool
    async def fast_a() -> str:
        return "a"

    @tool
    async def fast_b() -> str:
        return "b"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="ca", tool="fast_a", args={}),
                    ToolCall(id="cb", tool="fast_b", args={}),
                ]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    agent = Agent("test", model=model, tools=[fast_a, fast_b])
    events = [e async for e in agent.stream("...")]

    call_ids = {
        e.payload["call"]["id"] for e in events if e.kind == EventKind.TOOL_CALL
    }
    result_ids = {
        e.payload["result"]["call_id"]
        for e in events
        if e.kind == EventKind.TOOL_RESULT
    }
    assert call_ids == {"ca", "cb"}
    assert result_ids == {"ca", "cb"}
