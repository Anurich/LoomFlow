"""Agent.resume / session_id round-trip against the journaled runtime.

The bar: when an agent runs once with a given ``session_id`` against a
journaled runtime, a *second* call with the same ``session_id`` (and
the same prompt) returns cached model output and tool results without
re-executing the underlying functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, tool
from loomflow.core.types import EventKind, ToolCall
from loomflow.runtime import InMemoryJournalStore, JournaledRuntime, SqliteRuntime

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# session_id round-trip — explicit
# ---------------------------------------------------------------------------


async def test_run_with_session_id_uses_provided_value() -> None:
    """Passing ``session_id`` propagates into the RunResult."""
    agent = Agent("hi", model="echo")
    result = await agent.run("hello", session_id="custom-sess-1")
    assert result.session_id == "custom-sess-1"


async def test_resume_is_alias_for_run_with_session_id() -> None:
    """With no checkpoint recorded (checkpointing is opt-in via
    ``Tuning(checkpoint=True)``), ``resume`` falls back to the legacy
    behaviour: ``run(prompt, session_id=...)``."""
    agent = Agent("hi", model="echo")
    via_resume = await agent.resume("hello", session_id="custom-sess-2")
    assert via_resume.session_id == "custom-sess-2"


async def test_default_session_id_is_auto_generated() -> None:
    agent = Agent("hi", model="echo")
    r1 = await agent.run("hello")
    r2 = await agent.run("hello")
    assert r1.session_id != r2.session_id  # auto-generated, distinct


# ---------------------------------------------------------------------------
# Journal-backed replay against the same session_id
# ---------------------------------------------------------------------------


async def test_resume_replays_model_call_from_journal() -> None:
    """Running twice with the same session_id and journal returns the
    cached model output without re-running the underlying generator."""

    class CountingModel:
        name = "counting"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            yield {"kind": "text", "text": f"call-{self.calls}"}
            yield {"kind": "finish", "finish_reason": "stop", "usage": None}

    # Use ScriptedModel for the actual agent loop; CountingModel above
    # is for documentation only. The real verification: ScriptedModel
    # exhausts its turns after one call, so the *second* run with the
    # same session_id must replay rather than re-stream.
    model = ScriptedModel(
        [ScriptedTurn(text="The answer is 42.")]
    )

    runtime = JournaledRuntime(InMemoryJournalStore())
    agent = Agent("hi", model=model, runtime=runtime)

    r1 = await agent.run("what is the answer", session_id="fixed")
    assert r1.output == "The answer is 42."
    assert model.remaining == 0  # script consumed

    # Simulate a process restart: fresh Agent (fresh default memory)
    # sharing the same journaled runtime. Journal keys now fingerprint
    # the model call's inputs, so replay requires the seed messages to
    # match the recorded run — which they do across a restart, but NOT
    # when the first run's completed episode is still live in the same
    # agent's memory (rehydration would prepend the prior turn, making
    # the model input genuinely different — a conversation
    # continuation, not a crash-resume).
    agent_b = Agent("hi", model=model, runtime=runtime)
    r2 = await agent_b.resume("what is the answer", session_id="fixed")
    assert r2.output == "The answer is 42."
    assert model.remaining == 0  # still consumed; nothing new asked


async def test_resume_replays_tool_call_from_journal() -> None:
    """Tool calls are journaled; second run reuses cached results."""

    call_log: list[str] = []

    @tool
    async def expensive(label: str) -> str:
        """Pretend to do work."""
        call_log.append(label)
        return f"result-for-{label}"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="expensive", args={"label": "x"})
                ]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    runtime = JournaledRuntime(InMemoryJournalStore())
    agent = Agent("hi", model=model, tools=[expensive], runtime=runtime)

    r1 = await agent.run("do the thing", session_id="fixed-tool")
    assert "ok" in r1.output
    assert call_log == ["x"]

    # The model script is exhausted; without replay, the second run
    # would emit nothing useful. With replay, the journaled model
    # chunks AND the journaled tool result both come from cache.
    # Fresh Agent = process-restart simulation: journal keys carry an
    # input fingerprint, so replay needs identical seed messages —
    # true after a restart, but not with the same live agent whose
    # memory now rehydrates the completed first turn into the prompt.
    agent_b = Agent("hi", model=model, tools=[expensive], runtime=runtime)
    r2 = await agent_b.resume("do the thing", session_id="fixed-tool")
    assert r2.output == r1.output
    # Tool function NEVER ran a second time.
    assert call_log == ["x"]


# ---------------------------------------------------------------------------
# Cross-instance replay via SqliteRuntime
# ---------------------------------------------------------------------------


async def test_resume_against_fresh_sqlite_runtime(tmp_path: Path) -> None:
    """A new SqliteRuntime instance against the same DB file replays
    cached steps when given the same session_id."""

    counter = {"runs": 0}

    @tool
    async def expensive() -> str:
        counter["runs"] += 1
        return f"v{counter['runs']}"

    db = tmp_path / "journal.db"

    # First instance writes the journal.
    model_a = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="expensive", args={})]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    rt_a = SqliteRuntime(db)
    agent_a = Agent("hi", model=model_a, tools=[expensive], runtime=rt_a)
    r_a = await agent_a.run("go", session_id="resumable")
    assert "done" in r_a.output
    assert counter["runs"] == 1

    # Simulate process restart: brand-new model, brand-new runtime,
    # same DB, same session_id.
    model_b = ScriptedModel(
        [ScriptedTurn(text="this should not be reached")]
    )
    rt_b = SqliteRuntime(db)
    agent_b = Agent("hi", model=model_b, tools=[expensive], runtime=rt_b)
    r_b = await agent_b.resume("go", session_id="resumable")

    # Same output (model chunks replayed from journal); tool function
    # NEVER ran again.
    assert r_b.output == r_a.output
    assert counter["runs"] == 1


# ---------------------------------------------------------------------------
# Streaming with explicit session_id
# ---------------------------------------------------------------------------


async def test_stream_with_session_id_emits_events_with_that_id() -> None:
    agent = Agent("hi", model="echo")
    seen_ids: set[str] = set()
    async for event in agent.stream("hello", session_id="streamed-sess"):
        seen_ids.add(event.session_id)
        if event.kind == EventKind.COMPLETED:
            assert event.payload["result"]["session_id"] == "streamed-sess"
    assert seen_ids == {"streamed-sess"}
