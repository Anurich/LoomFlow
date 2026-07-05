"""Mid-run steering — ``_loom_steering`` metadata queue.

An application can pass a queue-like object (``pop_all() -> list[str]``)
in run metadata under ``_loom_steering``. The ReAct loop drains it at
the top of every iteration and appends the entries as fresh USER
messages, so guidance typed while the agent is working lands before
the NEXT model call instead of waiting for the whole run to finish.
Mirrors the ``_loom_images`` metadata pattern.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, InMemoryMemory
from loomflow.core.types import Message, Role, ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


class _RecordingModel(ScriptedModel):
    """ScriptedModel that records the messages of every call."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.seen: list[list[Message]] = []

    async def complete(self, messages: list[Message], **kw: Any):
        self.seen.append(list(messages))
        return await super().complete(messages, **kw)

    async def stream(self, messages: list[Message], **kw: Any):
        self.seen.append(list(messages))
        async for chunk in super().stream(messages, **kw):
            yield chunk


class _SteeringQueue:
    def __init__(self) -> None:
        self._items: list[str] = []

    def push(self, text: str) -> None:
        self._items.append(text)

    def pop_all(self) -> list[str]:
        out, self._items = self._items, []
        return out


def _user_texts(messages: list[Message]) -> list[str]:
    return [m.content for m in messages if m.role == Role.USER]


async def test_steering_lands_before_next_model_call() -> None:
    """A steering message pushed WHILE a tool runs (mid-run) must be
    a USER message in the NEXT model call's context."""
    queue = _SteeringQueue()

    def _work() -> str:
        # Simulates the user typing guidance while the agent works.
        queue.push("actually, only summarize the README")
        return "worked"

    model = _RecordingModel([
        ScriptedTurn(tool_calls=[ToolCall(tool="_work", args={})]),
        ScriptedTurn(text="done"),
    ])
    agent = Agent(
        "do the task",
        model=model,
        memory=InMemoryMemory(),
        tools=[_work],
    )

    result = await agent.run(
        "go", metadata={"_loom_steering": queue}
    )
    assert result.output == "done"
    assert len(model.seen) == 2
    # Not present in the first call (pushed during the tool run) …
    assert not any(
        "only summarize" in t for t in _user_texts(model.seen[0])
    )
    # … but injected as a USER message before the second.
    assert any(
        "only summarize" in t for t in _user_texts(model.seen[1])
    )
    # And drained — nothing left in the queue.
    assert queue.pop_all() == []


async def test_steering_absent_is_a_noop() -> None:
    """No ``_loom_steering`` in metadata → identical behavior."""
    model = _RecordingModel([ScriptedTurn(text="plain")])
    agent = Agent("do", model=model, memory=InMemoryMemory())
    result = await agent.run("go")
    assert result.output == "plain"


async def test_steering_event_emitted_on_stream() -> None:
    """Streaming consumers see a ``react.steering_injected`` event so
    UIs can render the injection."""
    queue = _SteeringQueue()
    queue.push("prefer the simple fix")

    model = _RecordingModel([ScriptedTurn(text="ok")])
    agent = Agent("do", model=model, memory=InMemoryMemory())

    names: list[str] = []
    async for event in agent.stream(
        "go", metadata={"_loom_steering": queue}
    ):
        payload = getattr(event, "payload", None) or {}
        name = payload.get("name") or payload.get("event") or ""
        names.append(str(name))
    assert any("steering_injected" in n for n in names)
