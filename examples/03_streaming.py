"""03_streaming — agent.stream() with event handling.

What it shows:
* ``agent.stream()`` yields :class:`Event` objects in real time.
* You can watch every model chunk, every tool dispatch, and every
  result as the loop progresses — perfect for a chat UI.
* Breaking out of the iteration cleanly cancels the producer task.

Run:
    python examples/03_streaming.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import Agent, ScriptedModel, ScriptedTurn, tool
from jeevesagent.core.types import EventKind, ToolCall


@tool
async def lookup(name: str) -> str:
    """Pretend to look up an entity."""
    await asyncio.sleep(0.05)
    return f"{name} → 42"


async def main() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(
                text="Looking up Atlas now. ",
                tool_calls=[ToolCall(id="c1", tool="lookup", args={"name": "Atlas"})],
            ),
            ScriptedTurn(text="The answer is 42."),
        ]
    )
    agent = Agent("you are a lookup bot", model=model, tools=[lookup])

    print("--- streaming events ---")
    async for event in agent.stream("look up Atlas"):
        if event.kind == EventKind.MODEL_CHUNK:
            chunk = event.payload["chunk"]
            if chunk["kind"] == "text" and chunk.get("text"):
                print(f"  TEXT: {chunk['text']!r}")
        elif event.kind == EventKind.TOOL_CALL:
            call = event.payload["call"]
            print(f"  TOOL_CALL: {call['tool']}({call['args']})")
        elif event.kind == EventKind.TOOL_RESULT:
            r = event.payload["result"]
            print(f"  TOOL_RESULT: ok={r['ok']} output={r['output']!r}")
        elif event.kind == EventKind.STARTED:
            print(f"  STARTED: session={event.session_id}")
        elif event.kind == EventKind.COMPLETED:
            print(f"  COMPLETED: turns={event.payload['result']['turns']}")


if __name__ == "__main__":
    asyncio.run(main())
