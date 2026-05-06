"""02_tools_parallel — @tool decorator + parallel dispatch.

What it shows:
* The ``@tool`` decorator promotes a Python callable to a typed Tool.
* When the model emits N tool calls in one turn, all N run in parallel
  through an ``anyio.create_task_group``.
* Sync tools are dispatched to a worker thread automatically.

Uses ``ScriptedModel`` so this example has zero LLM dependency — it
deterministically asks for two parallel tool calls then summarizes.

Run:
    python examples/02_tools_parallel.py
"""

from __future__ import annotations

import asyncio
import time

from jeevesagent import Agent, ScriptedModel, ScriptedTurn, tool
from jeevesagent.core.types import ToolCall


@tool
async def fetch_alpha() -> str:
    """Fetch alpha (slow operation)."""
    await asyncio.sleep(0.2)
    return "alpha=42"


@tool
async def fetch_beta() -> str:
    """Fetch beta (slow operation)."""
    await asyncio.sleep(0.2)
    return "beta=99"


async def main() -> None:
    # The scripted model emits two tool calls in turn 1, then a
    # summary in turn 2. Real LLMs would do the same kind of thing
    # given the right prompt.
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="ca", tool="fetch_alpha", args={}),
                    ToolCall(id="cb", tool="fetch_beta", args={}),
                ]
            ),
            ScriptedTurn(text="Both fetched. Summary: alpha=42, beta=99."),
        ]
    )
    agent = Agent(
        "You are a data fetcher.",
        model=model,
        tools=[fetch_alpha, fetch_beta],
    )

    started = time.monotonic()
    result = await agent.run("Fetch both.")
    elapsed = time.monotonic() - started

    print(f"Output:  {result.output}")
    print(f"Turns:   {result.turns}")
    print(f"Elapsed: {elapsed:.3f}s  (would be ~0.4s if serial; ~0.2s parallel)")


if __name__ == "__main__":
    asyncio.run(main())
