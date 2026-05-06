"""08_from_config — Declarative agent config via TOML.

What it shows:
* ``Agent.from_config("path/to/agent.toml")`` — load
  instructions / model / max_turns / auto_consolidate / budget from
  a TOML file.
* ``Agent.from_dict(cfg)`` — same shape, when you already have a
  config dict in hand (env vars, Pydantic settings, YAML, HTTP API).
* The ``@agent.with_tool`` decorator for inline tool registration.

Run:
    python examples/08_from_config.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jeevesagent import Agent


async def main() -> None:
    here = Path(__file__).parent

    # Load from a TOML file
    agent = Agent.from_config(here / "agent.toml")
    print("--- agent loaded from TOML ---")
    print(repr(agent))

    # Register a tool inline using @agent.with_tool. The decorator
    # returns the original function so it can still be called normally,
    # while also being available to the agent loop.
    @agent.with_tool
    async def search(query: str) -> str:
        """Search a knowledge base for ``query``."""
        return f"results for {query!r}: [doc-1, doc-2]"

    print("\n--- registered tools ---")
    print(await agent.tools_list())

    # Same agent, slightly different config — built from a dict
    # instead of a file. Useful when env vars / settings classes /
    # YAML / HTTP API supply the config.
    cfg = {
        "instructions": "You are terse.",
        "model": "echo",
        "max_turns": 5,
        "auto_consolidate": False,
        "budget": {
            "max_tokens": 50_000,
            "max_cost_usd": 1.0,
        },
    }
    agent_b = Agent.from_dict(cfg)
    print("\n--- agent loaded from dict ---")
    print(repr(agent_b))

    result = await agent_b.run("hello, dict-loaded agent")
    print(f"\noutput: {result.output}")
    print(f"turns:  {result.turns}, duration: {result.duration.total_seconds():.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
