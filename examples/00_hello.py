"""00_hello — Smallest possible agent.

What it shows:
* Zero-config: no API keys, no infrastructure.
* The default ``EchoModel`` echoes the prompt; the harness still
  exercises the full loop (memory recall, runtime journaling stub,
  episode persistence, telemetry no-ops).

Run:
    python examples/00_hello.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import Agent


async def main() -> None:
    # ``model="echo"`` is the zero-key fallback — the EchoModel
    # echoes the prompt back so you can verify the loop works
    # without burning tokens. For a real LLM, set an API key and
    # use ``model="claude-opus-4-7"`` or ``model="gpt-4o"``.
    agent = Agent("You are a helpful assistant.", model="echo")
    result = await agent.run("Say hello.")
    print(f"Output:  {result.output}")
    print(f"Turns:   {result.turns}")
    print(f"Tokens:  {result.tokens_in} in / {result.tokens_out} out")
    print(f"Session: {result.session_id}")


if __name__ == "__main__":
    asyncio.run(main())
