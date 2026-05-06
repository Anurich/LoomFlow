"""01_real_model — Talk to a real LLM.

What it shows:
* String-based model resolver: ``model="claude-opus-4-7"`` →
  AnthropicModel, ``"gpt-4o"`` → OpenAIModel, ``"echo"`` → EchoModel.
* Graceful fallback to ``EchoModel`` when no API key is set.

Set one of the keys to make a real call:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

Run:
    python examples/01_real_model.py
"""

from __future__ import annotations

import asyncio
import os

from jeevesagent import Agent


def _pick_model() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-opus-4-7"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    print("(No ANTHROPIC_API_KEY / OPENAI_API_KEY; falling back to echo.)")
    return "echo"


async def main() -> None:
    model = _pick_model()
    agent = Agent("You are a poet. Reply in two haiku.", model=model)
    result = await agent.run("Write about the rain.")
    print(f"--- {model} ---")
    print(result.output)
    print(f"\nUsed {result.tokens_in + result.tokens_out} tokens, "
          f"${result.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
