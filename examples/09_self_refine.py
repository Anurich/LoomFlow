"""09_self_refine — Iterative copywriting with self-critique.

What it shows:
* SelfRefine wraps any base architecture (default ReAct) with a
  critic + refiner cycle. The same model plays all three roles.
* Real-world use: marketing copy / tweet polishing — the model
  writes a draft, critiques itself, refines until the critic
  emits ``stop_phrase`` (default ``"no issues"``).
* Token-level streaming surfaces the draft and refinements live;
  architecture events show convergence.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/09_self_refine.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "\n  ✗ OPENAI_API_KEY required. "
        "Add OPENAI_API_KEY=sk-... to .env at repo root.\n"
    )

from jeevesagent import Agent, ReAct, SelfRefine  # noqa: E402


async def main() -> None:
    agent = Agent(
        instructions=(
            "You are a senior copywriter. You write tight, "
            "engaging marketing copy. When asked to revise, "
            "address every point of the critique fully."
        ),
        model="gpt-4.1-mini",
        architecture=SelfRefine(
            base=ReAct(),
            max_rounds=2,
            stop_phrase="no issues",
        ),
    )

    prompt = (
        "Write a tweet (under 280 characters) announcing the launch "
        "of JeevesAgent — a model-agnostic, MCP-native, async agent "
        "harness for production teams. Include exactly one emoji, "
        "no hashtags, and a clear value prop in the first line."
    )

    print("=" * 70)
    print("SelfRefine — iterative copywriting (token-streaming live)")
    print("=" * 70)

    in_critique = False
    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "self_refine.round_started":
                role = ev.payload.get("role", "?")
                round_num = ev.payload.get("round", "?")
                in_critique = role == "critic"
                print(
                    f"\n\n--- {role.upper()} (round {round_num}) ---"
                )
            elif name == "self_refine.critique":
                # Optional summary; the streaming model_chunks already
                # showed the full critique text live.
                pass
            elif name == "self_refine.converged":
                print(
                    f"\n\n--- ✓ converged at round "
                    f"{ev.payload.get('round')} ---"
                )
            elif name == "self_refine.max_rounds_reached":
                print(
                    f"\n\n--- max rounds reached "
                    f"({ev.payload.get('rounds')}) ---"
                )
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n\n" + "=" * 70)
            print("FINAL TWEET")
            print("=" * 70)
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}"
            )

    _ = in_critique  # quiet linter; reserved for future use


if __name__ == "__main__":
    asyncio.run(main())
