"""18_plan_and_execute — Planner → step executor → synthesizer.

What it shows:
* PlanAndExecute commits to a plan upfront, then walks through it
  step by step. One planner call + N step calls + one synthesizer
  call. Cheaper than ReAct on tasks with predictable structure
  (ReAct re-thinks before every action; PnE plans once).
* Real-world use: travel planning, recipe creation, multi-step
  document generation — anywhere the structure is stable and
  ReAct's per-turn rumination wastes tokens.
* The plan is a structured Pydantic object you can log / audit /
  override before execution.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/18_plan_and_execute.py
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

from jeevesagent import Agent, PlanAndExecute  # noqa: E402


async def main() -> None:
    agent = Agent(
        instructions=(
            "You plan and write 3-day trip itineraries. Each step "
            "produces ONE component of the trip; the synthesizer "
            "combines them into the final itinerary."
        ),
        model="gpt-4.1-mini",
        architecture=PlanAndExecute(
            max_steps=6,
            planner_prompt=(
                "You produce a step-by-step plan to write a "
                "complete 3-day trip itinerary. Each step is a "
                "concrete sub-task an LLM can complete in one "
                "response (e.g. 'list 3 must-see neighborhoods', "
                "'recommend a Day 1 morning activity', 'list "
                "transit tips').\n\n"
                "Output ONLY a JSON list of strings. 4-6 steps. "
                "No prose, no markdown fences."
            ),
            synthesizer_prompt=(
                "You compose a polished 3-day trip itinerary by "
                "combining the step outputs. Format as Day 1 / "
                "Day 2 / Day 3 with morning/afternoon/evening "
                "blocks. Be concrete and concise."
            ),
        ),
    )

    prompt = (
        "Plan a 3-day food-and-culture trip to Tokyo for a couple "
        "in their 30s, mid-budget, who like ramen, modern art, and "
        "quiet neighborhoods."
    )

    print("=" * 70)
    print("PlanAndExecute — Tokyo trip planner")
    print("=" * 70)
    print(f"Brief: {prompt}\n")

    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "plan.created":
                steps = ev.payload.get("steps", [])
                print(f"[plan: {len(steps)} step(s)]")
                for i, s in enumerate(steps, 1):
                    print(f"  {i}. {s}")
            elif name == "plan.step_started":
                idx = ev.payload.get("step_index", 0)
                desc = ev.payload.get("description", "")[:80]
                print(f"\n[step {idx + 1}] {desc}")
            elif name == "plan.synthesizer_started":
                print("\n[synthesizing final itinerary...]")
            elif name == "plan.completed":
                print(
                    f"\n--- ✓ completed "
                    f"({ev.payload.get('num_steps')} steps) ---"
                )
        elif kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n\n" + "=" * 70)
            print("FINAL ITINERARY")
            print("=" * 70)
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
