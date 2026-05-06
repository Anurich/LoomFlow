"""17_blackboard — Coordinator-led data analysis.

What it shows:
* BlackboardArchitecture has agents that share a public state
  board. A coordinator (LLM) reads the board each round and
  decides who contributes next; an optional decider synthesizes
  the final answer.
* Real-world use: exploratory analysis where the decomposition
  isn't known upfront — data discovery, research, root-cause
  investigation. Particularly strong for problems where you don't
  know which specialist's view will be needed when.
* Demonstrates: round-by-round contributions, transparent state,
  coordinator-driven turn order.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/17_blackboard.py
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

from jeevesagent import Agent, BlackboardArchitecture  # noqa: E402

hypothesis = Agent(
    instructions=(
        "You propose causal hypotheses for unexplained phenomena. "
        "Read the blackboard; if there's an open question, propose "
        "ONE concrete hypothesis with a brief mechanism. If others "
        "have proposed hypotheses, propose alternatives or refine "
        "existing ones. Be specific; under 4 sentences."
    ),
    model="gpt-4.1-mini",
)

evidence = Agent(
    instructions=(
        "You assess what evidence would support or refute the "
        "hypotheses on the blackboard. List 2-3 specific data "
        "points or queries that would discriminate between the "
        "hypotheses. Under 4 sentences."
    ),
    model="gpt-4.1-mini",
)

critic = Agent(
    instructions=(
        "You critique the analysis on the blackboard. Find weak "
        "claims, missing evidence, alternative explanations not "
        "yet considered. Be specific. Under 4 sentences."
    ),
    model="gpt-4.1-mini",
)

coordinator = Agent(
    instructions=(
        "You coordinate a small research team. Read the "
        "blackboard state and decide who should contribute "
        "next, or whether to terminate.\n\n"
        "Output JSON exactly:\n"
        '{"terminate": <bool>, "next_agent": '
        '<"hypothesis"|"evidence"|"critic"|null>, '
        '"instruction": <str|null>}\n\n'
        "Terminate when the hypotheses, evidence, and critique "
        "together support a clear conclusion. No prose, no "
        "markdown fences."
    ),
    model="gpt-4.1-mini",
)

decider = Agent(
    instructions=(
        "You synthesize the final answer from a multi-agent "
        "blackboard discussion. Give a confident conclusion that "
        "integrates the hypotheses, supporting evidence, and the "
        "critic's caveats. 2-4 sentences."
    ),
    model="gpt-4.1-mini",
)


async def main() -> None:
    agent = Agent(
        "Root-cause analysis coordinator.",
        model="gpt-4.1-mini",
        architecture=BlackboardArchitecture(
            agents={
                "hypothesis": hypothesis,
                "evidence": evidence,
                "critic": critic,
            },
            coordinator=coordinator,
            decider=decider,
            max_rounds=6,
        ),
    )

    prompt = (
        "Our SaaS product saw a 35% drop in daily active users "
        "over the past 14 days. Nothing changed in our pricing, "
        "marketing, or product. What's the most likely cause?"
    )

    print("=" * 70)
    print("BlackboardArchitecture — root-cause analysis")
    print("=" * 70)
    print(f"Problem: {prompt}\n")

    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "blackboard.coordinator_decided":
                round_num = ev.payload.get("round")
                terminate = ev.payload.get("terminate")
                picked = ev.payload.get("next_agent")
                if terminate:
                    print(
                        f"\n\n--- Round {round_num}: "
                        f"coordinator → terminate ---"
                    )
                else:
                    print(
                        f"\n\n--- Round {round_num}: "
                        f"coordinator → {picked} ---"
                    )
            elif name == "blackboard.invoking":
                ag = ev.payload.get("agent")
                print(f"\n[{ag} contributes:]")
            elif name == "blackboard.completed":
                size = ev.payload.get("board_size")
                print(
                    f"\n\n--- ✓ completed; board has {size} entries ---"
                )
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n" + "=" * 70)
            print("FINAL ANALYSIS")
            print("=" * 70)
            print(result.get("output", "(no output)"))


if __name__ == "__main__":
    asyncio.run(main())
