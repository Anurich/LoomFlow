"""14_tree_of_thoughts — BFS beam search for combinatorial reasoning.

What it shows:
* TreeOfThoughts explores multiple candidate "thoughts" per level
  rather than committing to the first reasoning path. Per-level:
  proposer generates ``branch_factor`` candidates, evaluator scores
  each 0-1, top ``beam_width`` survive to the next level.
* Real-world use: math/planning/puzzle problems where ReAct's
  greedy approach commits too early. The Game of 24 is the
  canonical benchmark.
* Architecture events expose the search tree — every proposal,
  every score, every prune. Easy to visualize.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/14_tree_of_thoughts.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ OPENAI_API_KEY not set — skipping this example.\n"
        "    Add OPENAI_API_KEY=sk-... to .env at repo root to run.\n"
    )
    sys.exit(0)

from jeevesagent import Agent, TreeOfThoughts  # noqa: E402


async def main() -> None:
    agent = Agent(
        instructions=(
            "You solve combinatorial math puzzles step by step. "
            "Think carefully about which operation to try first; "
            "subsequent steps build on prior steps."
        ),
        model="gpt-4.1-mini",
        architecture=TreeOfThoughts(
            branch_factor=3,
            max_depth=3,
            beam_width=2,
            solved_threshold=0.9,
            proposer_prompt=(
                "You are exploring ways to combine numbers using "
                "+, -, ×, ÷ to reach a target. Given the problem "
                "and any prior steps, propose ONE next step (a "
                "specific operation to try). Be concrete: state "
                "the operation and the resulting intermediate "
                "value. One paragraph max."
            ),
            evaluator_prompt=(
                "Score how promising this next step is for "
                "solving the puzzle. Output exactly:\n"
                "score: <0-1>\nbrief reason\n\n"
                "1.0 = this step plus simple follow-ups will "
                "definitely reach the target; 0.0 = wrong direction."
            ),
        ),
    )

    prompt = (
        "Game of 24: using each of the numbers 4, 7, 8, 8 exactly "
        "once with +, -, ×, ÷ (and parentheses), make 24. Show "
        "your reasoning."
    )

    print("=" * 70)
    print("TreeOfThoughts — Game of 24")
    print("=" * 70)
    print(f"Puzzle: {prompt}\n")

    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "tot.level_started":
                depth = ev.payload.get("depth")
                size = ev.payload.get("frontier_size")
                print(
                    f"\n[level {depth}] expanding {size} frontier "
                    f"node(s)"
                )
            elif name == "tot.proposed":
                depth = ev.payload.get("depth")
                content = ev.payload.get("content", "")
                print(f"  [d{depth} propose] {content[:90]}")
            elif name == "tot.evaluated":
                score = ev.payload.get("score", 0.0)
                print(f"    [→ score {score:.2f}]")
            elif name == "tot.pruned":
                kept_scores = ev.payload.get("kept_scores", [])
                print(
                    f"  [d{ev.payload.get('depth')} pruned, kept "
                    f"{len(kept_scores)}: {kept_scores}]"
                )
            elif name == "tot.solved":
                print(
                    f"\n--- ✓ SOLVED at depth "
                    f"{ev.payload.get('depth')} "
                    f"(score {ev.payload.get('score', 0.0):.2f}) ---"
                )
            elif name == "tot.completed":
                print(
                    f"\n--- explored "
                    f"{ev.payload.get('total_nodes')} thoughts ---"
                )
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n" + "=" * 70)
            print("WINNING PATH (highest-scored leaf)")
            print("=" * 70)
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
