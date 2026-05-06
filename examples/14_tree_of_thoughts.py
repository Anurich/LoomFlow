"""14_tree_of_thoughts — Branching exploration with per-branch evaluation.

What it shows:
* The ``TreeOfThoughts`` architecture explores multiple candidate
  reasoning paths in parallel rather than committing to the first
  one. At each level, the proposer generates ``branch_factor``
  candidates and the evaluator scores each. The top ``beam_width``
  by score survive to the next level.
* Useful for combinatorial / planning / math tasks where ReAct's
  greedy approach commits too early.
* Early-exit on ``solved_threshold`` — if a branch scores high
  enough, search stops and that branch wins.
* Architecture progress events surface each ``proposed`` /
  ``evaluated`` / ``pruned`` step so the search tree is observable.

We use ``ScriptedModel`` for determinism. The toy task: pick the
best opening move for a hypothetical chess problem. Round 1 explores
three candidate moves; the evaluator scores them; we either find a
winning move (≥ ``solved_threshold``) or expand further.

Run:
    python examples/14_tree_of_thoughts.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    ScriptedModel,
    ScriptedTurn,
    TreeOfThoughts,
)


async def main() -> None:
    # branch_factor=3, max_depth=2, beam_width=1.
    # Level 1 proposes 3 candidates; evaluator scores each; top 1 survives.
    # Level 2 proposes 3 from that survivor; evaluator scores each.
    # solved_threshold=0.95 means we stop early if anything scores >= 0.95.
    # Total worst-case calls: 3 propose + 3 eval (level 1) + 3 + 3 (level 2) = 12.
    model = ScriptedModel(
        [
            # Level 1: three candidate openings
            ScriptedTurn(text="Consider e4 (King's Pawn opening)."),
            ScriptedTurn(text="Consider d4 (Queen's Pawn opening)."),
            ScriptedTurn(text="Consider Nf3 (Réti opening)."),
            # Level 1: evaluations
            ScriptedTurn(
                text="score: 0.7\nClassic, leads to many strong lines."
            ),
            ScriptedTurn(
                text="score: 0.65\nSolid but more positional."
            ),
            ScriptedTurn(
                text="score: 0.5\nFlexible but slow."
            ),
            # Level 2: from e4 (survivor); three follow-ups
            ScriptedTurn(text="After e4: respond to e5 with Nf3 (Italian/Spanish setup)."),
            ScriptedTurn(text="After e4: respond to e5 with Bc4 (Italian Game)."),
            ScriptedTurn(text="After e4: respond to e5 with d4 (Center Game)."),
            # Level 2: evaluations
            ScriptedTurn(
                text="score: 0.97\nMost flexible; classical theory backs it."
            ),
            ScriptedTurn(
                text="score: 0.85\nDirect attack but committed."
            ),
            ScriptedTurn(
                text="score: 0.6\nGives up structure too early."
            ),
        ]
    )

    agent = Agent(
        "You are a chess opening analyst.",
        model=model,
        architecture=TreeOfThoughts(
            branch_factor=3,
            max_depth=2,
            beam_width=1,
            solved_threshold=0.95,
        ),
    )

    print("=== Streaming events ===")
    async for event in agent.stream(
        "What's the strongest opening line for White and how does Black "
        "best respond?"
    ):
        if event.kind != "architecture_event":
            continue
        name = event.payload.get("name", "")
        if name == "tot.level_started":
            depth = event.payload["depth"]
            size = event.payload["frontier_size"]
            print(f"\n[level {depth}] expanding {size} frontier node(s)")
        elif name == "tot.proposed":
            depth = event.payload["depth"]
            content = event.payload["content"]
            print(f"  [proposed @ d{depth}] {content}")
        elif name == "tot.evaluated":
            depth = event.payload["depth"]
            score = event.payload["score"]
            print(f"  [evaluated @ d{depth}] score={score:.2f}")
        elif name == "tot.pruned":
            depth = event.payload["depth"]
            kept_scores = event.payload["kept_scores"]
            print(
                f"  [pruned @ d{depth}] kept {len(kept_scores)}: "
                f"{kept_scores}"
            )
        elif name == "tot.solved":
            depth = event.payload["depth"]
            score = event.payload["score"]
            print(f"\n[solved @ d{depth}, score={score:.2f}]")
        elif name == "tot.completed":
            score = event.payload["winner_score"]
            total = event.payload["total_nodes"]
            print(
                f"\n[completed] winner score={score:.2f} "
                f"(explored {total} thoughts)"
            )

    # Re-run with fresh model for the final answer print.
    fresh = ScriptedModel(
        [
            ScriptedTurn(text="e4"),
            ScriptedTurn(text="d4"),
            ScriptedTurn(text="Nf3"),
            ScriptedTurn(text="score: 0.7"),
            ScriptedTurn(text="score: 0.65"),
            ScriptedTurn(text="score: 0.5"),
            ScriptedTurn(text="After e4 e5: Nf3"),
            ScriptedTurn(text="After e4 e5: Bc4"),
            ScriptedTurn(text="After e4 e5: d4"),
            ScriptedTurn(text="score: 0.97"),
            ScriptedTurn(text="score: 0.85"),
            ScriptedTurn(text="score: 0.6"),
        ]
    )
    fresh_agent = Agent(
        "chess analyst",
        model=fresh,
        architecture=TreeOfThoughts(
            branch_factor=3,
            max_depth=2,
            beam_width=1,
            solved_threshold=0.95,
        ),
    )
    result = await fresh_agent.run("strongest opening?")
    print(f"\n=== Final answer ===\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
