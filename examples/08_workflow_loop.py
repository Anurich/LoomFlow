"""Example 8 — Refinement loop with cycles.

Shape::

    draft → review → judge ─┬─ "ship_it"   → END
                            └─ "needs_work"→ revise → review → judge ...
                                                        ↑              ↓
                                                        └──────────────┘

Cycles are first-class in Loom workflows — useful for
refinement, retry, multi-pass review. ``max_visits_per_node``
caps the loop so a hard-to-please critic can't spin forever.

Three pieces to notice:

* The ``classify``/``judge`` step is **deterministic Python** — it
  parses the score the reviewer Agent produced and returns
  ``"ship_it"`` or ``"needs_work"``. No extra LLM call to "decide
  what to do next."
* ``add_router`` with the sentinel ``END`` as a route value
  terminates the loop cleanly when the gate passes.
* ``max_visits_per_node=4`` is the safety cap. If the critic
  never awards >= 9, the workflow raises with a descriptive error
  naming the looping node. Catch the ``RuntimeError`` to surface
  the latest best-effort draft (the state dict is mutated in
  place across iterations, so partial work survives the raise).

Run::

    OPENAI_API_KEY=sk-... python examples/08_workflow_loop.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ Skipping: OPENAI_API_KEY is not set. "
        "Export it (or add it to .env) to run this example.\n"
    )
    sys.exit(0)


from loomflow import (  # noqa: E402
    END,
    Agent,
    InMemoryAuditLog,
    InMemoryMemory,
    Workflow,
)

MODEL = "gpt-4.1-mini"


async def main() -> None:
    print("\n  Example 8 — Refinement loop with cycles\n")

    drafter = Agent(
        "Write a one-sentence product tagline for the topic the user "
        "provides. Be concise — under 15 words. Reply with only the "
        "tagline, no preamble.",
        model=MODEL,
        memory=InMemoryMemory(),
    )
    reviewer = Agent(
        "You are a strict editor. The user gives you a tagline. "
        "Score it 1-10 for clarity + punch, then on a new line write "
        "a one-sentence critique. Format EXACTLY:\n"
        "  SCORE: <n>\n"
        "  CRITIQUE: <text>",
        model=MODEL,
        memory=InMemoryMemory(),
    )
    revisor = Agent(
        "You receive a tagline and a critique. Produce an improved "
        "tagline addressing the critique. Reply with only the new "
        "tagline, no preamble.",
        model=MODEL,
        memory=InMemoryMemory(),
    )

    # Each step takes the loop's state dict, runs the right agent
    # on the right field, writes back, returns the dict for the
    # next step. Track the iteration count via a closure variable
    # so we can surface it even when the cap fires.
    iteration_count = {"reviews": 0}

    async def draft_step(state: dict[str, str]) -> dict[str, str]:
        result = await drafter.run(state["topic"])
        state["draft"] = result.output.strip()
        return state

    async def review_step(state: dict[str, str]) -> dict[str, str]:
        iteration_count["reviews"] += 1
        result = await reviewer.run(state["draft"])
        state["review"] = result.output.strip()
        return state

    def judge(state: dict[str, str]) -> dict[str, str]:
        """Deterministic: parse the SCORE the reviewer wrote into
        ``last_score``. No LLM call — the LLM already produced the
        score upstream."""
        review = state.get("review", "")
        score = 0
        for line in review.splitlines():
            if line.upper().startswith("SCORE:"):
                try:
                    score = int(line.split(":", 1)[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
                break
        state["last_score"] = str(score)
        return state

    async def revise_step(state: dict[str, str]) -> dict[str, str]:
        prompt = (
            f"Tagline: {state['draft']}\n\n"
            f"Critique:\n{state['review']}"
        )
        result = await revisor.run(prompt)
        state["draft"] = result.output.strip()
        return state

    # The router gates the loop. ``"ship_it"`` when the score is
    # high enough; ``"needs_work"`` otherwise. Pure Python — the
    # LLM already produced the score upstream.
    def route_after_judge(state: dict[str, str]) -> str:
        return "ship_it" if int(state.get("last_score", "0")) >= 9 else "needs_work"

    audit = InMemoryAuditLog()
    wf = Workflow("refinement", max_visits_per_node=4, audit_log=audit)
    wf.add_node("draft", draft_step)
    wf.add_node("review", review_step)
    wf.add_node("judge", judge)
    wf.add_node("revise", revise_step)
    wf.add_edge("draft", "review")
    wf.add_edge("review", "judge")
    wf.add_router(
        "judge",
        route_after_judge,
        {"ship_it": END, "needs_work": "revise"},
    )
    wf.add_edge("revise", "review")  # loop back
    wf.set_start("draft")

    state: dict[str, str] = {"topic": "AI-powered note-taking app for engineers"}

    try:
        result = await wf.run(state, user_id="alice", session_id="loop-demo")
        outcome = "shipped"
        final_draft = result.output["draft"]  # type: ignore[index]
        final_score = result.output.get("last_score", "?")  # type: ignore[union-attr]
        visited = result.visited
    except RuntimeError as exc:
        # Cap-exceeded path. The state dict still has the latest
        # draft + score because each step mutates in place.
        outcome = f"capped ({exc.__class__.__name__})"
        final_draft = state.get("draft", "(no draft)")
        final_score = state.get("last_score", "?")
        visited = ["(see audit log for full trace)"]

    print(f"  topic       : {state['topic']}")
    print(f"  outcome     : {outcome}")
    print(f"  iterations  : {iteration_count['reviews']} review pass(es)")
    print(f"  final score : {final_score}")
    print(f"  final draft : {final_draft}")
    print(f"  visited     : {' → '.join(visited)}")


if __name__ == "__main__":
    asyncio.run(main())
