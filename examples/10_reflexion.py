"""10_reflexion — Verbal reinforcement learning via memory.

What it shows:
* The ``Reflexion`` architecture wraps a base (default ``ReAct``).
  After each attempt, an evaluator scores the output (0-1). Below
  ``threshold``, a reflector emits a one-sentence "lesson" that
  appends to a memory block.
* The base architecture's ``memory.working()`` recall picks up the
  lesson on the next attempt automatically — zero plumbing needed
  on the base side.
* Cross-session learning: with a persistent memory backend
  (``SqliteMemory`` / ``PostgresMemory`` / ``RedisMemory``) the
  lesson survives process restarts. Future runs benefit too.

We use ``ScriptedModel`` so the example runs deterministically.
Attempt 1 fails, the reflector produces a lesson, Attempt 2 reads
the lesson from memory and succeeds.

Run:
    python examples/10_reflexion.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    InMemoryMemory,
    ReAct,
    Reflexion,
    ScriptedModel,
    ScriptedTurn,
)


async def main() -> None:
    # Six model turns total:
    # Attempt 1:
    #   1. ReAct generator: "rough answer (no dates)".
    #   2. Evaluator: "score: 0.3" (below 0.8 threshold)
    #   3. Reflector: produces a lesson about dates.
    # Attempt 2:
    #   4. ReAct generator: "polished answer with dates".
    #   5. Evaluator: "score: 0.9" — terminate.
    model = ScriptedModel(
        [
            ScriptedTurn(text="Tokyo became Japan's capital."),
            ScriptedTurn(
                text="score: 0.3\nThe answer omits the year."
            ),
            ScriptedTurn(
                text=(
                    "When asked about historical events, always "
                    "include the specific year."
                )
            ),
            ScriptedTurn(
                text="Tokyo became Japan's capital in 1868."
            ),
            ScriptedTurn(text="score: 0.9\nNow includes the year."),
        ]
    )

    memory = InMemoryMemory()  # for cross-run, swap to SqliteMemory
    agent = Agent(
        "You answer history questions about Japan.",
        model=model,
        memory=memory,
        architecture=Reflexion(
            base=ReAct(),
            threshold=0.8,
            max_attempts=3,
        ),
    )

    print("=== Streaming events ===")
    async for event in agent.stream("When did Tokyo become Japan's capital?"):
        if event.kind == "architecture_event":
            name = event.payload.get("name", "")
            if name == "reflexion.evaluated":
                attempt = event.payload["attempt"]
                score = event.payload["score"]
                print(f"\n[attempt {attempt}] evaluator score = {score:.2f}")
            elif name == "reflexion.lesson_produced":
                lesson = event.payload["lesson"]
                print(f"[reflector] lesson: {lesson}")
            elif name == "reflexion.threshold_met":
                attempt = event.payload["attempt"]
                print(f"\n[threshold met on attempt {attempt}]")

    # Show what's in the lessons memory block — these survive across
    # runs in a persistent backend.
    print("\n=== Memory contents (lessons block) ===")
    for block in await memory.working():
        if block.name == "reflexion_lessons":
            print(block.content)


if __name__ == "__main__":
    asyncio.run(main())
