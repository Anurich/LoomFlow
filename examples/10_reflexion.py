"""10_reflexion — Math word problem solver with verbal RL.

What it shows:
* Reflexion wraps a base (ReAct here) with an evaluator + reflector.
  The evaluator scores each attempt 0-1; below threshold, the
  reflector emits a one-sentence "lesson" persisted to a memory
  block. The next attempt sees the lesson via memory.working() and
  benefits.
* Real-world use: any task where attempts can fail in patterned
  ways — math problems, code generation, structured extraction.
  Reflexion turns failures into prompt-level training data.
* Cross-session learning works too with a persistent memory backend
  (SqliteMemory / PostgresMemory) — lessons survive process restarts.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/10_reflexion.py
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

from jeevesagent import (  # noqa: E402
    Agent,
    InMemoryMemory,
    ReAct,
    Reflexion,
)


async def main() -> None:
    memory = InMemoryMemory()
    agent = Agent(
        instructions=(
            "You solve math word problems step-by-step. "
            "Always show your work, then give a final numeric answer "
            "on a line by itself prefixed with 'Answer: '."
        ),
        model="gpt-4.1-mini",
        memory=memory,
        architecture=Reflexion(
            base=ReAct(),
            max_attempts=3,
            threshold=0.85,
            evaluator_prompt=(
                "Score the agent's answer for the math problem "
                "from 0 (wrong/incomplete) to 1 (correct, fully "
                "shown).\n\nOutput exactly:\nscore: <0-1>\nThen one "
                "line of justification."
            ),
            reflector_prompt=(
                "Identify in ONE sentence the specific reasoning "
                "mistake the agent made; describe what to do "
                "differently next time. Be concrete."
            ),
        ),
    )

    prompt = (
        "A train leaves station A at 9:00 AM travelling at 60 mph. "
        "A second train leaves station B (180 miles east of A) at "
        "9:30 AM travelling west at 80 mph. At what time do they "
        "meet, and how far from station A?"
    )

    print("=" * 70)
    print("Reflexion — math word problem with self-improvement")
    print("=" * 70)
    print()

    current_attempt = 0
    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "reflexion.attempt_started":
                current_attempt = ev.payload.get("attempt")
                print(
                    f"\n\n--- Attempt {current_attempt} of "
                    f"{ev.payload.get('max_attempts')} ---"
                )
            elif name == "reflexion.evaluated":
                score = ev.payload.get("score", 0.0)
                print(
                    f"\n\n[evaluator] attempt "
                    f"{ev.payload.get('attempt')} score = {score:.2f}"
                )
            elif name == "reflexion.lesson_produced":
                lesson = ev.payload.get("lesson", "")
                print(f"\n[reflector] lesson: {lesson}\n")
            elif name == "reflexion.threshold_met":
                print(
                    f"\n--- ✓ threshold met at attempt "
                    f"{ev.payload.get('attempt')} ---"
                )
            elif name == "reflexion.max_attempts_reached":
                print(
                    f"\n--- max attempts reached "
                    f"(final score "
                    f"{ev.payload.get('final_score', 0.0):.2f}) ---"
                )
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n\n" + "=" * 70)
            print("FINAL ANSWER")
            print("=" * 70)
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}"
            )

    # Show what was learned (lessons block in working memory)
    blocks = await memory.working()
    lessons_block = next(
        (b for b in blocks if b.name == "reflexion_lessons"), None
    )
    if lessons_block:
        print("\n=== Lessons persisted (would carry to next run) ===")
        print(lessons_block.content)


if __name__ == "__main__":
    asyncio.run(main())
