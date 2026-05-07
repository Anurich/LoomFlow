"""13_actor_critic — Code generation with adversarial review.

What it shows:
* ActorCritic requires two separate Agents — actor and critic.
  The actor generates; the critic reviews adversarially with
  structured JSON output (issues + 0-1 score). Below threshold,
  the actor refines.
* Real-world use: quality-critical code, security-sensitive
  text — anywhere you'd otherwise have a human review cycle.
* Use DIFFERENT models for actor and critic in production
  (e.g. Claude actor + GPT critic) — different priors catch
  different blind spots. Here we use the same model for the
  example; the asymmetry comes from the prompts.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/13_actor_critic.py
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

from jeevesagent import Agent, Team  # noqa: E402

actor = Agent(
    instructions=(
        "You write production Python code. Be complete, typed, "
        "and idiomatic. When given a critique, address every "
        "point. Output ONLY the code — no markdown fences, no "
        "commentary."
    ),
    model="gpt-4.1-mini",
)

critic = Agent(
    instructions=(
        "You are an ADVERSARIAL code reviewer. Your job is to find "
        "EVERY issue: missing type hints, no docstring, unhandled "
        "edge cases, mutable default args, weak error handling, "
        "performance concerns. Be ruthless.\n\n"
        "Output ONLY a JSON object — no markdown fences, no prose:\n"
        '{"issues": ["...", "..."], "score": 0.0-1.0, '
        '"summary": "..."}\n\n'
        "Score: 1.0 = ship it; 0.7-0.9 = mostly good; "
        "0.4-0.6 = real issues; 0.0-0.3 = unusable."
    ),
    model="gpt-4.1-mini",
)


async def main() -> None:
    agent = Team.actor_critic(
        actor=actor,
        critic=critic,
        max_rounds=2,
        approval_threshold=0.85,
        instructions="Code-quality coordinator.",
        model="gpt-4.1-mini",  # unused; ActorCritic drives sub-agents
    )

    prompt = (
        "Write a Python function `safe_divide(a: float, b: float) "
        "-> float` that divides two numbers safely. Handle edge "
        "cases."
    )

    print("=" * 70)
    print("ActorCritic — adversarial code review")
    print("=" * 70)
    print(f"Task: {prompt}\n")

    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "actor_critic.actor_started":
                round_num = ev.payload.get("round")
                phase = ev.payload.get("phase", "?")
                print(f"\n\n--- ACTOR (round {round_num}, {phase}) ---")
            elif name == "actor_critic.critic_started":
                print(
                    f"\n\n--- CRITIC (round "
                    f"{ev.payload.get('round')}) ---"
                )
            elif name == "actor_critic.critique":
                score = ev.payload.get("score", 0.0)
                issues = ev.payload.get("issues") or []
                print(
                    f"\n[critic verdict] score={score:.2f}, "
                    f"{len(issues)} issue(s):"
                )
                for issue in issues[:5]:
                    print(f"  • {issue}")
            elif name == "actor_critic.approved":
                print(
                    f"\n--- ✓ APPROVED at round "
                    f"{ev.payload.get('round')} "
                    f"(score={ev.payload.get('score', 0.0):.2f}) ---"
                )
            elif name == "actor_critic.max_rounds_reached":
                print(
                    f"\n--- max rounds reached "
                    f"(final score "
                    f"{ev.payload.get('final_score', 0.0):.2f}) ---"
                )
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n\n" + "=" * 70)
            print("FINAL CODE")
            print("=" * 70)
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
