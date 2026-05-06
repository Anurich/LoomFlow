"""09_self_refine — Iterative refinement via critique.

What it shows:
* The ``SelfRefine`` architecture wraps any base (default ``ReAct``)
  with a critic + refiner loop. Same model plays all three roles.
* Critique events surface through ``agent.stream`` so you can watch
  what the critic flagged and how the refiner addressed it.
* Convergence detection: when the critic emits the configured
  ``stop_phrase`` (default ``"no issues"``), refinement halts.

We use ``ScriptedModel`` for determinism — no API key, no network.
The script simulates a generator → critic → refiner → critic
sequence where round 2 converges.

Run:
    python examples/09_self_refine.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import Agent, ScriptedModel, ScriptedTurn, SelfRefine


async def main() -> None:
    # Three model turns:
    # 1. Generator (round 0): produces an initial draft.
    # 2. Critic (round 1): finds issues.
    # 3. Refiner (round 1): produces a revision.
    # 4. Critic (round 2): says "no issues" → converge.
    model = ScriptedModel(
        [
            ScriptedTurn(text="Draft: Tokyo is a city in Japan."),
            ScriptedTurn(
                text=(
                    "Issues: too short; missing population; missing "
                    "any specific cultural detail."
                )
            ),
            ScriptedTurn(
                text=(
                    "Revised: Tokyo is the capital of Japan and "
                    "home to ~14 million people in the metropolis "
                    "(37M+ in the greater metro area). Famous for "
                    "Shibuya Crossing, Tsukiji's old fish market, "
                    "and a punctual rail network."
                )
            ),
            ScriptedTurn(text="no issues"),
        ]
    )

    agent = Agent(
        "You write factual answers about cities.",
        model=model,
        architecture=SelfRefine(max_rounds=3),
    )

    print("=== Streaming events ===")
    async for event in agent.stream("Tell me about Tokyo."):
        if event.kind == "architecture_event":
            name = event.payload.get("name", "")
            if name == "self_refine.critique":
                critique = event.payload.get("critique", "")
                print(f"\n[critic, round {event.payload['round']}] {critique}")
            elif name == "self_refine.refined":
                refined = event.payload.get("output", "")[:80]
                print(
                    f"\n[refiner, round {event.payload['round']}] "
                    f"{refined}..."
                )
            elif name == "self_refine.converged":
                print(
                    f"\n[converged after round "
                    f"{event.payload['round']}]"
                )

    # Re-run with a fresh ScriptedModel to print the final RunResult.
    # In production, you'd consume ``stream()`` once and read the
    # last ``COMPLETED`` event's ``result`` payload — but for this
    # demo we want both views.
    fresh_model = ScriptedModel(
        [
            ScriptedTurn(text="Draft: Tokyo is a city in Japan."),
            ScriptedTurn(text="Issues: too short; missing details."),
            ScriptedTurn(
                text=(
                    "Revised: Tokyo is Japan's capital, ~14M in the "
                    "metropolis (37M+ greater metro). Famous for "
                    "Shibuya Crossing and Tsukiji."
                )
            ),
            ScriptedTurn(text="no issues"),
        ]
    )
    fresh_agent = Agent(
        "You write factual answers about cities.",
        model=fresh_model,
        architecture=SelfRefine(max_rounds=3),
    )
    result = await fresh_agent.run("Tell me about Tokyo.")
    print("\n=== Final ===")
    print(result.output)
    print(f"\nTurns:  {result.turns}")
    print(
        f"Tokens: {result.tokens_in} in / {result.tokens_out} out"
    )


if __name__ == "__main__":
    asyncio.run(main())
