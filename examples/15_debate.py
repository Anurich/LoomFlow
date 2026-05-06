"""15_debate — N debaters argue, judge synthesizes.

What it shows:
* The ``MultiAgentDebate`` architecture orchestrates N debater
  Agents across multiple rounds. Round 0 is independent (parallel
  via anyio task group). Rounds 1..K each debater sees the full
  prior transcript and defends or updates its position.
* Optional convergence check terminates early when all debaters
  converge on the same answer (whitespace-normalized).
* Judge synthesizes the final answer from the transcript. Pass
  ``judge=None`` to fall back to majority vote across the final
  round.
* Each debater + judge invocation gets a deterministic session_id
  (``{parent}__debater_<i>_round_<r>`` / ``{parent}__judge``) for
  replay correctness.

A toy investment-decision debate: optimist + skeptic + analyst
disagree about a Series B investment; a judge synthesizes the
verdict. ``ScriptedModel`` makes the example deterministic.

Run:
    python examples/15_debate.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    MultiAgentDebate,
    ScriptedModel,
    ScriptedTurn,
)


def _scripted_agent(label: str, replies: list[str]) -> Agent:
    return Agent(
        f"You are the {label}.",
        model=ScriptedModel([ScriptedTurn(text=r) for r in replies]),
    )


async def main() -> None:
    # Three debaters with deliberately divergent priors. In production,
    # use different MODELS (Claude + GPT + Llama) for genuine diversity.
    optimist = _scripted_agent(
        "investment optimist",
        replies=[
            "INVEST. 200% ARR growth is rare; the upside swamps the burn.",
            (
                "Maintaining INVEST. The skeptic raises good points on "
                "burn but at this growth rate the company will outrun "
                "those concerns within two quarters."
            ),
        ],
    )
    skeptic = _scripted_agent(
        "investment skeptic",
        replies=[
            "PASS. $4M annual burn against $2M ARR is unsustainable.",
            (
                "Maintaining PASS. The optimist hand-waves the burn. "
                "Two quarters of cash runway is not a margin of safety."
            ),
        ],
    )
    analyst = _scripted_agent(
        "quantitative analyst",
        replies=[
            (
                "CONDITIONAL: only invest if a clear path to $10M ARR "
                "exists within 18 months. Otherwise pass."
            ),
            (
                "Refining CONDITIONAL. The growth trajectory does suggest "
                "$10M ARR is reachable; the burn-rate concern is "
                "secondary if the unit economics improve at scale."
            ),
        ],
    )

    judge = _scripted_agent(
        "impartial chief investment officer",
        replies=[
            (
                "DECISION: invest, with conditions. The growth "
                "trajectory is too strong to pass; structure the "
                "investment with milestone-based releases tied to "
                "ARR targets to address the burn-rate concern."
            )
        ],
    )

    parent_model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "Investment committee moderator.",
        model=parent_model,
        architecture=MultiAgentDebate(
            debaters=[optimist, skeptic, analyst],
            judge=judge,
            rounds=1,
            convergence_check=False,
        ),
    )

    print("=== Streaming events ===")
    async for event in agent.stream(
        "Should we invest $10M in Series B of a vertical AI startup with "
        "$2M ARR, growing 200% YoY but burning $4M annually?"
    ):
        if event.kind != "architecture_event":
            continue
        name = event.payload.get("name", "")
        if name == "debate.round_started":
            r = event.payload["round"]
            phase = event.payload.get("phase", "?")
            print(f"\n[round {r} started — {phase}]")
        elif name == "debate.response":
            r = event.payload["round"]
            d = event.payload["debater"]
            resp = event.payload["response"]
            print(f"  [{d} @ r{r}] {resp[:100]}...")
        elif name == "debate.converged":
            r = event.payload["round"]
            print(f"\n[converged at round {r}]")
        elif name == "debate.judging":
            print("\n[judge deliberating...]")
        elif name == "debate.synthesized":
            method = event.payload["method"]
            print(f"\n[synthesized via {method}]")

    # Re-run with fresh agents for the final answer print.
    fresh_optimist = _scripted_agent("optimist", ["INVEST", "INVEST still"])
    fresh_skeptic = _scripted_agent("skeptic", ["PASS", "PASS still"])
    fresh_analyst = _scripted_agent(
        "analyst", ["CONDITIONAL", "CONDITIONAL refined"]
    )
    fresh_judge = _scripted_agent(
        "judge",
        ["Final verdict: INVEST with milestone-based releases."],
    )
    fresh_agent = Agent(
        "moderator",
        model=ScriptedModel([ScriptedTurn(text="never reached")]),
        architecture=MultiAgentDebate(
            debaters=[fresh_optimist, fresh_skeptic, fresh_analyst],
            judge=fresh_judge,
            rounds=1,
            convergence_check=False,
        ),
    )
    result = await fresh_agent.run("invest or pass?")
    print(f"\n=== Final verdict ===\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
