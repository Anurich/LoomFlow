"""15_debate — Investment-decision debate with judge synthesis.

What it shows:
* MultiAgentDebate runs N debater Agents across rounds. Round 0 is
  independent (parallel). Subsequent rounds each debater sees the
  full transcript so far and defends or updates its position.
* Real-world use: high-stakes contested questions where blind-spot
  triangulation matters — investment calls, plan reviews, factual
  research where one model's confidence is suspect.
* In production, use DIFFERENT models for each debater (Claude +
  GPT + Llama) for genuine prior diversity. Here we differentiate
  via persona prompts on the same base model.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/15_debate.py
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

from jeevesagent import Agent, Team  # noqa: E402

optimist = Agent(
    instructions=(
        "You are an optimist VC. Look for the upside case. "
        "Cite growth, market timing, founder strengths. "
        "Be specific; cite numbers from the prompt. Keep it "
        "under 5 sentences per round."
    ),
    model="gpt-4.1-mini",
)

skeptic = Agent(
    instructions=(
        "You are a skeptical investor. Stress-test claims. "
        "Cite burn rate, market risk, execution risk, "
        "competitive threats. Be specific; cite numbers. "
        "Under 5 sentences per round."
    ),
    model="gpt-4.1-mini",
)

analyst = Agent(
    instructions=(
        "You are a quantitative analyst. Stick to the unit "
        "economics — LTV/CAC, runway, growth rate, payback. "
        "If you don't have a number, say so explicitly. "
        "Under 5 sentences per round."
    ),
    model="gpt-4.1-mini",
)

judge = Agent(
    instructions=(
        "You are an impartial chief investment officer "
        "synthesizing the debate. Output a final decision "
        "(INVEST / PASS / CONDITIONAL) with one paragraph of "
        "reasoning that integrates the strongest points from "
        "each debater."
    ),
    model="gpt-4.1-mini",
)


async def main() -> None:
    agent = Team.debate(
        debaters=[optimist, skeptic, analyst],
        judge=judge,
        rounds=1,
        convergence_check=False,
        instructions="Investment committee moderator.",
        model="gpt-4.1-mini",
    )

    prompt = (
        "Should we invest $5M in Series A of a vertical AI startup "
        "with these metrics? ARR: $1.2M, growing 180% YoY. Burn: "
        "$300K/month. Runway: 12 months at current burn. Market: "
        "estimated $2B TAM in legaltech AI. Founder: ex-Google, "
        "second-time CEO. Competition: 3 well-funded competitors."
    )

    print("=" * 70)
    print("MultiAgentDebate — investment decision")
    print("=" * 70)
    print(f"Question: {prompt}\n")

    current_debater = ""
    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "debate.round_started":
                round_num = ev.payload.get("round")
                phase = ev.payload.get("phase", "?")
                print(
                    f"\n\n=== ROUND {round_num} — {phase.upper()} ==="
                )
            elif name == "debate.response":
                debater = ev.payload.get("debater", "")
                if debater != current_debater:
                    current_debater = debater
                    # No print — model_chunks for that debater
                    # already printed (they came right before this
                    # event in the stream).
            elif name == "debate.judging":
                print("\n\n=== JUDGE DELIBERATING ===")
            elif name == "debate.synthesized":
                method = ev.payload.get("method", "?")
                print(f"\n\n--- synthesized via {method} ---")
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n\n" + "=" * 70)
            print("FINAL VERDICT")
            print("=" * 70)
            print(result.get("output", "(no output)"))


if __name__ == "__main__":
    asyncio.run(main())
