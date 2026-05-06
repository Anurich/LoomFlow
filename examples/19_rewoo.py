"""19_rewoo — Plan-then-tool-execute with placeholder substitution.

What it shows:
* ReWOO emits a plan whose steps are real TOOL CALLS (not text-only
  steps). Args can reference prior step outputs via ``{{En}}``
  placeholders. Independent steps run in parallel via anyio.
* Total cost: 2 LLM calls (planner + solver) + N tool calls. ReAct
  on the same task is roughly N+1 LLM calls.
* Real-world use: predictable multi-tool research / lookups —
  parallel data fetches + synthesis. Cheaper than ReAct on
  tool-heavy workloads with stable structure.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/19_rewoo.py
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

from jeevesagent import Agent, ReWOO, tool  # noqa: E402

# ---------------------------------------------------------------------------
# Tools — fake country-data lookups. In production these would hit
# real APIs (REST Countries, World Bank, etc.).
# ---------------------------------------------------------------------------


_COUNTRIES = {
    "japan": {
        "capital": "Tokyo",
        "population": "125 million",
        "language": "Japanese",
        "currency": "Yen (¥)",
    },
    "brazil": {
        "capital": "Brasília",
        "population": "215 million",
        "language": "Portuguese",
        "currency": "Real (R$)",
    },
    "iceland": {
        "capital": "Reykjavík",
        "population": "390,000",
        "language": "Icelandic",
        "currency": "Króna (kr)",
    },
}


def _lookup(country: str, key: str) -> str:
    info = _COUNTRIES.get(country.lower())
    if info is None:
        return f"unknown country: {country!r}"
    return info.get(key, f"no {key} for {country}")


@tool
def find_capital(country: str) -> str:
    """Look up a country's capital city."""
    return _lookup(country, "capital")


@tool
def find_population(country: str) -> str:
    """Look up a country's population."""
    return _lookup(country, "population")


@tool
def find_language(country: str) -> str:
    """Look up a country's primary language."""
    return _lookup(country, "language")


@tool
def find_currency(country: str) -> str:
    """Look up a country's currency."""
    return _lookup(country, "currency")


async def main() -> None:
    agent = Agent(
        instructions=(
            "You assemble structured country fact sheets using "
            "the available lookup tools."
        ),
        model="gpt-4.1-mini",
        tools=[
            find_capital,
            find_population,
            find_language,
            find_currency,
        ],
        architecture=ReWOO(parallel_levels=True),
    )

    prompt = (
        "Build a fact sheet for Japan covering capital, population, "
        "primary language, and currency. Use the lookup tools."
    )

    print("=" * 70)
    print("ReWOO — country fact-sheet (parallel tool execution)")
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
            if name == "rewoo.plan_created":
                steps = ev.payload.get("steps", [])
                print(f"[plan: {len(steps)} step(s)]")
                for s in steps:
                    deps_str = (
                        f" (deps: {s['depends_on']})"
                        if s.get("depends_on")
                        else ""
                    )
                    print(f"  {s['id']} → {s['tool']}{deps_str}")
            elif name == "rewoo.level_started":
                ids = ev.payload.get("step_ids", [])
                level = ev.payload.get("level")
                print(
                    f"\n[level {level}] running in parallel: {ids}"
                )
            elif name == "rewoo.step_completed":
                sid = ev.payload.get("step_id")
                tool_name = ev.payload.get("tool")
                output = ev.payload.get("output", "")[:60]
                err = ev.payload.get("error")
                if err:
                    print(f"  [{sid}/{tool_name}] ERROR: {err}")
                else:
                    print(f"  [{sid}/{tool_name}] → {output}")
            elif name == "rewoo.solver_started":
                print("\n[synthesizing fact sheet...]")
            elif name == "rewoo.completed":
                print(
                    f"\n--- ✓ completed "
                    f"({ev.payload.get('num_steps')} steps) ---"
                )
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n\n" + "=" * 70)
            print("FACT SHEET")
            print("=" * 70)
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
