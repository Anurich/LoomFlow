"""19_rewoo — Plan-then-tool-execute with placeholder substitution.

What it shows:
* The ``ReWOO`` architecture commits to a plan upfront, then runs
  each step as a real tool call. Args can reference prior step
  outputs via ``{{En}}`` placeholders — the worker substitutes them
  from the prior result map at dispatch time.
* Independent steps (no dependency on each other) run in **parallel**
  via an anyio task group.
* Total cost: **2 LLM calls + N tool calls**. ReAct on the same
  task is roughly N+1 LLM calls (one per turn), so ReWOO is
  cheaper for tool-heavy workloads where the plan is predictable.

Toy task: parallel data lookup + serial summarization. Two
independent ``fetch_*`` calls run in parallel as level 0; the
``summarize`` step waits for both, then synthesizes.

Run:
    python examples/19_rewoo.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    ReWOO,
    ScriptedModel,
    ScriptedTurn,
    tool,
)


@tool
def fetch_population(country: str) -> str:
    """Look up the population of a country."""
    fake_db = {
        "Japan": "125 million",
        "Brazil": "215 million",
    }
    return fake_db.get(country, "unknown")


@tool
def fetch_capital(country: str) -> str:
    """Look up the capital city of a country."""
    fake_db = {
        "Japan": "Tokyo",
        "Brazil": "Brasília",
    }
    return fake_db.get(country, "unknown")


@tool
def format_fact(country: str, capital: str, population: str) -> str:
    """Compose a single sentence."""
    return (
        f"{country}'s capital is {capital}; "
        f"its population is {population}."
    )


async def main() -> None:
    # The planner should emit:
    # E1: fetch_capital(country=Japan)        ← independent
    # E2: fetch_population(country=Japan)     ← independent (parallel with E1)
    # E3: format_fact(country=Japan, capital={{E1}}, population={{E2}})
    # Then the solver synthesizes.
    plan_json = (
        "[\n"
        '  {"id": "E1", "tool": "fetch_capital", "args": {"country": "Japan"}},\n'
        '  {"id": "E2", "tool": "fetch_population", "args": {"country": "Japan"}},\n'
        '  {"id": "E3", "tool": "format_fact",\n'
        '   "args": {"country": "Japan", "capital": "{{E1}}", "population": "{{E2}}"}}\n'
        "]"
    )
    model = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),  # planner
            ScriptedTurn(  # solver
                text=(
                    "Japan: capital Tokyo, population 125 million."
                )
            ),
        ]
    )
    agent = Agent(
        "Country fact lookup.",
        model=model,
        tools=[fetch_population, fetch_capital, format_fact],
        architecture=ReWOO(parallel_levels=True),
    )

    print("=== Streaming events ===")
    async for event in agent.stream(
        "Tell me about Japan: capital + population in one sentence."
    ):
        if event.kind != "architecture_event":
            continue
        name = event.payload.get("name", "")
        if name == "rewoo.plan_created":
            steps = event.payload["steps"]
            print(f"[plan: {len(steps)} step(s)]")
            for s in steps:
                deps_str = (
                    f" (deps: {s['depends_on']})"
                    if s["depends_on"]
                    else ""
                )
                print(f"  {s['id']} → {s['tool']}{deps_str}")
        elif name == "rewoo.level_started":
            level = event.payload["level"]
            ids = event.payload["step_ids"]
            print(f"\n[level {level}] running in parallel: {ids}")
        elif name == "rewoo.step_completed":
            sid = event.payload["step_id"]
            tool_name = event.payload["tool"]
            output = event.payload["output"]
            err = event.payload.get("error")
            if err:
                print(f"  [{sid}/{tool_name}] ERROR: {err}")
            else:
                print(f"  [{sid}/{tool_name}] → {output[:60]}")
        elif name == "rewoo.solver_started":
            print("\n[synthesizing...]")
        elif name == "rewoo.completed":
            n = event.payload["num_steps"]
            print(f"\n[completed — {n} step(s) total]")

    # Re-run with fresh model for the final answer.
    fresh = ScriptedModel(
        [
            ScriptedTurn(text=plan_json),
            ScriptedTurn(
                text="Japan: capital Tokyo, population 125 million."
            ),
        ]
    )
    fresh_agent = Agent(
        "lookup",
        model=fresh,
        tools=[fetch_population, fetch_capital, format_fact],
        architecture=ReWOO(),
    )
    result = await fresh_agent.run("about Japan")
    print(f"\n=== Final answer ===\n{result.output}")
    print(f"\nTurns: {result.turns}  (1 planner + 1 solver = 2 LLM calls)")


if __name__ == "__main__":
    asyncio.run(main())
