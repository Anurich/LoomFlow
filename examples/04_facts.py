"""04_facts — Bi-temporal facts + auto-consolidation.

What it shows:
* ``VectorMemory`` with a ``Consolidator`` that extracts facts from
  episodes via an LLM.
* ``auto_consolidate=True`` so facts get extracted automatically
  after every run.
* Facts surface in the next run's system message, so the agent
  "remembers" what previous runs taught it.
* Bi-temporal supersession: when a contradicting fact arrives, the
  prior fact's ``valid_until`` is closed off without deleting it.

This example uses :class:`ScriptedModel` for both the agent and the
consolidator so it runs deterministically without API keys.

Run:
    python examples/04_facts.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from jeevesagent import (
    Agent,
    Consolidator,
    Fact,
    ScriptedModel,
    ScriptedTurn,
    VectorMemory,
)


async def main() -> None:
    # The consolidator uses a scripted model that always returns one
    # extracted fact per episode. (Real consolidators use Claude/GPT.)
    extracted_json = (
        '[{"subject":"user","predicate":"name_is",'
        '"object":"Alice","confidence":0.95}]'
    )
    consolidator_model = ScriptedModel(
        [ScriptedTurn(text=extracted_json)] * 10
    )

    memory = VectorMemory(consolidator=Consolidator(model=consolidator_model))

    # Pre-seed a historical fact so we can show supersession.
    base = datetime.now(UTC) - timedelta(days=30)
    await memory.facts.append(
        Fact(
            subject="user",
            predicate="lives_in",
            object="Tokyo",
            valid_from=base,
            recorded_at=base,
        )
    )

    # Add a more-recent contradicting fact — supersession kicks in.
    await memory.facts.append(
        Fact(
            subject="user",
            predicate="lives_in",
            object="Paris",
            valid_from=datetime.now(UTC),
            recorded_at=datetime.now(UTC),
        )
    )

    print("--- all facts ---")
    for f in await memory.facts.all_facts():
        suffix = " (superseded)" if f.valid_until is not None else ""
        print(f"  {f.format()}{suffix}")

    print("\n--- query at base + 5 days ---")
    on_day_5 = base + timedelta(days=5)
    for f in await memory.facts.query(predicate="lives_in", valid_at=on_day_5):
        print(f"  {f.format()}")

    print("\n--- query right now ---")
    for f in await memory.facts.query(predicate="lives_in", valid_at=datetime.now(UTC)):
        print(f"  {f.format()}")

    # Now run the agent with auto-consolidate; it'll extract more
    # facts from each conversation.
    agent_model = ScriptedModel(
        [ScriptedTurn(text="Got it!")] * 10
    )
    agent = Agent(
        "Personal assistant.",
        model=agent_model,
        memory=memory,
        auto_consolidate=True,
    )
    await agent.run("I prefer dark mode.")
    await agent.run("My team is on the enterprise plan.")

    print("\n--- facts after two more runs ---")
    for f in await memory.facts.all_facts():
        print(f"  {f.format()}")


if __name__ == "__main__":
    asyncio.run(main())
