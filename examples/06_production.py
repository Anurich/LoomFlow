"""06_production — Production-shaped agent in ~30 lines.

What it shows:
* All seven cross-cutting concerns wired up: model, memory + facts,
  durable runtime, tools, permissions, budget, audit log, telemetry,
  auto-consolidation.
* A clean shape you can copy-paste into a real service.

Falls back gracefully when API keys aren't set.

Run:
    python examples/06_production.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import timedelta
from pathlib import Path

from jeevesagent import (
    Agent,
    Consolidator,
    EchoModel,
    FileAuditLog,
    InMemoryMemory,
    Mode,
    NoTelemetry,
    ScriptedModel,
    ScriptedTurn,
    SqliteRuntime,
    StandardPermissions,
    tool,
)
from jeevesagent.governance.budget import BudgetConfig, StandardBudget


@tool
async def search(query: str) -> str:
    """Search a knowledge base."""
    return f"results for {query!r}: [doc-1, doc-2]"


def _build_model() -> object:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-opus-4-7"  # string resolver picks AnthropicModel
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    print("(no API key; using EchoModel)")
    return EchoModel()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Use a scripted consolidator so this example doesn't burn
        # extraction tokens against the real LLM. Production code
        # should use the same model the agent uses.
        consolidator = Consolidator(
            model=ScriptedModel(
                [
                    ScriptedTurn(
                        text='[{"subject":"user","predicate":"asked_about",'
                        '"object":"agents","confidence":0.8}]'
                    )
                ]
                * 10
            )
        )

        agent = Agent(
            instructions=(
                "You are a research assistant. "
                "Use the search tool to find relevant material."
            ),
            model=_build_model(),
            memory=InMemoryMemory(consolidator=consolidator),
            runtime=SqliteRuntime(tmp_path / "journal.db"),
            tools=[search],
            permissions=StandardPermissions(mode=Mode.DEFAULT),
            budget=StandardBudget(
                BudgetConfig(
                    max_tokens=200_000,
                    max_cost_usd=5.0,
                    max_wall_clock=timedelta(minutes=10),
                    soft_warning_at=0.8,
                )
            ),
            audit_log=FileAuditLog(tmp_path / "audit.jsonl", secret="example"),
            telemetry=NoTelemetry(),  # swap for OTelTelemetry in real life
            auto_consolidate=True,
        )

        result = await agent.run(
            "Tell me what we know about agent harnesses."
        )

        print("--- run summary ---")
        print(f"output: {result.output[:120]}...")
        print(f"turns:  {result.turns}")
        print(f"tokens: {result.tokens_in} in / {result.tokens_out} out")
        print(f"cost:   ${result.cost_usd:.4f}")

        # Show what landed in the audit log.
        print("\n--- audit log entries ---")
        # FileAuditLog persists JSONL to disk; query by session.
        log = FileAuditLog(tmp_path / "audit.jsonl", secret="example")
        for entry in await log.query(session_id=result.session_id):
            print(f"  seq={entry.seq} actor={entry.actor:6} action={entry.action}")

        # And the facts that auto-consolidation extracted.
        print("\n--- extracted facts ---")
        for f in await agent._memory.facts.all_facts():  # type: ignore[attr-defined]
            print(f"  {f.format()}")


if __name__ == "__main__":
    asyncio.run(main())
