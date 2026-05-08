"""Example 4 — Structured outputs with type-safe validation.

Real production agents need to emit *data*, not just prose. This
example shows the framework's structured-output contract end-to-end:

  1. Define a Pydantic ``BaseModel`` describing the shape you want.
  2. Pass it as ``output_schema=`` to ``agent.run()``.
  3. Read ``result.parsed`` — a typed, validated instance.

Under the hood:

  * The framework appends a JSON-schema directive to the run's
    system prompt (the static instructions you passed at
    construction stay unchanged).
  * The model emits JSON that matches; the framework parses it.
  * If parsing fails, the framework gives the model up to
    ``output_validation_retries`` (default 1) to fix the output —
    feeding the validation error back as a USER message so the
    model can correct itself. After that budget is exhausted,
    :class:`OutputValidationError` is raised.

What this example does:

  * Extracts a structured ``MeetingSummary`` from a raw meeting
    transcript — title, date, attendees, action items, decisions.
  * Demonstrates that the model output validates as a typed object
    you can index into, serialise back to JSON, etc.

Run::

    OPENAI_API_KEY=sk-... python examples/04_structured_outputs.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print("\n  ✗ Set OPENAI_API_KEY in your environment or .env file.\n")
    sys.exit(1)


from jeevesagent import Agent  # noqa: E402

# --------------------------------------------------------------------
# 1. Schema definition — what we want extracted
# --------------------------------------------------------------------


class ActionItem(BaseModel):
    owner: str = Field(description="Name of the person assigned to the task.")
    description: str = Field(description="What needs to be done, one sentence.")
    due_date: date | None = Field(
        default=None,
        description="ISO date string when the task is due, or null.",
    )
    priority: Literal["low", "medium", "high"] = "medium"


class MeetingSummary(BaseModel):
    title: str = Field(description="One-line title summarising the meeting.")
    meeting_date: date = Field(
        description="The date the meeting took place, ISO format."
    )
    attendees: list[str] = Field(
        description="Names of everyone present at the meeting."
    )
    decisions: list[str] = Field(
        description="Concrete decisions reached during the meeting."
    )
    action_items: list[ActionItem]
    overall_sentiment: Literal["positive", "neutral", "tense"]


# --------------------------------------------------------------------
# 2. Sample input — a raw meeting transcript
# --------------------------------------------------------------------


TRANSCRIPT = """
Acme Corp — Engineering Sync
Date: 2026-05-04

Attendees: Mira Castellanos (CEO), Tomás Reyes (VP Eng), Priya Iyer
(staff engineer), Lukas Brandt (PM).

Mira opened by confirming the launch slip — the v2.0 release will
ship on May 28, two weeks later than originally planned. The team
agreed to use the extra time to harden the migration tooling rather
than ship more features.

Priya raised concern about the rollback story for tenants on the
old schema. Tomás said the platform team would own building a
shadow-replica path before launch; Priya volunteered to write the
RFC by next Friday. Lukas will update the launch comms to
customers by Monday so they have a week's notice of the date
change.

The meeting ended on a constructive note — everyone aligned on
the new date and the migration-tooling priority.
"""


async def main() -> None:
    agent = Agent(
        "You extract structured meeting summaries from raw "
        "transcripts. Be faithful to the source — do not invent "
        "attendees, dates, or action items that are not in the "
        "transcript.",
        model="gpt-4.1-mini",
    )

    print("\n  Example 4 — Structured outputs\n")
    print(f"  Schema: {MeetingSummary.__name__}")
    print(f"  Fields: {sorted(MeetingSummary.model_fields)}\n")

    result = await agent.run(
        TRANSCRIPT,
        output_schema=MeetingSummary,
        output_validation_retries=1,
    )

    summary: MeetingSummary = result.parsed  # type: ignore[assignment]

    # Now we have a typed object — index into it, serialise it,
    # pass it to downstream code, store it in a typed table.
    print("─" * 72)
    print(f"Title       : {summary.title}")
    print(f"Date        : {summary.meeting_date}")
    print(f"Sentiment   : {summary.overall_sentiment}")
    print(f"Attendees   : {', '.join(summary.attendees)}")
    print()
    print("Decisions:")
    for d in summary.decisions:
        print(f"  • {d}")
    print()
    print("Action items:")
    for item in summary.action_items:
        due = item.due_date.isoformat() if item.due_date else "—"
        print(
            f"  • [{item.priority:6s}] {item.owner:20s} due {due}: "
            f"{item.description}"
        )
    print("─" * 72)
    print(
        f"  ({result.turns} turns, "
        f"{result.tokens_in}+{result.tokens_out} tokens)"
    )
    print()
    print("  Raw model output (JSON kept on result.output for logging):")
    print(f"  {result.output[:200]}{'...' if len(result.output) > 200 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
