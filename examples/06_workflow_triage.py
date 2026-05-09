"""Example 6 — Workflow + Agent composition.

Demonstrates the two production-shape patterns the framework
supports:

1. **Workflow with Agent steps.** A deterministic DAG (the
   developer wrote `if/else` for the routing) calls an
   :class:`~jeevesagent.Agent` at each leaf for the open-ended
   reply. This is the most common production shape: "trustworthy
   skeleton with smart leaves."

2. **Agent that calls a Workflow as a tool.** The opposite
   composition — an open-ended customer-support agent has a
   deterministic refund-processing workflow available as a tool.
   When the user asks for a refund, the agent invokes the
   workflow as one tool call and the workflow's strict steps
   ensure compliance.

Both patterns share the same observability spine — telemetry
spans, audit-log entries, ``user_id`` partition — so a unified
trace shows exactly which decisions were workflow-deterministic
and which were LLM-driven.

Run::

    OPENAI_API_KEY=sk-... python examples/06_workflow_triage.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ Skipping: OPENAI_API_KEY is not set. "
        "Export it (or add it to .env) to run this example.\n"
    )
    sys.exit(0)


from jeevesagent import (  # noqa: E402
    Agent,
    InMemoryAuditLog,
    InMemoryMemory,
    Workflow,
)

# --------------------------------------------------------------------
# Pattern 1: Workflow with Agent steps
# --------------------------------------------------------------------
#
# The classifier is a tiny model — cheap, fast, deterministic
# routing. The specialists are full models — they get the original
# question and produce the answer. The DEVELOPER controls the
# branching; the LLM only does the work each step asks of it.

MODEL = "gpt-4.1-mini"


async def classify_topic(text: str) -> str:
    """Tag the question as one of: billing | tech | general."""
    classifier = Agent(
        "Classify the customer question into exactly one of: "
        "billing, tech, general. Reply with only the single word.",
        model=MODEL,
        memory=InMemoryMemory(),
    )
    result = await classifier.run(text)
    label = result.output.strip().lower()
    return label if label in {"billing", "tech", "general"} else "general"


def make_specialist(role: str, system: str) -> Agent:
    """Build a topic specialist Agent."""
    return Agent(system, model=MODEL, memory=InMemoryMemory())


async def run_triage_workflow() -> None:
    print("=" * 72)
    print("  Pattern 1: Workflow with Agent steps")
    print("=" * 72)

    billing = make_specialist(
        "billing",
        "You handle billing questions. Be concise, cite invoice numbers "
        "if helpful, and tell the user how to escalate if you can't help.",
    )
    tech = make_specialist(
        "tech",
        "You handle technical questions about Acme's product. Keep "
        "answers short, suggest specific commands or settings.",
    )
    general = make_specialist(
        "general",
        "You handle generic customer questions. Friendly tone, redirect "
        "to a specialist if the question is really about billing or tech.",
    )

    wf = Workflow.route(
        classifier=classify_topic,
        routes={"billing": billing, "tech": tech, "general": general},
    )

    questions = [
        "I think I was double-charged last month — what should I do?",
        "How do I rotate my API key?",
        "Hi, just wanted to say thanks for the great support last week.",
    ]

    for q in questions:
        print("─" * 72)
        print(f"Q: {q}")
        result = await wf.run(q, user_id="alice", session_id="demo")
        # Visited shows which branch the workflow picked — useful
        # for debugging and as a sanity check that the classifier
        # is doing its job.
        branch = next(n for n in result.visited if n.startswith("route_"))
        print(f"   → routed to {branch}")
        print(f"A: {result.output}")
    print("─" * 72 + "\n")


# --------------------------------------------------------------------
# Pattern 2: Agent with a Workflow exposed as a tool
# --------------------------------------------------------------------
#
# The agent is the open-ended reasoner. The refund workflow is a
# deterministic three-step process the agent can invoke when
# appropriate. The agent decides WHEN to call it; the workflow
# decides HOW it executes.

# The refund workflow's steps. In production each would hit a real
# system (database, email, billing API). For this example we just
# return strings so the trace is visible.

async def validate_refund_request(text: str) -> dict[str, object]:
    return {
        "input": text,
        "validated": True,
        "amount_estimate": "$49.00",
    }


async def create_refund_record(state: dict[str, object]) -> dict[str, object]:
    state["refund_id"] = "REF-2026-1234"
    state["status"] = "issued"
    return state


async def notify_customer(state: dict[str, object]) -> str:
    rid = state["refund_id"]
    amount = state["amount_estimate"]
    return f"Refund {rid} for {amount} issued and customer notified."


refund_workflow = Workflow.chain(
    [validate_refund_request, create_refund_record, notify_customer],
    name="process_refund",
)


async def run_agent_with_workflow_tool() -> None:
    print("=" * 72)
    print("  Pattern 2: Agent with a Workflow exposed as a tool")
    print("=" * 72)

    audit = InMemoryAuditLog()

    agent = Agent(
        "You are a customer-support agent for Acme. When a customer "
        "asks for a refund, ALWAYS use the process_refund tool — "
        "never invent a refund yourself. Return whatever the tool "
        "produced as your final answer.",
        model=MODEL,
        tools=[
            refund_workflow.as_tool(
                name="process_refund",
                description=(
                    "Validate, record, and notify the customer about a "
                    "refund request. Pass the customer's exact question "
                    "as the input."
                ),
            ),
        ],
        audit_log=audit,
    )

    result = await agent.run(
        "I want a refund for my last invoice — it was double-charged.",
        user_id="alice",
        session_id="refund-demo",
    )
    print("─" * 72)
    print(f"Agent answer: {result.output}")

    # Inspect the audit log — both the agent's tool_call entry AND
    # the workflow's per-step entries land here, all attributed to
    # the same user_id.
    entries = await audit.query(user_id="alice")
    print(f"\n  Audit log: {len(entries)} entries from this run")
    for e in entries[-6:]:
        print(f"    [{e.actor:>9}] {e.action}")
    print("─" * 72 + "\n")


async def main() -> None:
    print("\n  Example 6 — Workflow + Agent composition\n")
    await run_triage_workflow()
    await run_agent_with_workflow_tool()


if __name__ == "__main__":
    asyncio.run(main())
