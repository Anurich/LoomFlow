"""Example 9 — Expose a Workflow as an Agent tool.

The opposite composition direction from Example 7. There an outer
workflow contained Agents; here an outer Agent has a workflow
available as a tool.

Use case: an open-ended customer-support Agent might handle a
hundred different conversation paths, but when the user actually
asks for a refund the response MUST follow a deterministic flow:
validate → record → notify. Wrapping that flow as a workflow and
exposing it via :meth:`~jeevesagent.Workflow.as_tool` gives the
Agent a deterministic compliance gate inside its otherwise free
reasoning.

The unified audit log shows both the agent's tool_call entry and
the workflow's per-step entries — same ``user_id``, same
``session_id``, single trace.

Run::

    OPENAI_API_KEY=sk-... python examples/09_workflow_as_tool.py
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


from jeevesagent import Agent, InMemoryAuditLog, Workflow  # noqa: E402

MODEL = "gpt-4.1-mini"


# Refund workflow — deterministic three-step process. Each step
# in production would hit a real system (DB, email, billing API);
# here they return strings so the trace is visible.

async def validate_refund_request(text: str) -> dict[str, object]:
    return {"input": text, "validated": True, "amount_estimate": "$49.00"}


async def create_refund_record(state: dict[str, object]) -> dict[str, object]:
    state["refund_id"] = "REF-2026-1234"
    state["status"] = "issued"
    return state


async def notify_customer(state: dict[str, object]) -> str:
    return (
        f"Refund {state['refund_id']} for "
        f"{state['amount_estimate']} issued and customer notified."
    )


refund_workflow = Workflow.chain(
    [validate_refund_request, create_refund_record, notify_customer],
    name="process_refund",
)


async def main() -> None:
    print("\n  Example 9 — Workflow exposed as Agent tool\n")

    audit = InMemoryAuditLog()

    # The agent has the refund workflow available as a tool. Any
    # other tools the agent owns (search, lookup, etc.) live next
    # to it on the same ``tools=[]`` list.
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
    print(f"  agent answer: {result.output}\n")

    entries = await audit.query(user_id="alice")
    print(f"  audit log: {len(entries)} entries from this run")
    for e in entries:
        print(f"    [{e.actor:>9}] {e.action}")


if __name__ == "__main__":
    asyncio.run(main())
