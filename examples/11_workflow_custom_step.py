"""Example 11 — Agent wrapped in a custom step function.

When you pass an ``Agent`` directly to ``Workflow.chain([...])``
the framework auto-wraps ``agent.run(input)``. That's the
shortest path and what Examples 7 and 10 demonstrate.

But sometimes you need more control:

* You want to **format the prompt** from multiple state fields,
  not just pass the previous step's output verbatim.
* You want to **capture the agent's RunResult metadata**
  (``tokens_in``, ``tokens_out``, ``cost_usd``, ``turns``,
  ``interrupted``) into the workflow's state, not just keep the
  final ``output`` string.
* You want to **post-process the agent's output** (parse JSON,
  strip artifacts, validate against a schema) before passing it
  on.

For any of these, write a regular ``async def`` step that calls
``agent.run`` internally. The framework treats it as a normal
node — the Agent invocation is just an implementation detail.

Run::

    OPENAI_API_KEY=sk-... python examples/11_workflow_custom_step.py
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


from jeevesagent import Agent, InMemoryMemory, Workflow  # noqa: E402

MODEL = "gpt-4.1-mini"


async def main() -> None:
    print("\n  Example 11 — Agent wrapped in a custom step\n")

    # The agent is constructed once and reused inside a custom
    # step. The step has full control over what to pass it and
    # what to capture from its RunResult.
    summarizer = Agent(
        "You summarize a customer ticket. Output exactly two lines:\n"
        "  TLDR: <one-sentence summary>\n"
        "  PRIORITY: low|medium|high",
        model=MODEL,
        memory=InMemoryMemory(),
    )

    # ---- Step 1: prep state from raw input ------------------------------

    async def prepare(raw: dict[str, str]) -> dict[str, object]:
        return {
            "customer_name": raw["customer"],
            "subject": raw["subject"],
            "body": raw["body"],
            # Empty fields the summarize step will fill:
            "summary": "",
            "priority": "",
            "tokens_used": 0,
            "agent_turns": 0,
        }

    # ---- Step 2: custom-step that wraps the Agent ----------------------

    async def summarize_with_metadata(state: dict[str, object]) -> dict[str, object]:
        """Format a multi-field prompt, run the Agent, capture
        BOTH the parsed output AND the run metadata into state.

        This is the pattern when "just call agent.run(prev_output)"
        isn't enough.
        """
        # Build the prompt from multiple state fields:
        prompt = (
            f"From: {state['customer_name']}\n"
            f"Subject: {state['subject']}\n"
            f"\n"
            f"{state['body']}"
        )

        # Capture full RunResult, not just .output:
        result = await summarizer.run(prompt)

        # Post-process the output — split TLDR / PRIORITY into
        # separate state fields the next step can use directly.
        summary = ""
        priority = ""
        for line in result.output.splitlines():
            if line.upper().startswith("TLDR:"):
                summary = line.split(":", 1)[1].strip()
            elif line.upper().startswith("PRIORITY:"):
                priority = line.split(":", 1)[1].strip().lower()

        state["summary"] = summary
        state["priority"] = priority or "medium"
        # Capture metadata the workflow's downstream steps care about:
        state["tokens_used"] = result.tokens_in + result.tokens_out
        state["agent_turns"] = result.turns
        return state

    # ---- Step 3: enrich downstream based on captured metadata ----------

    async def route_by_priority(state: dict[str, object]) -> dict[str, object]:
        """Demonstrates that we can use the metadata captured from
        the agent run to drive downstream behaviour."""
        priority = state["priority"]
        state["routing_target"] = {
            "high": "oncall_pager",
            "medium": "support_queue",
            "low": "self_service",
        }.get(str(priority), "support_queue")
        return state

    wf = Workflow.chain(
        [prepare, summarize_with_metadata, route_by_priority],
        name="ticket-triage",
    )

    raw_ticket = {
        "customer": "alice@example.com",
        "subject": "Production outage — AcmeTrace dashboard down",
        "body": (
            "Our entire engineering team can't access the AcmeTrace "
            "dashboard. We get a 500 error on every page. This is "
            "blocking on-call response. Need help urgently."
        ),
    }

    result = await wf.run(raw_ticket, user_id="alice", session_id="custom-step-demo")
    final = result.output

    print(f"  customer    : {final['customer_name']}")
    print(f"  subject     : {final['subject']}")
    print(f"  summary     : {final['summary']}")
    print(f"  priority    : {final['priority']}")
    print(f"  routed to   : {final['routing_target']}")
    print(f"  tokens used : {final['tokens_used']}")
    print(f"  agent turns : {final['agent_turns']}")
    print(f"  visited     : {' → '.join(result.visited)}")


if __name__ == "__main__":
    asyncio.run(main())
