"""16_swarm — Customer support with peer handoffs.

What it shows:
* Swarm has peer agents that pass control to each other via
  ``handoff(target, message)``. No central supervisor — routing
  is decentralized. Cycle detection + max_handoffs prevent
  runaway loops.
* Real-world use: exploratory or research-mode systems where
  the flow can't be pre-specified. Production warning: prefer
  Supervisor or Router for predictable routing.
* The handoff tool is injected per-peer-turn via
  ``Agent.run(extra_tools=...)`` — peers don't need to know
  about the swarm at construction time.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/16_swarm.py
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

from jeevesagent import Agent, Swarm, tool  # noqa: E402


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up a billing invoice."""
    fake = {
        "INV-2025-04": "amount=$299, status=paid, date=2026-04-01"
    }
    return fake.get(invoice_id, f"no record for {invoice_id!r}")


@tool
def search_changelog(version: str) -> str:
    """Search the technical changelog for a specific version."""
    fake = {
        "v2.3": "v2.3 (2026-04-15): fixed auth bug; added webhook retries",
        "v2.4": "v2.4 (2026-05-01): rewrote rate-limiter; ships next week",
    }
    return fake.get(version, f"no entry for {version!r}")


# ---------------------------------------------------------------------------
# Peer agents — each can hand off to another peer.
# ---------------------------------------------------------------------------

triage = Agent(
    instructions=(
        "You triage customer inquiries. Use the handoff tool to "
        "pass the conversation to:\n"
        "- 'billing' for invoices, payments, refunds\n"
        "- 'tech' for bugs, errors, version questions\n"
        "Pass the user's full original question in the handoff "
        "message; the specialist answers from there. "
        "If the question is general, answer it yourself."
    ),
    model="gpt-4.1-mini",
)

billing = Agent(
    instructions=(
        "You handle billing questions. Use lookup_invoice to find "
        "records. Be direct. If the question is technical (bugs, "
        "errors), hand off to 'tech'."
    ),
    model="gpt-4.1-mini",
    tools=[lookup_invoice],
)

tech = Agent(
    instructions=(
        "You handle technical questions. Use search_changelog for "
        "version-specific info. If the question becomes about "
        "billing, hand off to 'billing'."
    ),
    model="gpt-4.1-mini",
    tools=[search_changelog],
)


async def main() -> None:
    agent = Agent(
        "Customer support hub.",
        model="gpt-4.1-mini",
        architecture=Swarm(
            agents={
                "triage": triage,
                "billing": billing,
                "tech": tech,
            },
            entry_agent="triage",
            max_handoffs=4,
            detect_cycles=True,
        ),
    )

    prompt = (
        "Hi — I was charged $299 on invoice INV-2025-04 but I'm "
        "still seeing the auth bug from v2.3. Did v2.4 fix it?"
    )

    print("=" * 70)
    print("Swarm — peer handoffs in customer support")
    print("=" * 70)
    print(f"User: {prompt}\n")

    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "tool_call":
            call = ev.payload.get("call", {})
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            if tool_name == "handoff":
                target = args.get("target", "?")
                msg = args.get("message", "")[:60]
                print(
                    f"\n\n[handoff requested → {target}] {msg}..."
                )
            else:
                print(f"\n  [tool] {tool_name}({args})")
        elif kind == "tool_result":
            result = ev.payload.get("result", {})
            tool_name_hint = (result.get("output") or "")[:80]
            # only print non-handoff results (less noise)
            if "[handoff requested" not in tool_name_hint:
                print(f"  [result] {tool_name_hint}")
        elif kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "swarm.active":
                agent_name = ev.payload.get("agent")
                count = ev.payload.get("handoff_count", 0)
                print(f"\n\n--- ACTIVE: {agent_name} (turn {count}) ---")
            elif name == "swarm.handoff":
                f = ev.payload.get("from_agent")
                t = ev.payload.get("to_agent")
                print(f"\n  [{f} → {t}]")
            elif name == "swarm.cycle_detected":
                print("\n  [cycle detected — bailing]")
            elif name == "swarm.completed":
                print(
                    f"\n--- ✓ completed by "
                    f"{ev.payload.get('agent')} ---"
                )


if __name__ == "__main__":
    asyncio.run(main())
