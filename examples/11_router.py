"""11_router — Customer support intent routing.

What it shows:
* Router classifies the user's request via a small fast LLM call,
  then dispatches to the matching specialist Agent. Each specialist
  has its own tools and runs to completion.
* Real-world use: customer support / API gateway / helpdesk —
  cheapest multi-agent pattern (1 classifier + 1 specialist).
* Confidence threshold + ``fallback_route`` handle ambiguous
  inputs gracefully.
* All events from the specialist (model chunks, tool calls, tool
  results) stream through to the parent ``agent.stream(...)``.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/11_router.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ OPENAI_API_KEY not set — skipping this example.\n"
        "    Add OPENAI_API_KEY=sk-... to .env at repo root to run.\n"
    )
    sys.exit(0)

from jeevesagent import Agent, RouterRoute, Team, tool  # noqa: E402

# ---------------------------------------------------------------------------
# Real-ish tools per specialist. In production these would hit your
# billing system, ticketing system, knowledge base, etc.
# ---------------------------------------------------------------------------


@tool
def lookup_account(email: str) -> str:
    """Look up a customer's billing account by email."""
    fake_db = {
        "anupam@example.com": (
            "plan=Pro ($49/mo), next_bill=2026-06-01, "
            "last 3 charges: $49 (May 1), $49 (Apr 1), $49 (Mar 1)"
        ),
    }
    return fake_db.get(
        email, f"no account found for {email!r}"
    )


@tool
def issue_refund(email: str, amount_usd: float, reason: str) -> str:
    """Issue a refund. Returns a confirmation id."""
    return (
        f"refund_id=rfnd_8a2k9 amount=${amount_usd:.2f} "
        f"to={email} reason={reason!r} eta=3-5 business days"
    )


@tool
def search_docs(query: str) -> str:
    """Search the technical knowledge base."""
    fake_kb = {
        "502": (
            "Known issue: 502 errors after 2026-04-15 are caused "
            "by stale CDN config. Workaround: hard-refresh; "
            "permanent fix shipping in v2.4."
        ),
        "rate limit": (
            "Free tier: 100 req/hour. Pro tier: 10,000 req/hour. "
            "Override via X-Override-Limit header (Enterprise)."
        ),
    }
    for k, v in fake_kb.items():
        if k in query.lower():
            return v
    return f"no kb match for {query!r}"


# ---------------------------------------------------------------------------
# Specialists — each is a fully-built Agent with its own tools.
# ---------------------------------------------------------------------------


billing = Agent(
    "You are a billing specialist. Use lookup_account to find the "
    "customer's records, then issue_refund if appropriate. Be "
    "decisive and concise.",
    model="gpt-4.1-mini",
    tools=[lookup_account, issue_refund],
)

tech = Agent(
    "You are a technical support engineer. Use search_docs to "
    "look up known issues. Quote the workaround verbatim.",
    model="gpt-4.1-mini",
    tools=[search_docs],
)

general = Agent(
    "You handle general inquiries. If you can't help, direct the "
    "user to the right team.",
    model="gpt-4.1-mini",
)


async def main() -> None:
    # Team.router is the ergonomic builder for the classify-and-
    # dispatch pattern. Equivalent to ``Agent(architecture=Router(
    # routes=..., fallback_route=..., require_confidence_above=...))``.
    agent = Team.router(
        routes=[
            RouterRoute(
                name="billing",
                agent=billing,
                description="Refunds, charges, subscription changes",
            ),
            RouterRoute(
                name="tech",
                agent=tech,
                description="API errors, integration bugs, downtime",
            ),
            RouterRoute(
                name="general",
                agent=general,
                description="Anything that doesn't fit billing or tech",
            ),
        ],
        instructions="You route customer support tickets to the right specialist.",
        model="gpt-4.1-mini",  # cheap classifier
        fallback_route="general",
        require_confidence_above=0.7,
    )

    queries = [
        "I was charged twice for my Pro plan in May. "
        "My email is anupam@example.com. Please refund the duplicate.",
        "I'm getting 502 errors from your API since this morning. "
        "What's the status?",
    ]

    for i, query in enumerate(queries, 1):
        print("=" * 70)
        print(f"QUERY {i}")
        print("=" * 70)
        print(f"User: {query}\n")

        async for ev in agent.stream(query):
            kind = ev.kind.value
            if kind == "model_chunk":
                chunk = ev.payload.get("chunk", {})
                if chunk.get("kind") == "text" and chunk.get("text"):
                    print(chunk["text"], end="", flush=True)
            elif kind == "tool_call":
                call = ev.payload.get("call", {})
                print(
                    f"\n  [tool] {call.get('tool')}({call.get('args')})"
                )
            elif kind == "tool_result":
                result = ev.payload.get("result", {})
                output = (result.get("output") or "")[:120]
                print(f"  [result] {output}")
            elif kind == "architecture_event":
                name = ev.payload.get("name", "")
                if name == "router.classified":
                    print(
                        f"[classifier] route="
                        f"{ev.payload.get('route')!r} "
                        f"confidence={ev.payload.get('confidence'):.2f}"
                    )
                elif name == "router.dispatched":
                    print(
                        f"[dispatched → {ev.payload.get('route')}]\n"
                    )
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
