"""11_router — Classify input, dispatch to one specialist Agent.

What it shows:
* The ``Router`` architecture is the cheapest multi-agent pattern.
  One small classifier call decides which specialist owns the
  input; the chosen specialist runs to completion. No cross-
  specialist synthesis.
* Each route is a fully-constructed ``Agent`` — its own model,
  memory, tools, architecture. Specialists are independent.
* Confidence threshold + ``fallback_route`` handle ambiguous
  inputs gracefully — the classifier emits a confidence number,
  and Router uses the fallback when it's below your threshold.
* Replay-safe: each specialist invocation gets a deterministic
  session id (``{parent}__route_{name}``).

A customer-support intent router with billing / tech / general
specialists. ``ScriptedModel`` simulates the classifier and each
specialist; in production each would be a real ``Agent`` with a
real model.

Run:
    python examples/11_router.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    Router,
    RouterRoute,
    ScriptedModel,
    ScriptedTurn,
)


def _specialist(label: str, reply: str) -> Agent:
    """Build a specialist Agent that produces a fixed text response."""
    return Agent(
        f"You are the {label} specialist.",
        model=ScriptedModel([ScriptedTurn(text=reply)]),
    )


async def main() -> None:
    # Build three specialists. Each has its own model in production
    # (a small fast model is fine for routing-receivers).
    billing = _specialist(
        "billing",
        "Refund processed. You'll see the credit within 3-5 days.",
    )
    tech = _specialist(
        "tech",
        "Try clearing your cache and reloading. The 502 is a "
        "transient gateway issue we're tracking.",
    )
    general = _specialist(
        "general",
        "Thanks for reaching out — let me find someone to help.",
    )

    # The classifier is a model on the parent Agent. In production
    # use a small fast model like Haiku or 4o-mini for cheap
    # classification.
    classifier_model = ScriptedModel(
        [
            ScriptedTurn(text="route: billing\nconfidence: 0.95"),
        ]
    )
    agent = Agent(
        "You route customer support tickets to the right specialist.",
        model=classifier_model,
        architecture=Router(
            routes=[
                RouterRoute(
                    name="billing",
                    agent=billing,
                    description=(
                        "Refunds, charges, plan changes, invoices."
                    ),
                ),
                RouterRoute(
                    name="tech",
                    agent=tech,
                    description=(
                        "API errors, downtime, integration problems, "
                        "SDK bugs."
                    ),
                ),
                RouterRoute(
                    name="general",
                    agent=general,
                    description="Anything that doesn't fit elsewhere.",
                ),
            ],
            fallback_route="general",
            require_confidence_above=0.7,
        ),
    )

    print("=== Streaming events ===")
    async for event in agent.stream("I was charged twice last week."):
        if event.kind == "architecture_event":
            name = event.payload.get("name", "")
            if name == "router.classified":
                route = event.payload["route"]
                conf = event.payload["confidence"]
                print(
                    f"[classified] route='{route}' "
                    f"confidence={conf:.2f}"
                )
            elif name == "router.dispatched":
                print(f"[dispatched] → {event.payload['route']}")
            elif name == "router.completed":
                turns = event.payload.get("specialist_turns", "?")
                print(
                    f"[completed] specialist used {turns} turn(s)"
                )

    # Run again to print the final answer cleanly.
    classifier_model_2 = ScriptedModel(
        [ScriptedTurn(text="route: billing\nconfidence: 0.95")]
    )
    billing2 = _specialist(
        "billing",
        "Refund processed. Credit lands in 3-5 days.",
    )
    agent2 = Agent(
        "You route customer support tickets.",
        model=classifier_model_2,
        architecture=Router(
            routes=[
                RouterRoute(name="billing", agent=billing2),
                RouterRoute(
                    name="general", agent=_specialist("general", "...")
                ),
            ],
            fallback_route="general",
        ),
    )
    result = await agent2.run("I was charged twice last week.")
    print(f"\n=== Final answer ===\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
