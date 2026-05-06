"""16_swarm — Peer agents pass control via a handoff tool.

What it shows:
* The ``Swarm`` architecture lets peer ``Agent`` instances pass
  control to each other directly via ``handoff(target, message)``.
  No central supervisor; routing is decentralized.
* The handoff tool is injected per agent turn via the new
  ``Agent.run(extra_tools=...)`` primitive — the peers' static
  configuration is untouched.
* Cycle detection: the architecture watches recent handoffs and
  terminates if A→B→A→B is detected.
* ``max_handoffs`` caps total chain length so runaway loops bail
  cleanly.

⚠ Production caution
--------------------
The 2026 production literature flags Swarm as exploratory-only — it
has goal-drift and deadlock failure modes that hierarchical
:class:`Supervisor` doesn't. Use only for prototyping or
research-mode systems where flow can't be pre-specified.

Toy customer-support scenario: triage agent passes mixed billing +
tech queries to the right specialist; specialists answer directly.

Run:
    python examples/16_swarm.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    ScriptedModel,
    ScriptedTurn,
    Swarm,
)
from jeevesagent.core.types import ToolCall


def _final_agent(label: str, reply: str) -> Agent:
    return Agent(
        f"You are the {label} specialist.",
        model=ScriptedModel([ScriptedTurn(text=reply)]),
    )


async def main() -> None:
    # Triage agent: emits a handoff tool call to "billing", then
    # natural text saying it's passing along.
    triage = Agent(
        "You triage incoming queries.",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            tool="handoff",
                            args={
                                "target": "billing",
                                "message": (
                                    "billing query: charged twice "
                                    "last week"
                                ),
                            },
                        )
                    ]
                ),
                ScriptedTurn(
                    text="Routing to billing specialist."
                ),
            ]
        ),
    )

    billing = _final_agent(
        "billing",
        "Refund processed for the duplicate charge. Credit will "
        "land in 3-5 business days.",
    )
    tech = _final_agent(
        "tech",
        "Cleared the cache and reset the API key. Please retry.",
    )

    parent_model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "Customer support host.",
        model=parent_model,
        architecture=Swarm(
            agents={
                "triage": triage,
                "billing": billing,
                "tech": tech,
            },
            entry_agent="triage",
            max_handoffs=5,
            detect_cycles=True,
        ),
    )

    print("=== Streaming events ===")
    async for event in agent.stream(
        "I was charged twice last week."
    ):
        if event.kind != "architecture_event":
            continue
        name = event.payload.get("name", "")
        if name == "swarm.started":
            print(f"[started] entry={event.payload['entry_agent']}")
        elif name == "swarm.active":
            agent_name = event.payload["agent"]
            count = event.payload["handoff_count"]
            print(f"[active @ count={count}] {agent_name}")
        elif name == "swarm.handoff":
            f = event.payload["from_agent"]
            t = event.payload["to_agent"]
            msg = event.payload.get("message", "")
            print(f"[handoff] {f} → {t}: {msg[:60]}")
        elif name == "swarm.cycle_detected":
            print("[cycle detected — bailing]")
        elif name == "swarm.max_handoffs":
            count = event.payload["handoffs"]
            print(f"[max_handoffs reached @ count={count}]")
        elif name == "swarm.completed":
            agent_name = event.payload["agent"]
            print(f"[completed by {agent_name}]")

    # Re-run with fresh agents for the final answer print.
    fresh_triage = Agent(
        "triage",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="c",
                            tool="handoff",
                            args={"target": "billing"},
                        )
                    ]
                ),
                ScriptedTurn(text="passing along"),
            ]
        ),
    )
    fresh_billing = _final_agent("billing", "Refund processed.")
    fresh_agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=Swarm(
            agents={
                "triage": fresh_triage,
                "billing": fresh_billing,
            },
            entry_agent="triage",
        ),
    )
    result = await fresh_agent.run(
        "I was charged twice last week."
    )
    print(f"\n=== Final answer ===\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
