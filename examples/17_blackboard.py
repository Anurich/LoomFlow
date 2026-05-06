"""17_blackboard — Coordinator + agents share a state board.

What it shows:
* The ``BlackboardArchitecture`` orchestrates a team via a shared
  blackboard. A coordinator (LLM Agent) reads the current state and
  decides which agent contributes next; each contribution is posted
  back to the board; an optional decider synthesizes the final.
* Without a coordinator, falls back to round-robin selection
  (useful for smoke tests / prototyping).
* Without a decider, falls back to "last contribution wins" or
  "last answer-kind entry."
* Coordinator output is parsed as JSON
  (``{"terminate": ..., "next_agent": ..., "instruction": ...}``)
  with markdown-fence stripping and a safe-default on parse
  failure.

Toy data-discovery scenario: hypothesis + evidence + critic agents
take turns; coordinator picks who; decider synthesizes.

Run:
    python examples/17_blackboard.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    BlackboardArchitecture,
    ScriptedModel,
    ScriptedTurn,
)


def _scripted_agent(label: str, replies: list[str]) -> Agent:
    return Agent(
        f"You are the {label} agent.",
        model=ScriptedModel([ScriptedTurn(text=r) for r in replies]),
    )


async def main() -> None:
    # Three specialist agents take turns when the coordinator picks them.
    hypothesis = _scripted_agent(
        "hypothesis",
        [
            (
                "Hypothesis: 30% YoY growth driven primarily by "
                "expansion into mid-market accounts."
            )
        ],
    )
    evidence = _scripted_agent(
        "evidence",
        [
            (
                "Evidence: mid-market segment (51-500 employees) "
                "grew 47% vs 18% in enterprise. Data: "
                "revenue/segment.csv."
            )
        ],
    )
    critic = _scripted_agent(
        "critic",
        [
            (
                "Critic: hypothesis well-supported but doesn't "
                "explain enterprise softness. Suggest digging into "
                "churn data."
            )
        ],
    )

    # Coordinator picks each agent in turn, then terminates.
    coordinator = _scripted_agent(
        "coordinator",
        [
            (
                '{"terminate": false, "next_agent": "hypothesis", '
                '"instruction": "propose a hypothesis"}'
            ),
            (
                '{"terminate": false, "next_agent": "evidence", '
                '"instruction": "find supporting evidence"}'
            ),
            (
                '{"terminate": false, "next_agent": "critic", '
                '"instruction": "challenge the analysis"}'
            ),
            (
                '{"terminate": true, "next_agent": null, '
                '"instruction": null}'
            ),
        ],
    )
    decider = _scripted_agent(
        "decider",
        [
            (
                "FINAL: The 30% YoY growth is driven by mid-market "
                "expansion (47% growth), evidenced by segment-level "
                "revenue data. The critic's call to investigate "
                "enterprise churn is well-taken — that's a follow-up."
            )
        ],
    )

    parent_model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "Data-discovery analyst",
        model=parent_model,
        architecture=BlackboardArchitecture(
            agents={
                "hypothesis": hypothesis,
                "evidence": evidence,
                "critic": critic,
            },
            coordinator=coordinator,
            decider=decider,
            max_rounds=8,
        ),
    )

    print("=== Streaming events ===")
    async for event in agent.stream(
        "What's driving our 30% YoY growth?"
    ):
        if event.kind != "architecture_event":
            continue
        name = event.payload.get("name", "")
        if name == "blackboard.started":
            agents = event.payload["agents"]
            print(f"[started] agents={agents}")
        elif name == "blackboard.coordinator_decided":
            r = event.payload["round"]
            terminate = event.payload["terminate"]
            picked = event.payload.get("next_agent")
            if terminate:
                print(f"[round {r}] coordinator → terminate")
            else:
                print(f"[round {r}] coordinator → {picked}")
        elif name == "blackboard.contribution":
            r = event.payload["round"]
            ag = event.payload["agent"]
            content = event.payload["content"]
            print(f"  [{ag} @ r{r}] {content[:80]}...")
        elif name == "blackboard.completed":
            board_size = event.payload["board_size"]
            print(f"\n[completed; board has {board_size} entries]")

    # Re-run with fresh agents for final answer.
    fresh_hyp = _scripted_agent("hypothesis", ["mid-market hypothesis"])
    fresh_ev = _scripted_agent("evidence", ["evidence found"])
    fresh_critic = _scripted_agent("critic", ["concerns noted"])
    fresh_coord = _scripted_agent(
        "coordinator",
        [
            '{"terminate": false, "next_agent": "hypothesis", "instruction": ""}',
            '{"terminate": false, "next_agent": "evidence", "instruction": ""}',
            '{"terminate": false, "next_agent": "critic", "instruction": ""}',
            '{"terminate": true, "next_agent": null, "instruction": null}',
        ],
    )
    fresh_decider = _scripted_agent(
        "decider", ["Final synthesized answer about mid-market growth."]
    )
    fresh_agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=BlackboardArchitecture(
            agents={
                "hypothesis": fresh_hyp,
                "evidence": fresh_ev,
                "critic": fresh_critic,
            },
            coordinator=fresh_coord,
            decider=fresh_decider,
            max_rounds=8,
        ),
    )
    result = await fresh_agent.run("growth analysis")
    print(f"\n=== Final answer ===\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
