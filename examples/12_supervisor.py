"""12_supervisor — Workers + a delegate(...) tool, parallel for free.

What it shows:
* The ``Supervisor`` architecture wraps a base (default ``ReAct``)
  with one extra tool: ``delegate(worker, instructions)``. When
  the supervising model emits a ``delegate`` call, the named worker
  ``Agent`` runs to completion and returns its output as the tool
  result.
* Parallel delegation comes free: ``ReAct``'s tool dispatch is
  already a structured task group, so two ``delegate`` calls in
  one supervisor turn → both workers run concurrently.
* Workers are full ``Agent`` instances. They can be any architecture
  themselves (Reflexion-wrapped, DeepAgent-wrapped, etc.).
* Each worker invocation gets a fresh session id, so the same
  worker can be called twice in one turn without journal collisions.

A research-team supervisor with a researcher and a coder. The
supervisor delegates to both in one turn (parallel), then synthesizes
their outputs.

Run:
    python examples/12_supervisor.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    ScriptedModel,
    ScriptedTurn,
    Supervisor,
)
from jeevesagent.core.types import ToolCall


def _worker(label: str, reply: str) -> Agent:
    return Agent(
        f"You are the {label} specialist.",
        model=ScriptedModel([ScriptedTurn(text=reply)]),
    )


async def main() -> None:
    researcher = _worker(
        "researcher",
        "Found three papers: Yao et al. 2022 (ReAct), Madaan et al. "
        "2023 (Self-Refine), Shinn et al. 2023 (Reflexion).",
    )
    coder = _worker(
        "coder",
        "Implemented `class Architecture(Protocol)` with `name`, "
        "`run`, and `declared_workers`.",
    )

    # The supervisor's model:
    # Turn 1: emit TWO delegate calls in one turn → parallel
    #         (research + coding run concurrently).
    # Turn 2: synthesize the two results into a final answer.
    supervisor_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={
                            "worker": "researcher",
                            "instructions": (
                                "Find the seminal papers on agent "
                                "loop architectures."
                            ),
                        },
                    ),
                    ToolCall(
                        id="c2",
                        tool="delegate",
                        args={
                            "worker": "coder",
                            "instructions": (
                                "Implement an Architecture protocol "
                                "in Python."
                            ),
                        },
                    ),
                ]
            ),
            ScriptedTurn(
                text=(
                    "I found the seminal papers (ReAct, Self-Refine, "
                    "Reflexion) and implemented an Architecture "
                    "protocol class with the three required methods. "
                    "Both pieces are ready."
                )
            ),
        ]
    )

    agent = Agent(
        "You are a research lead managing a small team.",
        model=supervisor_model,
        architecture=Supervisor(
            workers={
                "researcher": researcher,
                "coder": coder,
            }
        ),
    )

    print("=== Streaming events ===")
    delegations: list[str] = []
    async for event in agent.stream(
        "Build a reference implementation of the Architecture protocol "
        "with citations to the relevant literature."
    ):
        if event.kind == "tool_call":
            call = event.payload.get("call", {})
            if call.get("tool") == "delegate":
                worker = call.get("args", {}).get("worker", "?")
                delegations.append(worker)
                print(f"[delegating] → {worker}")
        elif event.kind == "tool_result":
            result = event.payload.get("result", {})
            output = (result.get("output") or "")[:80]
            print(f"[worker returned] {output}...")
        elif event.kind == "architecture_event":
            name = event.payload.get("name", "")
            if name == "supervisor.workers_ready":
                print(
                    f"[workers ready] {event.payload['workers']}"
                )
            elif name == "supervisor.completed":
                print("[supervisor completed]")

    print(f"\nDelegated to: {delegations}")

    # Re-run for the final RunResult.
    supervisor_model_2 = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={
                            "worker": "researcher",
                            "instructions": "research it",
                        },
                    ),
                    ToolCall(
                        id="c2",
                        tool="delegate",
                        args={
                            "worker": "coder",
                            "instructions": "code it",
                        },
                    ),
                ]
            ),
            ScriptedTurn(
                text=(
                    "Both deliverables are ready: "
                    "(a) literature review, (b) reference impl."
                )
            ),
        ]
    )
    agent2 = Agent(
        "You are a research lead.",
        model=supervisor_model_2,
        architecture=Supervisor(
            workers={
                "researcher": _worker("researcher", "research done"),
                "coder": _worker("coder", "code done"),
            }
        ),
    )
    result = await agent2.run(
        "Build a reference implementation with citations."
    )
    print(f"\n=== Final answer ===\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
