"""Example 6 — Linear workflow chain (no LLM required).

The simplest workflow shape: a list of steps that run in order,
each step's output feeding the next. No model adapters, no API
keys — pure framework plumbing — so this is the lowest-friction
way to feel out the API.

Three things this example demonstrates:

* :class:`~jeevesagent.Workflow.chain` — sugar constructor for a
  linear sequence. Each entry can be an ``async def``, a sync
  function, an ``Agent``, or another ``Workflow``; the framework
  coerces.
* The ``user_id`` partition flowing through every step — set once
  on ``run`` and visible inside steps via ``get_run_context()``.
* :class:`~jeevesagent.WorkflowResult` — ``output``, ``visited``,
  and ``per_step`` for inspecting what happened.

Run::

    python examples/06_workflow_chain.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import Workflow
from jeevesagent.core.context import get_run_context

# Each step is a plain async function. Inputs flow step-to-step:
# the value returned by ``parse_request`` is passed to ``validate``,
# whose return is passed to ``process``, and so on.

async def parse_request(raw: str) -> dict[str, str]:
    """Pretend the input is a CSV-shaped customer request."""
    fields = [p.strip() for p in raw.split(",")]
    return {
        "user_id": fields[0] if len(fields) > 0 else "",
        "intent": fields[1] if len(fields) > 1 else "",
        "amount": fields[2] if len(fields) > 2 else "",
    }


async def validate(state: dict[str, str]) -> dict[str, str]:
    """Reject malformed inputs early."""
    if not state.get("user_id"):
        raise ValueError("missing user_id in request")
    state["validated"] = "true"
    return state


async def process(state: dict[str, str]) -> dict[str, str]:
    """The real work — illustrative; in production this would
    hit a database, billing system, etc. We also peek at the live
    RunContext to show how user_id flows in from Workflow.run."""
    ctx = get_run_context()
    state["processed_for_session"] = ctx.session_id or "(no session)"
    state["processed_amount"] = state["amount"]
    return state


async def format_receipt(state: dict[str, str]) -> str:
    """Final step — produce a string the caller can show."""
    return (
        f"Receipt: user={state['user_id']} "
        f"intent={state['intent']} "
        f"amount={state['processed_amount']} "
        f"(session={state['processed_for_session']})"
    )


async def main() -> None:
    print("\n  Example 6 — Linear workflow chain\n")

    wf = Workflow.chain(
        [parse_request, validate, process, format_receipt],
        name="customer-request",
    )

    raw_input = "alice, refund, 49.00"
    print(f"  input  : {raw_input!r}")

    result = await wf.run(
        raw_input, user_id="alice", session_id="req-2026-001"
    )

    print(f"  output : {result.output}")
    print(f"  visited: {' → '.join(result.visited)}")
    print()

    # ``per_step`` exposes each step's intermediate output without
    # re-running anything — useful for debugging or asserting in
    # tests.
    print("  Per-step trace:")
    for node, value in result.per_step.items():
        rendered = value if isinstance(value, str) else dict(value)
        print(f"    [{node:>15}] {rendered}")


if __name__ == "__main__":
    asyncio.run(main())
