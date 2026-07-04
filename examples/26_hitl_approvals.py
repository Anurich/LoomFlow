"""26_hitl_approvals.py — rich human-in-the-loop approvals.

Destructive tool calls hit the ``approval_handler`` before they run.
Handlers that return ``bool`` keep working (allow / deny), but
returning an :class:`ApprovalDecision` unlocks the interesting moves::

    ApprovalDecision(action="deny",  reason="prod DB — refused")
    ApprovalDecision(action="edit",  edited_args={...})   # fix, then run
    ApprovalDecision(action="remember_allow")  # cache for (user, tool)
    ApprovalDecision(action="remember_deny")   # ...deny stays fail-closed

* **deny** — the tool never runs; the reason lands in the transcript
  and the audit log.
* **edit** — the tool runs with YOUR args, not the model's. The audit
  trail records a ``tool_call_edited`` entry carrying both the edited
  and the original args.
* **remember_allow / remember_deny** — the decision is cached per
  ``(user_id, tool)`` for the remainder of the run, so the human is
  asked once, not once per call. The cache is per-run, never persisted.
* A raising handler is still a deny — approvals fail closed.

(There's also an interrupt/park pattern — the handler awaits
``runtime.wait_for_signal(...)`` and a caller unblocks it with
``agent.signal(session_id, "approval", {...})`` — see the
ApprovalDecision docstring.)

Runs OFFLINE with :class:`ScriptedModel` (no API key): the "model"
plans destructive ``delete_records`` calls; the handlers play the
human.

Run with::

    python examples/26_hitl_approvals.py
"""

from __future__ import annotations

from typing import Any

import anyio

from loomflow import (
    Agent,
    ApprovalDecision,
    InMemoryAuditLog,
    Mode,
    ScriptedModel,
    ScriptedTurn,
    StandardPermissions,
    ToolCall,
)

# ---------------------------------------------------------------------------
# A destructive tool. The scripted model marks its calls destructive,
# which routes them through the approval gate under Mode.DEFAULT.
# ---------------------------------------------------------------------------

DELETED: list[str] = []


def delete_records(table: str, where: str) -> str:
    """Delete rows from a table (pretend)."""
    DELETED.append(f"{table} WHERE {where}")
    return f"deleted from {table} where {where}"


def _turns(*calls: dict[str, Any], final: str = "done") -> list[ScriptedTurn]:
    return [
        ScriptedTurn(
            tool_calls=[ToolCall(tool="delete_records", args=dict(c), destructive=True)]
        )
        for c in calls
    ] + [ScriptedTurn(text=final)]


def _agent(model: ScriptedModel, handler: Any, **kwargs: Any) -> Agent:
    return Agent(
        "You are a database operator.",
        model=model,
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        tools=[delete_records],
        approval_handler=handler,
        **kwargs,
    )


def banner(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


async def main() -> None:
    # ---- 1. Deny, with a reason the model (and audit log) sees -------
    banner("Part 1 — deny with a reason")

    async def deny(call: ToolCall, user_id: str | None) -> ApprovalDecision:
        print(f"  [human] {user_id} wants {call.tool}({call.args}) → DENY")
        return ApprovalDecision(action="deny", reason="production table — refused")

    agent = _agent(
        ScriptedModel(_turns({"table": "orders", "where": "1=1"}, final="aborted")),
        deny,
    )
    result = await agent.run("Wipe the orders table.", user_id="alice")
    print(f"  final output:   {result.output!r}")
    print(f"  rows deleted:   {DELETED}  (nothing ran)")

    # ---- 2. Edit the args, then allow ---------------------------------
    banner("Part 2 — edit-args: fix the call instead of refusing it")

    audit = InMemoryAuditLog(secret="org-hmac-key")

    async def edit(call: ToolCall, user_id: str | None) -> ApprovalDecision:
        print(f"  [human] model planned: {call.args}")
        return ApprovalDecision(
            action="edit",
            edited_args={"table": "orders", "where": "created_at < '2020-01-01'"},
            reason="scoped the delete to stale rows",
        )

    agent = _agent(
        ScriptedModel(_turns({"table": "orders", "where": "1=1"})),
        edit,
        audit_log=audit,
    )
    result = await agent.run("Clean up old orders.", user_id="alice", session_id="edit-1")
    print(f"  final output:   {result.output!r}")
    print(f"  what actually ran: {DELETED}")

    # The audit trail carries BOTH versions of the call.
    entries = await audit.all_entries()
    print(f"  audit actions:  {[e.action for e in entries]}")
    edited = next(e for e in entries if e.action == "tool_call_edited")
    print(f"    tool_call_edited.original_args: {edited.payload['original_args']}")
    print(f"    tool_call_edited.args:          {edited.payload['args']}")

    # ---- 3. remember_allow: ask the human once, not N times -----------
    banner("Part 3 — remember_allow: one ask covers the whole run")

    asks: list[str] = []

    async def remember(call: ToolCall, user_id: str | None) -> ApprovalDecision:
        asks.append(call.tool)
        print(f"  [human] first {call.tool} this run → allow and REMEMBER")
        return ApprovalDecision(action="remember_allow")

    agent = _agent(
        ScriptedModel(
            _turns(
                {"table": "tmp_a", "where": "expired"},
                {"table": "tmp_b", "where": "expired"},
                {"table": "tmp_c", "where": "expired"},
            )
        ),
        remember,
    )
    before = len(DELETED)
    result = await agent.run("Purge the temp tables.", user_id="alice")
    print(f"  final output:      {result.output!r}")
    print(f"  destructive calls: {len(DELETED) - before} executed")
    print(f"  human asked:       {len(asks)} time(s)")
    print("  → The cache is per (user_id, tool) and per RUN — a new run")
    print("    asks again, and remember_deny stays fail-closed the same way.")


if __name__ == "__main__":
    anyio.run(main)
