"""Example 12 — Audit log (HMAC-signed, durable on disk).

Every primitive in Loom writes audit entries when an
:class:`~loomflow.AuditLog` is configured: ``run_started`` /
``run_completed`` / ``tool_call`` / ``tool_result`` on Agents,
``step_started`` / ``step_completed`` / ``step_failed`` on
Workflows. Entries carry the live ``user_id`` for multi-tenant
queries, are HMAC-signed for tamper detection, and a JSONL file
backend recovers the highest sequence number across process
restarts.

This example builds an Agent + a Workflow with a shared
``FileAuditLog``, runs both, then inspects what was written —
including HMAC verification.

Three things this example demonstrates:

* Two backends behind the same Protocol — :class:`InMemoryAuditLog`
  for tests / notebooks, :class:`FileAuditLog` for production.
* Per-entry HMAC signatures + the public ``verify_signature``
  helper — a tampered entry will not verify.
* ``user_id`` is a first-class filter on ``query`` — multi-
  tenant audit queries without payload-digging.

No API keys required — uses :class:`EchoModel` so the example
runs anywhere.

Run::

    python examples/12_audit_log.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import anyio

from loomflow import Agent, EchoModel, Workflow, tool
from loomflow.security import FileAuditLog, InMemoryAuditLog
from loomflow.security.audit import verify_signature

# ---------------------------------------------------------------------------
# A pretend tool — just to produce ``tool_call`` / ``tool_result``
# audit entries when the agent is asked to fetch the weather.
# ---------------------------------------------------------------------------


@tool
async def get_weather(city: str) -> str:
    """Look up the (fake) current weather for a city."""
    return f"It's sunny and 72°F in {city}."


async def main() -> None:
    # ---- 1. In-memory backend (fast; tests / notebooks) -----------------
    print("=" * 60)
    print("Part 1 — InMemoryAuditLog (fast, ephemeral)")
    print("=" * 60)

    in_mem_audit = InMemoryAuditLog(secret="my-org-hmac-key")
    agent = Agent(
        "You are a helpful assistant.",
        model=EchoModel(),
        tools=[get_weather],
        audit_log=in_mem_audit,
    )

    await agent.run("Hello", user_id="alice", session_id="conv-1")
    await agent.run("Hi there", user_id="bob", session_id="conv-2")

    # Every entry carries user_id — query is partitioned, not
    # "scan the payload yourself".
    alice_entries = await in_mem_audit.query(user_id="alice")
    bob_entries = await in_mem_audit.query(user_id="bob")
    print(f"  Alice's audit entries: {len(alice_entries)}")
    print(f"  Bob's audit entries:   {len(bob_entries)}")
    print(f"  Sample action chain:   "
          f"{[e.action for e in alice_entries]}")

    # ---- 2. File backend (durable; JSONL on disk) -----------------------
    print()
    print("=" * 60)
    print("Part 2 — FileAuditLog (durable, JSONL)")
    print("=" * 60)

    log_path = Path("./_audit_log_demo.jsonl")
    # Sync pathlib ops dispatched to a worker thread — ruff's
    # ASYNC240 catches direct .exists() / .unlink() inside async
    # functions; this is the idiomatic anyio fix.
    if await anyio.to_thread.run_sync(log_path.exists):
        await anyio.to_thread.run_sync(log_path.unlink)

    file_audit = FileAuditLog(log_path, secret="my-org-hmac-key")

    # Wrap a Workflow with the file audit log — every step gets
    # ``step_started`` / ``step_completed`` entries automatically.
    async def step_a(x: str) -> str:
        return x.upper()

    async def step_b(x: str) -> str:
        return f"{x}!"

    wf = Workflow.chain([step_a, step_b], audit_log=file_audit)
    result = await wf.run("hello", user_id="alice")
    print(f"  Workflow result: {result.output!r}")

    # Read back the JSONL.
    size = await anyio.to_thread.run_sync(lambda: log_path.stat().st_size)
    print(f"  Wrote {size} bytes to {log_path}")
    file_entries = await file_audit.query(user_id="alice")
    print(f"  Entries written: {len(file_entries)}")
    print(f"  Actions:         {[e.action for e in file_entries]}")

    # ---- 3. HMAC signature verification ----------------------------------
    print()
    print("=" * 60)
    print("Part 3 — Tamper detection via HMAC")
    print("=" * 60)

    sample = file_entries[0]
    is_valid = verify_signature(sample, secret="my-org-hmac-key")
    print(f"  Original entry verifies:    {is_valid}")

    # Now mutate the payload by adding a key not already present.
    # The signature is over the canonical serialised body — any
    # tampering (additions / changes / removals) breaks verification.
    # NB: spread order matters when constructing the new payload.
    # ``{**sample.payload, "extra": True}`` adds a NEW key;
    # ``{"extra": True, **sample.payload}`` would be silently
    # overwritten back to the original by the spread.
    tampered = sample.model_copy(
        update={"payload": {**sample.payload, "tampered": True}}
    )
    tampered_valid = verify_signature(tampered, secret="my-org-hmac-key")
    print(f"  Tampered entry verifies:    {tampered_valid} "
          f"← False detects the change")

    # Different secret = different signature, also fails.
    wrong_secret_valid = verify_signature(sample, secret="wrong-key")
    print(f"  Wrong-secret check passes:  {wrong_secret_valid} "
          f"← catches secret-rotation mistakes")

    # ---- 4. Persistence across process restart ---------------------------
    print()
    print("=" * 60)
    print("Part 4 — Restart recovery (seq counter persists)")
    print("=" * 60)

    # Pretend the process restarted: build a fresh FileAuditLog
    # against the same path. It scans the existing file and
    # resumes the seq counter — new entries don't collide.
    restarted = FileAuditLog(log_path, secret="my-org-hmac-key")
    before = (await restarted.query(user_id="alice"))[-1].seq
    print(f"  Highest seq before new run:  {before}")

    wf2 = Workflow.chain([step_a, step_b], audit_log=restarted)
    await wf2.run("world", user_id="alice")
    after = (await restarted.query(user_id="alice"))[-1].seq
    print(f"  Highest seq after new run:   {after}")
    print(f"  Δ = {after - before} new entries appended, "
          f"no seq collisions.")

    # Cleanup so re-running the example is idempotent.
    await anyio.to_thread.run_sync(log_path.unlink)


if __name__ == "__main__":
    asyncio.run(main())
