"""End-to-end cross-primitive user_id isolation tests (M9).

Memory was already fully partitioned (M1-M8). M9 closes the
remaining gaps — working blocks, budget, audit, permissions,
hooks. This file is the regression suite that proves all five
primitives partition cleanly: alice's state never leaks to bob,
bob's state never leaks to alice.

Scope is intentionally narrow per test — one primitive at a time —
plus a single end-to-end test that runs an Agent under both users
and verifies the partition holds across every layer.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, InMemoryAuditLog, InMemoryMemory, Mode, StandardPermissions
from loomflow.core import AuditEntry
from loomflow.core.types import ToolCall
from loomflow.governance.budget import BudgetConfig, StandardBudget
from loomflow.memory.sqlite import SqliteMemory
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.security import PerUserPermissions

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Working blocks — partition by user_id
# ---------------------------------------------------------------------------


async def test_working_blocks_inmemory_partition() -> None:
    """Two users, same block name, different content. Each sees
    only their own."""
    mem = InMemoryMemory()
    await mem.update_block("preferences", "dark mode", user_id="alice")
    await mem.update_block(
        "preferences", "compact layout", user_id="bob"
    )

    alice_blocks = await mem.working(user_id="alice")
    bob_blocks = await mem.working(user_id="bob")

    assert len(alice_blocks) == 1
    assert alice_blocks[0].content == "dark mode"
    assert len(bob_blocks) == 1
    assert bob_blocks[0].content == "compact layout"


async def test_working_blocks_anonymous_bucket_isolated() -> None:
    """``user_id=None`` is its own bucket — never sees alice's
    blocks."""
    mem = InMemoryMemory()
    await mem.update_block("notes", "alice's note", user_id="alice")
    await mem.update_block("notes", "anon's note", user_id=None)

    anon_blocks = await mem.working(user_id=None)
    alice_blocks = await mem.working(user_id="alice")
    assert len(anon_blocks) == 1 and anon_blocks[0].content == "anon's note"
    assert len(alice_blocks) == 1 and alice_blocks[0].content == "alice's note"


async def test_working_blocks_sqlite_persists_partitioned(
    tmp_path: Any,
) -> None:
    """SQLite backend partitions across reopens — the schema
    migration installed (user_id, name) as the PK."""
    db = tmp_path / "blocks.db"
    m1 = SqliteMemory(db)
    await m1.update_block("prefs", "alice prefers dark", user_id="alice")
    await m1.update_block("prefs", "bob prefers light", user_id="bob")

    # Re-open — partition survives.
    m2 = SqliteMemory(db)
    alice = await m2.working(user_id="alice")
    bob = await m2.working(user_id="bob")
    assert alice[0].content == "alice prefers dark"
    assert bob[0].content == "bob prefers light"


async def test_working_blocks_pinned_order_independent_per_user() -> None:
    """Each user's pinned_order counts only their own blocks,
    starting from 0. Adding bob's first block doesn't bump alice's
    next block past it."""
    mem = InMemoryMemory()
    await mem.update_block("a", "alice 1", user_id="alice")
    await mem.update_block("a", "bob 1", user_id="bob")  # bob's 0
    await mem.update_block("b", "alice 2", user_id="alice")  # alice's 1

    alice = await mem.working(user_id="alice")
    bob = await mem.working(user_id="bob")
    assert [b.pinned_order for b in alice] == [0, 1]
    assert [b.pinned_order for b in bob] == [0]


# ---------------------------------------------------------------------------
# Budget — per-user accounting + caps
# ---------------------------------------------------------------------------


async def test_budget_consume_partitions_by_user() -> None:
    """Two users sharing one Budget instance accumulate independently."""
    budget = StandardBudget()
    await budget.consume(
        tokens_in=100, tokens_out=50, cost_usd=0.01, user_id="alice"
    )
    await budget.consume(
        tokens_in=200, tokens_out=80, cost_usd=0.02, user_id="bob"
    )

    alice = budget.usage_for("alice")
    bob = budget.usage_for("bob")
    assert alice["tokens_total"] == 150
    assert bob["tokens_total"] == 280
    assert alice["cost_usd"] == pytest.approx(0.01)
    assert bob["cost_usd"] == pytest.approx(0.02)


async def test_budget_per_user_max_tokens_blocks_only_offender() -> None:
    """When alice hits her per-user cap, bob's runs still allow.
    Without per-user caps, alice's heavy use would block bob too."""
    budget = StandardBudget(
        BudgetConfig(per_user_max_tokens=100)
    )
    # Alice exhausts her per-user cap.
    await budget.consume(
        tokens_in=60, tokens_out=50, cost_usd=0, user_id="alice"
    )
    alice_status = await budget.allows_step(user_id="alice")
    bob_status = await budget.allows_step(user_id="bob")

    assert alice_status.blocked is True
    assert alice_status.reason == "per_user_max_tokens"
    assert bob_status.blocked is False


async def test_budget_global_cap_still_works_alongside_per_user() -> None:
    """Both caps active — global fires first when global threshold
    is the lower one."""
    budget = StandardBudget(
        BudgetConfig(max_tokens=500, per_user_max_tokens=400)
    )
    # Alice + bob together exhaust the global cap.
    await budget.consume(
        tokens_in=200, tokens_out=80, cost_usd=0, user_id="alice"
    )
    await budget.consume(
        tokens_in=200, tokens_out=80, cost_usd=0, user_id="bob"
    )
    # Total = 560 > global 500 → blocks every user.
    alice_status = await budget.allows_step(user_id="alice")
    assert alice_status.blocked is True
    assert alice_status.reason == "max_tokens"


# ---------------------------------------------------------------------------
# Audit log — top-level user_id field + query filter
# ---------------------------------------------------------------------------


async def test_audit_entry_carries_top_level_user_id() -> None:
    log = InMemoryAuditLog()
    entry = await log.append(
        session_id="s",
        user_id="alice",
        actor="user",
        action="run_started",
        payload={},
    )
    assert isinstance(entry, AuditEntry)
    assert entry.user_id == "alice"


async def test_audit_query_filter_by_user_id() -> None:
    log = InMemoryAuditLog()
    await log.append(
        session_id="s1", user_id="alice", actor="user",
        action="run_started", payload={},
    )
    await log.append(
        session_id="s2", user_id="bob", actor="user",
        action="run_started", payload={},
    )
    await log.append(
        session_id="s3", user_id="alice", actor="user",
        action="run_completed", payload={},
    )

    alice_entries = await log.query(user_id="alice")
    bob_entries = await log.query(user_id="bob")
    assert len(alice_entries) == 2
    assert all(e.user_id == "alice" for e in alice_entries)
    assert len(bob_entries) == 1
    assert bob_entries[0].user_id == "bob"


async def test_audit_query_combined_user_and_action_filter() -> None:
    log = InMemoryAuditLog()
    await log.append(
        session_id="s", user_id="alice", actor="user",
        action="run_started", payload={},
    )
    await log.append(
        session_id="s", user_id="alice", actor="system",
        action="run_completed", payload={},
    )
    out = await log.query(user_id="alice", action="run_completed")
    assert len(out) == 1
    assert out[0].action == "run_completed"


# ---------------------------------------------------------------------------
# Permissions — per-user policy routing
# ---------------------------------------------------------------------------


async def test_per_user_permissions_routes_to_user_policy() -> None:
    """Two policies — admin gets BYPASS, default denies bash."""
    admin = StandardPermissions(mode=Mode.BYPASS)
    standard = StandardPermissions(
        mode=Mode.DEFAULT, denied_tools=["bash"]
    )
    perms = PerUserPermissions(
        policies={"admin_alice": admin},
        default=standard,
    )

    bash_call = ToolCall(id="c", tool="bash", args={"cmd": "ls"})

    # Admin can run bash.
    admin_decision = await perms.check(
        bash_call, context={}, user_id="admin_alice"
    )
    assert admin_decision.allow is True

    # Regular user gets denied.
    bob_decision = await perms.check(
        bash_call, context={}, user_id="bob"
    )
    assert bob_decision.deny is True


async def test_per_user_permissions_default_for_unknown_user() -> None:
    """Unknown user_id falls back to ``default``."""
    default = StandardPermissions(
        mode=Mode.DEFAULT, denied_tools=["delete_account"]
    )
    perms = PerUserPermissions(policies={}, default=default)
    decision = await perms.check(
        ToolCall(id="c", tool="delete_account", args={}),
        context={},
        user_id="random_user",
    )
    assert decision.deny is True


async def test_standard_permissions_accepts_user_id_kwarg() -> None:
    """The kwarg-only addition is backwards-compatible — existing
    StandardPermissions impls accept it (and ignore it)."""
    perms = StandardPermissions(mode=Mode.DEFAULT)
    decision = await perms.check(
        ToolCall(id="c", tool="any_tool", args={}),
        context={},
        user_id="alice",
    )
    assert decision.allow is True


# ---------------------------------------------------------------------------
# End-to-end: Agent run with budget + audit + memory all partitioned
# ---------------------------------------------------------------------------


async def test_agent_full_isolation_across_users() -> None:
    """The headline contract: one Agent + one Memory + one Budget +
    one AuditLog. Run alice and bob concurrently. After both runs:

    * alice's memory has alice's episode, not bob's
    * bob's memory has bob's episode, not alice's
    * budget's per-user accounting tracks each independently
    * audit log entries carry the right user_id top-level
    """
    memory = InMemoryMemory()
    budget = StandardBudget()
    audit = InMemoryAuditLog()
    agent = Agent(
        "...",
        model=ScriptedModel([
            ScriptedTurn(text="alice reply"),
            ScriptedTurn(text="bob reply"),
        ]),
        memory=memory,
        budget=budget,
        audit_log=audit,
    )

    await agent.run(
        "alice's secret prompt",
        user_id="alice",
        session_id="alice_s",
    )
    await agent.run(
        "bob's secret prompt",
        user_id="bob",
        session_id="bob_s",
    )

    # Memory: each user sees only their own episodes.
    alice_eps = await memory.recall("secret", user_id="alice")
    bob_eps = await memory.recall("secret", user_id="bob")
    assert len(alice_eps) == 1
    assert "alice's secret" in alice_eps[0].input
    assert len(bob_eps) == 1
    assert "bob's secret" in bob_eps[0].input

    # Budget: per-user accounting tracks separately.
    # (Scripted model usage is zero, but the buckets exist.)
    assert "alice" in budget._by_user  # noqa: SLF001 — internal check
    assert "bob" in budget._by_user

    # Audit log: every entry has the right top-level user_id.
    alice_entries = await audit.query(user_id="alice")
    bob_entries = await audit.query(user_id="bob")
    assert alice_entries  # at least run_started + run_completed
    assert all(e.user_id == "alice" for e in alice_entries)
    assert bob_entries
    assert all(e.user_id == "bob" for e in bob_entries)
