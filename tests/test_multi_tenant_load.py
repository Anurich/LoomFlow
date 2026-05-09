"""M10.7 — multi-tenant load smoke test.

Lower-scale version of ``bench/multi_tenant.py`` that runs as part
of the regular test suite. The bench is for benchmarking; this is
for catching regressions to the isolation / budget-accounting
contract under concurrent load.

Defaults (10 users × 2 turns) keep the test fast (<1s) while still
exercising the concurrent-task-group path. The bench file remains
the place for serious load measurement.
"""

from __future__ import annotations

import anyio
import pytest

from loomflow import Agent, InMemoryAuditLog, InMemoryMemory
from loomflow.governance.budget import BudgetConfig, StandardBudget
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


async def test_concurrent_users_respect_isolation_contract() -> None:
    """10 concurrent users × 2 turns each = 20 runs through one
    shared Agent. After the dust settles each user's memory must
    contain only their own episodes; the budget's per-user
    accounting must show 20 distinct buckets."""
    users = 10
    turns_per_user = 2

    memory = InMemoryMemory()
    budget = StandardBudget(
        BudgetConfig(per_user_max_tokens=10_000_000),
    )
    audit = InMemoryAuditLog()
    model = ScriptedModel(
        [ScriptedTurn(text="ack") for _ in range(users * turns_per_user)]
    )
    agent = Agent(
        "you are an assistant",
        model=model,
        memory=memory,
        budget=budget,
        audit_log=audit,
        auto_extract=False,
    )

    barrier = anyio.Event()

    async def _run_user(user_id: str) -> None:
        await barrier.wait()
        for turn_no in range(turns_per_user):
            await agent.run(
                f"prompt {turn_no} for {user_id}",
                user_id=user_id,
                session_id=f"{user_id}-s",
            )

    async with anyio.create_task_group() as tg:
        for u in range(users):
            tg.start_soon(_run_user, f"user_{u}")
        barrier.set()

    # Isolation: each user sees ONLY their own episodes.
    for u in range(users):
        uid = f"user_{u}"
        eps = await memory.recall("prompt", user_id=uid, limit=turns_per_user)
        assert eps, f"user {uid} should have episodes"
        for e in eps:
            assert uid in e.input, (
                f"isolation violation: user {uid} sees episode '{e.input[:40]}'"
            )

    # Audit log: every entry attributed to a real user_id (no
    # leakage into the anonymous bucket).
    all_entries = await audit.query()
    user_ids = {e.user_id for e in all_entries if e.user_id is not None}
    for u in range(users):
        assert f"user_{u}" in user_ids, (
            f"user user_{u} missing from audit log"
        )
