"""M10.1 — bounded per-user state across primitives.

The framework is multi-tenant by default; without bounds, the
per-``user_id`` dicts inside :class:`StandardBudget` and
:class:`InMemoryMemory` grow until the process OOMs. These tests
prove:

1. The :class:`BoundedDict` primitive evicts by LRU and TTL.
2. ``StandardBudget`` honours ``max_users`` / ``user_idle_ttl_seconds``.
3. ``InMemoryMemory`` honours the same kwargs for working blocks.
4. With the defaults disabled (``None``), behaviour matches today's
   unbounded implementation — single-tenant code is unaffected.
"""

from __future__ import annotations

import time

import anyio
import pytest

from loomflow import InMemoryMemory
from loomflow.core._eviction import BoundedDict
from loomflow.governance.budget import (
    BudgetConfig,
    StandardBudget,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# BoundedDict primitive
# ---------------------------------------------------------------------------


def test_bounded_dict_lru_eviction_on_overflow() -> None:
    """Setting beyond ``max_keys`` evicts the least-recently-touched."""
    d: BoundedDict[str, int] = BoundedDict(max_keys=2)
    d["a"] = 1
    d["b"] = 2
    d["c"] = 3  # evicts 'a' (oldest insertion)
    assert "a" not in d
    assert "b" in d and "c" in d


def test_bounded_dict_get_refreshes_lru() -> None:
    """A read on 'a' moves it to the most-recent end, so adding 'c'
    evicts 'b' instead."""
    d: BoundedDict[str, int] = BoundedDict(max_keys=2)
    d["a"] = 1
    d["b"] = 2
    _ = d.get("a")
    d["c"] = 3
    assert "b" not in d
    assert "a" in d and "c" in d


def test_bounded_dict_ttl_eviction() -> None:
    """An entry older than ``ttl_seconds`` is dropped on next touch."""
    d: BoundedDict[str, int] = BoundedDict(ttl_seconds=0.05)
    d["a"] = 1
    time.sleep(0.1)
    # Force a sweep; 'a' is past TTL.
    evicted = d.evict_expired()
    assert evicted == 1
    assert "a" not in d


def test_bounded_dict_setdefault_inserts_and_caps() -> None:
    """``setdefault`` past the cap also triggers LRU eviction."""
    d: BoundedDict[str, list[int]] = BoundedDict(max_keys=2)
    d.setdefault("a", [1])
    d.setdefault("b", [2])
    d.setdefault("c", [3])
    assert "a" not in d
    assert d["b"] == [2]
    assert d["c"] == [3]


def test_bounded_dict_unbounded_by_default() -> None:
    """No kwargs = behaves exactly like a regular dict (no eviction)."""
    d: BoundedDict[str, int] = BoundedDict()
    for i in range(1000):
        d[str(i)] = i
    assert len(d) == 1000


def test_bounded_dict_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        BoundedDict(max_keys=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BoundedDict(ttl_seconds=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# StandardBudget — bounded per-user accounting
# ---------------------------------------------------------------------------


async def test_budget_evicts_lru_user_past_max_users() -> None:
    """Past ``max_users``, the oldest-touched user's bucket is
    dropped — running totals reset for that user. Active users are
    unaffected."""
    budget = StandardBudget(
        BudgetConfig(),
        max_users=2,
        user_idle_ttl_seconds=None,
    )
    await budget.consume(tokens_in=10, tokens_out=0, cost_usd=0, user_id="alice")
    await budget.consume(tokens_in=20, tokens_out=0, cost_usd=0, user_id="bob")
    # Charlie pushes alice out (alice was oldest-touched).
    await budget.consume(tokens_in=30, tokens_out=0, cost_usd=0, user_id="charlie")

    # Alice's running totals are gone.
    assert budget.usage_for("alice")["tokens_total"] == 0
    # Bob and Charlie kept theirs.
    assert budget.usage_for("bob")["tokens_total"] == 20
    assert budget.usage_for("charlie")["tokens_total"] == 30


async def test_budget_ttl_evicts_idle_user() -> None:
    """A user idle past the TTL is dropped on the next touch."""
    budget = StandardBudget(
        BudgetConfig(),
        max_users=None,
        user_idle_ttl_seconds=0.05,
    )
    await budget.consume(tokens_in=10, tokens_out=0, cost_usd=0, user_id="alice")
    await anyio.sleep(0.1)
    # Touching with bob triggers a TTL sweep that drops alice.
    await budget.consume(tokens_in=5, tokens_out=0, cost_usd=0, user_id="bob")
    assert budget.usage_for("alice")["tokens_total"] == 0
    assert budget.usage_for("bob")["tokens_total"] == 5


async def test_budget_unbounded_when_kwargs_none() -> None:
    """Passing ``max_users=None`` + ``user_idle_ttl_seconds=None``
    matches the pre-M10 behaviour — single-tenant code with
    StandardBudget() and no kwargs gets no eviction."""
    budget = StandardBudget(
        BudgetConfig(),
        max_users=None,
        user_idle_ttl_seconds=None,
    )
    for i in range(50):
        await budget.consume(
            tokens_in=1, tokens_out=0, cost_usd=0, user_id=f"u{i}"
        )
    # All 50 retained.
    for i in range(50):
        assert budget.usage_for(f"u{i}")["tokens_total"] == 1


# ---------------------------------------------------------------------------
# InMemoryMemory — bounded working-block state
# ---------------------------------------------------------------------------


async def test_inmemory_evicts_lru_user_blocks_past_max_users() -> None:
    """Past ``max_users``, the oldest-touched user's working blocks
    are dropped together. Active users keep theirs."""
    mem = InMemoryMemory(max_users=2, user_idle_ttl_seconds=None)
    await mem.update_block("prefs", "alice prefs", user_id="alice")
    await mem.update_block("prefs", "bob prefs", user_id="bob")
    await mem.update_block("prefs", "charlie prefs", user_id="charlie")

    alice_blocks = await mem.working(user_id="alice")
    bob_blocks = await mem.working(user_id="bob")
    charlie_blocks = await mem.working(user_id="charlie")

    assert alice_blocks == []  # evicted
    assert len(bob_blocks) == 1 and bob_blocks[0].content == "bob prefs"
    assert (
        len(charlie_blocks) == 1
        and charlie_blocks[0].content == "charlie prefs"
    )


async def test_inmemory_ttl_evicts_idle_user_blocks() -> None:
    """A user idle past the TTL has all their working blocks
    dropped on the next touch."""
    mem = InMemoryMemory(
        max_users=None, user_idle_ttl_seconds=0.05
    )
    await mem.update_block("prefs", "alice prefs", user_id="alice")
    await anyio.sleep(0.1)
    # Bob's write triggers the TTL sweep.
    await mem.update_block("prefs", "bob prefs", user_id="bob")
    alice_blocks = await mem.working(user_id="alice")
    bob_blocks = await mem.working(user_id="bob")
    assert alice_blocks == []
    assert len(bob_blocks) == 1


async def test_inmemory_unbounded_when_kwargs_none() -> None:
    """Pre-M10 behaviour preserved when both bounds are None."""
    mem = InMemoryMemory(
        max_users=None, user_idle_ttl_seconds=None
    )
    for i in range(50):
        await mem.update_block(
            "prefs", f"user {i} prefs", user_id=f"u{i}"
        )
    # All 50 users' blocks retained.
    for i in range(50):
        blocks = await mem.working(user_id=f"u{i}")
        assert len(blocks) == 1
