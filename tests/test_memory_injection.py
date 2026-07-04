"""G7 — Token-budgeted + decaying memory injection tests.

Unit-level coverage of :mod:`loomflow.memory._injection`
(:func:`budget_items` packing, decay ordering, truncation marker,
always-at-least-one) plus integration through the ReAct seed assembly
(:func:`loomflow.architecture.react._build_seed_messages`): budget
respected, working blocks pinned (never dropped), decay reorders
equally-relevant episodes, and ``memory_token_budget=None`` is exact
parity with the legacy item-count path.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from loomflow import Agent, InMemoryMemory, ScriptedModel, ScriptedTurn, Tuning
from loomflow.architecture import Dependencies
from loomflow.architecture.react import _build_seed_messages
from loomflow.core.types import Episode, Fact, Message, Role, ToolResult
from loomflow.governance.budget import NoBudget
from loomflow.memory._injection import (
    TRUNCATION_MARKER,
    budget_items,
    decay_factor,
    estimate_tokens,
)
from loomflow.observability.tracing import NoTelemetry
from loomflow.runtime.inproc import InProcRuntime
from loomflow.security.hooks import HookRegistry
from loomflow.security.permissions import AllowAll

pytestmark = pytest.mark.anyio

_NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# budget_items — unit
# ---------------------------------------------------------------------------


def test_budget_respected_greedy_fill() -> None:
    items: list[tuple[str, float, datetime | None]] = [
        ("x" * 400, 1.0, None) for _ in range(5)
    ]  # 100 tokens each
    out = budget_items(
        items, budget_tokens=250, half_life_days=None, now=_NOW
    )
    # Two fit fully, the third is truncated to the 50-token remainder.
    assert len(out) == 3
    assert out[0] == "x" * 400 and out[1] == "x" * 400
    assert out[2].endswith(TRUNCATION_MARKER)
    assert len(out[2]) <= 50 * 4 + len(TRUNCATION_MARKER)
    # Fully-included items stay within budget.
    assert sum(estimate_tokens(t) for t in out[:2]) <= 250


def test_decay_orders_fresh_above_old_at_equal_relevance() -> None:
    old = ("old item", 1.0, _NOW - timedelta(days=100))
    fresh = ("fresh item", 1.0, _NOW - timedelta(days=1))
    out = budget_items(
        [old, fresh], budget_tokens=1000, half_life_days=30.0, now=_NOW
    )
    assert out == ["fresh item", "old item"]


def test_no_decay_preserves_input_order_on_ties() -> None:
    old = ("old item", 1.0, _NOW - timedelta(days=100))
    fresh = ("fresh item", 1.0, _NOW - timedelta(days=1))
    out = budget_items(
        [old, fresh], budget_tokens=1000, half_life_days=None, now=_NOW
    )
    assert out == ["old item", "fresh item"]


def test_relevance_beats_recency_when_gap_is_large() -> None:
    weak_fresh = ("weak", 0.1, _NOW - timedelta(days=1))
    strong_old = ("strong", 1.0, _NOW - timedelta(days=60))
    out = budget_items(
        [weak_fresh, strong_old],
        budget_tokens=1000,
        half_life_days=30.0,
        now=_NOW,
    )
    # 1.0 * 0.5**2 = 0.25 still beats 0.1 * ~1.0.
    assert out == ["strong", "weak"]


def test_untimestamped_items_never_decay() -> None:
    assert decay_factor(None, half_life_days=30.0, now=_NOW) == 1.0
    aged = decay_factor(
        _NOW - timedelta(days=30), half_life_days=30.0, now=_NOW
    )
    assert aged == pytest.approx(0.5)


def test_always_includes_at_least_one_item_truncated() -> None:
    out = budget_items(
        [("y" * 1000, 1.0, None)],
        budget_tokens=10,
        half_life_days=None,
        now=_NOW,
    )
    assert len(out) == 1
    assert out[0] == "y" * 40 + TRUNCATION_MARKER


def test_naive_timestamps_are_reconciled_not_fatal() -> None:
    naive_old = ("naive", 1.0, datetime(2020, 1, 1))  # noqa: DTZ001
    out = budget_items(
        [naive_old, ("fresh", 1.0, _NOW)],
        budget_tokens=1000,
        half_life_days=30.0,
        now=_NOW,
    )
    assert out == ["fresh", "naive"]


def test_empty_items_returns_empty() -> None:
    assert (
        budget_items([], budget_tokens=100, half_life_days=None, now=_NOW)
        == []
    )


# ---------------------------------------------------------------------------
# Integration — ReAct seed assembly
# ---------------------------------------------------------------------------


class _EmptyToolHost:
    async def list_tools(self, *, query: str | None = None) -> list[object]:
        return []

    async def call(
        self, tool: str, args: dict[str, object], *, call_id: str = ""
    ) -> ToolResult:
        return ToolResult.error_(call_id or "none", "no tools")


def _make_deps(memory: InMemoryMemory, **overrides: object) -> Dependencies:
    deps = Dependencies(
        model=ScriptedModel([ScriptedTurn(text="ok")]),  # type: ignore[arg-type]
        memory=memory,
        runtime=InProcRuntime(),
        tools=_EmptyToolHost(),  # type: ignore[arg-type]
        budget=NoBudget(),
        permissions=AllowAll(),
        hooks=HookRegistry(),
        telemetry=NoTelemetry(),
        audit_log=None,
        max_turns=5,
    )
    return replace(deps, **overrides) if overrides else deps  # type: ignore[arg-type]


async def _fixture_memory() -> InMemoryMemory:
    mem = InMemoryMemory()
    await mem.update_block("profile", "user prefers postgres")
    await mem.facts.append(
        Fact(subject="postgres", predicate="uses", object="replication")
    )
    await mem.facts.append(
        Fact(subject="postgres", predicate="runs_on", object="port 5432")
    )
    await mem.remember(
        Episode(
            session_id="other",
            input="postgres replication question",
            output="explained logical replication",
            occurred_at=_NOW - timedelta(days=2),
        )
    )
    return mem


async def test_none_budget_is_parity_with_legacy_path() -> None:
    mem = await _fixture_memory()
    baseline = await _build_seed_messages(
        _make_deps(mem), "instr", "postgres replication"
    )
    # Legacy structure: partitioned headers, working block pinned.
    joined = "\n".join(m.content for m in baseline if m.role == Role.SYSTEM)
    assert "Known facts:" in joined
    assert "Relevant past episodes:" in joined
    assert "<profile>" in joined

    # A half-life alone (budget None) must not change ANYTHING.
    with_decay_only = await _build_seed_messages(
        _make_deps(mem, memory_decay_half_life=30.0),
        "instr",
        "postgres replication",
    )
    assert with_decay_only == baseline


async def test_budget_produces_single_ranked_block_within_budget() -> None:
    mem = await _fixture_memory()
    deps = _make_deps(mem, memory_token_budget=4000)
    msgs = await _build_seed_messages(deps, "instr", "postgres replication")
    recall = [
        m
        for m in msgs
        if m.role == Role.SYSTEM and "Recalled memory" in m.content
    ]
    assert len(recall) == 1
    # Fact + episode lines merged into one uniformly-budgeted list.
    assert "postgres uses replication" in recall[0].content
    assert "postgres replication question" in recall[0].content
    # Legacy headers absent under the budgeted path.
    assert "Known facts:" not in recall[0].content


async def test_budget_truncates_oversized_episode_with_marker() -> None:
    mem = InMemoryMemory()
    await mem.remember(
        Episode(
            session_id="other",
            input="huge postgres dump",
            output="x" * 10_000,
            occurred_at=_NOW,
        )
    )
    deps = _make_deps(mem, memory_token_budget=200)
    msgs = await _build_seed_messages(deps, "instr", "postgres dump")
    (recall,) = [
        m
        for m in msgs
        if m.role == Role.SYSTEM and "Recalled memory" in m.content
    ]
    assert TRUNCATION_MARKER in recall.content
    # Item content is capped at the budget (chars/4 heuristic).
    assert len(recall.content) <= 200 * 4 + 200  # header + marker slack


async def test_decay_reorders_equally_relevant_episodes() -> None:
    mem = InMemoryMemory()
    old_ts = _NOW - timedelta(days=365)
    fresh_ts = _NOW - timedelta(days=1)
    for ts in (old_ts, fresh_ts):
        await mem.remember(
            Episode(
                session_id="other",
                input="postgres replication question",
                output="explained logical replication",
                occurred_at=ts,
            )
        )
    deps = _make_deps(
        mem, memory_token_budget=4000, memory_decay_half_life=30.0
    )
    msgs = await _build_seed_messages(deps, "instr", "postgres replication")
    (recall,) = [
        m
        for m in msgs
        if m.role == Role.SYSTEM and "Recalled memory" in m.content
    ]
    fresh_pos = recall.content.find(fresh_ts.isoformat())
    old_pos = recall.content.find(old_ts.isoformat())
    assert fresh_pos != -1 and old_pos != -1
    assert fresh_pos < old_pos


async def test_working_blocks_pinned_never_dropped() -> None:
    mem = InMemoryMemory()
    await mem.update_block("profile", "z" * 400)  # ~100 tokens
    await mem.remember(
        Episode(
            session_id="other",
            input="postgres question",
            output="answer",
            occurred_at=_NOW,
        )
    )
    # Budget smaller than the block alone: block still ships in full,
    # recall allowance is exhausted so no recall block is added.
    deps = _make_deps(mem, memory_token_budget=10)
    msgs = await _build_seed_messages(deps, "instr", "postgres question")
    block_msgs = [
        m for m in msgs if m.role == Role.SYSTEM and "<profile>" in m.content
    ]
    assert len(block_msgs) == 1
    assert "z" * 400 in block_msgs[0].content  # untruncated
    assert not any("Recalled memory" in m.content for m in msgs)


# ---------------------------------------------------------------------------
# Tuning → Dependencies threading
# ---------------------------------------------------------------------------


async def test_tuning_defaults_off_and_agent_runs_unchanged() -> None:
    t = Tuning()
    assert t.memory_token_budget is None
    assert t.memory_decay_half_life_days is None

    mem = await _fixture_memory()
    model = ScriptedModel([ScriptedTurn(text="done")])
    agent = Agent(
        "instr",
        model=model,
        memory=mem,
        tuning=Tuning(memory_token_budget=4000, memory_decay_half_life_days=30.0),
    )
    result = await agent.run("postgres replication")
    assert result.output == "done"


def _seed_system_contents(msgs: list[Message]) -> list[str]:
    return [m.content for m in msgs if m.role == Role.SYSTEM]
