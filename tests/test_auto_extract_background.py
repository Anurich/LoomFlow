"""Tests for ``AutoExtractMemory(background=True)`` — fire-and-forget
LLM fact extraction.

Pins four behaviours that together make background-by-default safe
+ semantically correct:

1. ``remember()`` returns BEFORE the extraction completes (the whole
   point — kill the 3-10s per-turn latency that motivated the change).
2. ``aclose()`` drains in-flight tasks so a process shutting down
   has a chance to finish writing facts.
3. ``aclose()`` returns a non-zero count when the drain timeout
   expires before all tasks finish — gives ops visibility into
   facts that may have been lost.
4. ``background=False`` opt-out keeps the legacy synchronous
   semantics so tests + sync-required callers aren't broken.
"""

from __future__ import annotations

import asyncio

import anyio
import pytest

from loomflow.core.types import Episode, Fact
from loomflow.memory.auto_extract import AutoExtractMemory
from loomflow.memory.consolidator import Consolidator
from loomflow.memory.facts import InMemoryFactStore
from loomflow.memory.inmemory import InMemoryMemory

pytestmark = pytest.mark.anyio


def _make_inner() -> InMemoryMemory:
    inner = InMemoryMemory()
    inner.facts = InMemoryFactStore()
    return inner


class _SlowConsolidator(Consolidator):
    """Sleeps inside consolidate() so tests can verify that
    remember() returns BEFORE consolidation finishes."""

    def __init__(self, *, delay: float = 0.1) -> None:
        # Skip parent __init__; we don't need a model.
        self._delay = delay
        self.started = False
        self.finished = False

    async def consolidate(self, episodes, *, store) -> None:  # type: ignore[no-untyped-def]
        self.started = True
        await anyio.sleep(self._delay)
        for ep in episodes:
            await store.append(
                Fact(
                    subject=ep.user_id or "anon",
                    predicate="said",
                    object=ep.input[:32],
                    confidence=1.0,
                    user_id=ep.user_id,
                )
            )
        self.finished = True


async def test_background_remember_returns_before_extraction_completes() -> None:
    """The headline contract: ``remember()`` must return BEFORE the
    LLM fact-extraction completes. That's the latency win."""
    inner = _make_inner()
    slow = _SlowConsolidator(delay=0.5)
    mem = AutoExtractMemory(inner, slow, background=True)

    await mem.remember(
        Episode(session_id="s", input="hi", output="ok", user_id="u")
    )
    # If background is working, the consolidator hasn't finished
    # by the time remember() returns. ``started`` may or may not
    # be True (race with the event loop scheduling); ``finished``
    # MUST be False — that's the no-blocking guarantee.
    assert slow.finished is False, (
        "remember() blocked on extraction — background path broken"
    )

    # Drain so the test cleans up properly.
    remaining = await mem.aclose(timeout=5.0)
    assert remaining == 0
    assert slow.finished is True


async def test_aclose_drains_pending_extractions() -> None:
    """After ``aclose()`` returns with count=0, every fact that
    would have been extracted is actually in the inner store."""
    inner = _make_inner()
    mem = AutoExtractMemory(
        inner, _SlowConsolidator(delay=0.05), background=True
    )

    for i in range(3):
        await mem.remember(
            Episode(
                session_id="s",
                input=f"prompt {i}",
                output="ok",
                user_id="alice",
            )
        )

    remaining = await mem.aclose(timeout=5.0)
    assert remaining == 0
    facts = await inner.facts.query(user_id="alice", limit=10)
    assert len(facts) == 3


async def test_aclose_returns_count_on_timeout() -> None:
    """When the drain timeout fires before all tasks complete,
    ``aclose()`` returns the count of still-pending tasks so the
    caller can log / alert. Tasks aren't cancelled — they keep
    running until the loop closes — but the caller knows."""
    inner = _make_inner()
    slow = _SlowConsolidator(delay=1.0)
    mem = AutoExtractMemory(inner, slow, background=True)

    await mem.remember(
        Episode(session_id="s", input="x", output="y", user_id="u")
    )

    # Drain with a too-short timeout — task is still in flight.
    remaining = await mem.aclose(timeout=0.05)
    assert remaining == 1

    # The task is still alive; finish draining for cleanup.
    final = await mem.aclose(timeout=5.0)
    assert final == 0


async def test_aclose_no_pending_returns_zero() -> None:
    """A clean ``aclose()`` with no work in flight is a fast
    no-op."""
    inner = _make_inner()
    mem = AutoExtractMemory(
        inner, _SlowConsolidator(), background=True
    )
    assert await mem.aclose(timeout=1.0) == 0


async def test_background_false_blocks_synchronously() -> None:
    """Legacy opt-out: ``background=False`` keeps remember()
    blocking on extraction. Pin the contract so tests that need
    sync semantics (deterministic post-remember assertions) keep
    working."""
    inner = _make_inner()
    slow = _SlowConsolidator(delay=0.05)
    mem = AutoExtractMemory(inner, slow, background=False)

    await mem.remember(
        Episode(session_id="s", input="hi", output="ok", user_id="u")
    )
    # With background=False, remember() must NOT return until
    # extraction has run to completion.
    assert slow.finished is True
    facts = await inner.facts.query(user_id="u", limit=10)
    assert len(facts) == 1


async def test_background_pending_set_is_bounded() -> None:
    """In long-lived processes (REPL sessions) hundreds of
    episodes accumulate. The ``_pending`` set must stay bounded
    via the done-callback — not grow without bound."""
    inner = _make_inner()
    mem = AutoExtractMemory(
        inner, _SlowConsolidator(delay=0.01), background=True
    )
    for i in range(20):
        await mem.remember(
            Episode(
                session_id="s",
                input=f"i={i}",
                output="ok",
                user_id="u",
            )
        )
    # Let all the quick tasks finish.
    await asyncio.sleep(0.3)
    # The done-callback should have removed every completed task.
    # Allow a small tolerance for a still-scheduling task; what
    # we're really pinning is "doesn't grow with each remember()".
    assert len(mem._pending) <= 2


async def test_failed_extraction_in_background_does_not_break() -> None:
    """A consolidator that raises must NOT crash the event loop
    when called in background — the extraction is best-effort
    and the failure is logged + swallowed."""

    class _FailingConsolidator(Consolidator):
        def __init__(self) -> None:
            pass

        async def consolidate(self, episodes, *, store):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    inner = _make_inner()
    mem = AutoExtractMemory(
        inner, _FailingConsolidator(), background=True
    )
    # Should NOT raise from remember even though extraction will fail.
    await mem.remember(
        Episode(session_id="s", input="x", output="y", user_id="u")
    )
    # Drain — should complete cleanly (errors swallowed inside
    # _maybe_extract).
    remaining = await mem.aclose(timeout=5.0)
    assert remaining == 0
