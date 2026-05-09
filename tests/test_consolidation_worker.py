"""ConsolidationWorker tests — periodic consolidation in the background."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import anyio
import pytest

from loomflow import InMemoryMemory, ScriptedModel, ScriptedTurn
from loomflow.core.types import Episode
from loomflow.memory import ConsolidationWorker, Consolidator

pytestmark = pytest.mark.anyio


def _ep(text: str = "hi", out: str = "ack") -> Episode:
    return Episode(
        session_id="s1",
        input=text,
        output=out,
        occurred_at=datetime.now(UTC),
    )


def _consolidator_extracting(json_arrays: list[str]) -> Consolidator:
    """Build a Consolidator whose model returns each JSON array in turn."""
    return Consolidator(
        model=ScriptedModel(
            [ScriptedTurn(text=arr) for arr in json_arrays] * 5
        )
    )


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------


async def test_run_once_returns_zero_when_no_consolidator() -> None:
    memory = InMemoryMemory()
    worker = ConsolidationWorker(memory, interval_seconds=60)
    assert await worker.run_once() == 0
    assert worker.iterations == 1


async def test_run_once_returns_count_of_new_facts() -> None:
    extracted = '[{"subject":"u","predicate":"p","object":"o","confidence":0.9}]'
    memory = InMemoryMemory(consolidator=_consolidator_extracting([extracted]))
    await memory.remember(_ep())

    worker = ConsolidationWorker(memory, interval_seconds=60)
    n = await worker.run_once()
    assert n == 1
    assert worker.total_extracted == 1


async def test_run_once_idempotent_on_already_consolidated_episodes() -> None:
    extracted = '[{"subject":"u","predicate":"p","object":"o","confidence":0.9}]'
    memory = InMemoryMemory(consolidator=_consolidator_extracting([extracted]))
    await memory.remember(_ep())

    worker = ConsolidationWorker(memory, interval_seconds=60)
    first = await worker.run_once()
    second = await worker.run_once()
    assert first == 1
    assert second == 0  # nothing new on the second pass


async def test_run_once_routes_errors_to_callback_and_returns_zero() -> None:
    """A buggy consolidator must not crash the worker."""

    class _BoomModel:
        name = "boom"

        async def stream(self, messages, **kw):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

    memory = InMemoryMemory(consolidator=Consolidator(model=_BoomModel()))
    await memory.remember(_ep())

    seen_errors: list[BaseException] = []

    async def on_error(exc: BaseException) -> None:
        seen_errors.append(exc)

    worker = ConsolidationWorker(memory, interval_seconds=60, on_error=on_error)
    n = await worker.run_once()
    assert n == 0
    assert len(seen_errors) == 1
    assert isinstance(seen_errors[0], RuntimeError)
    assert "kaboom" in str(seen_errors[0])


async def test_on_consolidated_callback_fires_with_count() -> None:
    extracted = (
        '[{"subject":"u","predicate":"p1","object":"o1"},'
        '{"subject":"u","predicate":"p2","object":"o2"}]'
    )
    memory = InMemoryMemory(consolidator=_consolidator_extracting([extracted]))
    await memory.remember(_ep())

    seen_counts: list[int] = []

    async def on_consolidated(n: int) -> None:
        seen_counts.append(n)

    worker = ConsolidationWorker(
        memory,
        interval_seconds=60,
        on_consolidated=on_consolidated,
    )
    await worker.run_once()
    assert seen_counts == [2]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_zero_or_negative_interval_rejected() -> None:
    with pytest.raises(ValueError):
        ConsolidationWorker(InMemoryMemory(), interval_seconds=0)
    with pytest.raises(ValueError):
        ConsolidationWorker(InMemoryMemory(), interval_seconds=-1.5)


# ---------------------------------------------------------------------------
# run_forever — verify cancellation works inside a task group
# ---------------------------------------------------------------------------


async def test_run_forever_can_be_cancelled_via_task_group() -> None:
    """Spawning ``run_forever`` and cancelling its task group should
    terminate the worker cooperatively without leaking tasks."""
    extracted = '[{"subject":"u","predicate":"p","object":"o"}]'
    memory = InMemoryMemory(consolidator=_consolidator_extracting([extracted]))
    await memory.remember(_ep())

    worker = ConsolidationWorker(memory, interval_seconds=0.01)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker.run_forever)
        # Let it tick a few times.
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    # ``iterations`` advanced; we made it through the cancel cleanly.
    assert worker.iterations >= 1


# ---------------------------------------------------------------------------
# Async context manager helper
# ---------------------------------------------------------------------------


async def test_async_context_manager_runs_in_background() -> None:
    extracted = '[{"subject":"u","predicate":"p","object":"o"}]'
    memory = InMemoryMemory(consolidator=_consolidator_extracting([extracted]))
    await memory.remember(_ep())

    seen: list[int] = []

    async def on_c(n: int) -> None:
        seen.append(n)

    worker = ConsolidationWorker(
        memory,
        interval_seconds=0.01,
        on_consolidated=on_c,
    )

    async with worker:
        # Worker is running in the background; sleep a beat.
        await anyio.sleep(0.05)

    # On context exit the worker is cancelled. We should have seen at
    # least one consolidation by now.
    assert len(seen) >= 1
    assert all(c > 0 for c in seen)


# ---------------------------------------------------------------------------
# Backend without ``.facts`` attribute
# ---------------------------------------------------------------------------


async def test_backend_without_facts_returns_zero() -> None:
    """A custom Memory without a ``.facts`` attribute exits run_once
    with zero rather than crashing."""

    class _BareMemory:
        async def working(self) -> list[Any]:
            return []

        async def update_block(self, name: str, content: str) -> None:
            pass

        async def append_block(self, name: str, content: str) -> None:
            pass

        async def remember(self, episode: Episode) -> str:
            return episode.id

        async def recall(self, query: str, **kwargs: Any) -> list[Any]:
            return []

        async def consolidate(self) -> None:
            return None

    worker = ConsolidationWorker(_BareMemory(), interval_seconds=60)  # type: ignore[arg-type]
    n = await worker.run_once()
    assert n == 0
