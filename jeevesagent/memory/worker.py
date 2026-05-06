"""Background consolidation worker.

A long-running anyio task that periodically calls
``memory.consolidate()``. Useful for very long-lived agents where
per-run consolidation (the ``auto_consolidate=True`` flag on
:class:`Agent`) is wasteful — you'd rather batch every N seconds.

Usage::

    worker = ConsolidationWorker(memory, interval_seconds=60)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker.run_forever)

        # main agent work here…
        await main()

    # On task-group exit the worker is cancelled cleanly; any
    # in-flight consolidate call gets cooperatively interrupted at the
    # next ``await``.

Errors raised by the underlying ``memory.consolidate()`` call are
caught and routed to the optional ``on_error`` callback so a transient
LLM hiccup doesn't kill the worker. New facts trigger
``on_consolidated(count)`` when set; both callbacks are awaitable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from ..core.protocols import Memory

OnConsolidatedCb = Callable[[int], Awaitable[None]]
OnErrorCb = Callable[[BaseException], Awaitable[None]]


class ConsolidationWorker:
    """Periodic consolidator for any :class:`Memory` backend."""

    def __init__(
        self,
        memory: Memory,
        *,
        interval_seconds: float = 60.0,
        on_consolidated: OnConsolidatedCb | None = None,
        on_error: OnErrorCb | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._memory = memory
        self._interval = interval_seconds
        self._on_consolidated = on_consolidated
        self._on_error = on_error
        self._iterations = 0
        self._total_extracted = 0

    @property
    def iterations(self) -> int:
        """Number of consolidate cycles attempted (test introspection)."""
        return self._iterations

    @property
    def total_extracted(self) -> int:
        """Cumulative count of facts extracted across all cycles."""
        return self._total_extracted

    # ---- one-shot API (handy for tests + manual invocation) ------------

    async def run_once(self) -> int:
        """Run a single consolidation pass. Returns the number of new
        facts extracted (``0`` when no fact store / nothing changed).

        Errors in ``memory.consolidate()`` are routed to ``on_error``
        and **not** re-raised, so callers can use this in a polling
        loop without wrapping it in their own try/except.
        """
        self._iterations += 1
        fact_store = getattr(self._memory, "facts", None)
        before = 0
        if fact_store is not None:
            before = len(await fact_store.all_facts())
        try:
            await self._memory.consolidate()
        except BaseException as exc:  # noqa: BLE001 — surface via callback
            if self._on_error is not None:
                await self._on_error(exc)
            return 0
        if fact_store is None:
            return 0
        after = len(await fact_store.all_facts())
        count = max(0, after - before)
        self._total_extracted += count
        if count > 0 and self._on_consolidated is not None:
            await self._on_consolidated(count)
        return count

    # ---- run-forever loop ---------------------------------------------

    async def run_forever(self) -> None:
        """Sleep ``interval_seconds`` then consolidate. Repeat until
        cancelled.

        Spawn this in an :func:`anyio.create_task_group` — the cancel
        scope at scope exit terminates the worker cooperatively.
        """
        while True:
            await anyio.sleep(self._interval)
            await self.run_once()

    # ---- async-context-manager sugar -----------------------------------

    async def __aenter__(self) -> ConsolidationWorker:
        # Lazily attach a task-group on first use so the worker can be
        # used standalone (without callers managing a task group). The
        # contract: ``async with worker: ...`` runs ``run_forever``
        # in the background; exiting the block cancels it.
        self._tg = await anyio.create_task_group().__aenter__()
        self._tg.start_soon(self.run_forever)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        tg = getattr(self, "_tg", None)
        if tg is None:
            return
        tg.cancel_scope.cancel()
        await tg.__aexit__(*exc_info)
