"""Per-session signal mailbox shared by the runtime implementations.

The H2 fix: ``signal()``/``deliver()`` used to write into a dict that
nothing ever read. :class:`SignalMailbox` makes the channel real —
``deliver(name, payload)`` enqueues, ``wait(name)`` parks an anyio task
until a matching signal arrives (or pops immediately if one is already
queued), and ``poll(name)`` is the non-blocking pop-if-present.

Multiple payloads under one name queue FIFO. Multiple waiters on the
same name are woken oldest-first, one per delivered payload; which
waiter ends up with which payload is scheduler-dependent, but no
payload is ever lost or delivered twice.

All state is task-local to one event loop (anyio primitives, no
threads), matching how the in-proc and journaled runtimes are used.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import anyio

__all__ = ["SignalMailbox"]


class SignalMailbox:
    """Named FIFO signal queues with async waiters."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[Any]] = {}
        self._waiters: dict[str, deque[anyio.Event]] = {}

    def deliver(self, name: str, payload: Any) -> None:
        """Enqueue ``payload`` under ``name`` and wake one waiter."""
        self._queues.setdefault(name, deque()).append(payload)
        self._wake_one(name)

    def poll(self, name: str) -> Any | None:
        """Pop the oldest queued payload for ``name``, or ``None``.

        Non-blocking. Note the ambiguity: a queued payload that is
        itself ``None`` is indistinguishable from "nothing queued".
        """
        queue = self._queues.get(name)
        if not queue:
            return None
        return self._pop(name, queue)

    async def wait(self, name: str) -> Any:
        """Return the oldest payload for ``name``, parking until one
        arrives.

        Cancellation-safe: a waiter cancelled after being woken but
        before consuming passes its wakeup to the next waiter so the
        queued payload is never stranded.
        """
        while True:
            queue = self._queues.get(name)
            if queue:
                return self._pop(name, queue)
            event = anyio.Event()
            self._waiters.setdefault(name, deque()).append(event)
            try:
                await event.wait()
            except BaseException:
                if event.is_set():
                    # Woken but interrupted before consuming: hand the
                    # wakeup to the next parked waiter.
                    self._wake_one(name)
                else:
                    self._discard_waiter(name, event)
                raise
            # Loop: another task may have raced us to the payload, in
            # which case we park again.

    # ---- internals --------------------------------------------------------

    def _pop(self, name: str, queue: deque[Any]) -> Any:
        payload = queue.popleft()
        if not queue:
            del self._queues[name]
        return payload

    def _wake_one(self, name: str) -> None:
        waiters = self._waiters.get(name)
        if waiters:
            waiters.popleft().set()
            if not waiters:
                del self._waiters[name]

    def _discard_waiter(self, name: str, event: anyio.Event) -> None:
        waiters = self._waiters.get(name)
        if waiters is None:
            return
        try:
            waiters.remove(event)
        except ValueError:
            pass
        if not waiters:
            del self._waiters[name]
