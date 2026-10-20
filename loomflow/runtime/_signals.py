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

:class:`SessionTable` holds the runtimes' per-session mailboxes. A
session that is open, or has a task waiting on it, stays put. A
mailbox holding signals for a session that is not open — sent before
its run, or after it ended — is kept in a bounded, expiring map
(``max_pending`` sessions, ``pending_ttl_s`` seconds), so signals to
sessions that never run can't grow memory without limit.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar

import anyio

from ..core._eviction import BoundedDict

__all__ = ["DEFAULT_MAX_PENDING", "DEFAULT_PENDING_TTL_S", "SessionTable", "SignalMailbox"]

#: Sessions not open whose undelivered signals are kept.
DEFAULT_MAX_PENDING = 10_000
#: How long such signals are kept (seconds).
DEFAULT_PENDING_TTL_S = 3600.0


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

    def has_signals(self) -> bool:
        """True when any payload is queued."""
        return bool(self._queues)

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


class _HasMailbox(Protocol):
    _mailbox: SignalMailbox


_S = TypeVar("_S", bound=_HasMailbox)


class SessionTable(Generic[_S]):
    """Per-session state for a runtime: pinned while a session is open
    or awaited, bounded and expiring otherwise (see module docstring)."""

    def __init__(
        self,
        factory: Callable[[str], _S],
        *,
        max_pending: int | None = DEFAULT_MAX_PENDING,
        pending_ttl_s: float | None = DEFAULT_PENDING_TTL_S,
    ) -> None:
        self._factory = factory
        self._open: dict[str, _S] = {}
        self._refs: dict[str, int] = {}
        self._pending: BoundedDict[str, _S] = BoundedDict(
            max_keys=max_pending, ttl_seconds=pending_ttl_s
        )

    def acquire(self, session_id: str) -> _S:
        """Pin ``session_id`` (an open context or a waiter)."""
        sess = self._open.get(session_id)
        if sess is None:
            sess = self._pending.pop(session_id) or self._factory(session_id)
            self._open[session_id] = sess
        self._refs[session_id] = self._refs.get(session_id, 0) + 1
        return sess

    def release(self, session_id: str, *, keep_signals: bool) -> None:
        """Unpin; on the last release the session leaves the table —
        its undelivered signals move to the pending map when
        ``keep_signals`` (a waiter leaving), else they're dropped (the
        run ended)."""
        remaining = self._refs.get(session_id, 1) - 1
        if remaining > 0:
            self._refs[session_id] = remaining
            return
        self._refs.pop(session_id, None)
        sess = self._open.pop(session_id, None)
        if keep_signals and sess is not None and sess._mailbox.has_signals():
            self._pending[session_id] = sess

    def for_signal(self, session_id: str) -> _S:
        """The session a signal for ``session_id`` goes to."""
        sess = self._open.get(session_id)
        if sess is not None:
            return sess
        pending = self._pending.get(session_id)
        if pending is None:
            pending = self._factory(session_id)
            self._pending[session_id] = pending
        return pending

    def get(self, session_id: str) -> _S | None:
        return self._open.get(session_id) or self._pending.get(session_id)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._open or session_id in self._pending

    def __len__(self) -> int:
        return len(self._open) + len(self._pending)

    async def wait(self, session_id: str, name: str, timeout: float | None) -> Any:
        """Wait for a signal on ``session_id`` (pinned meanwhile);
        ``TimeoutError`` after ``timeout`` seconds."""
        sess = self.acquire(session_id)
        try:
            with anyio.fail_after(timeout):
                return await sess._mailbox.wait(name)
        finally:
            self.release(session_id, keep_signals=True)
