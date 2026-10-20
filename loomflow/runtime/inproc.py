"""In-process runtime: no durability, no journal.

Every step just runs. Used in dev, tests, and demos. Production users
swap in :class:`SqliteRuntime` / :class:`PostgresRuntime`.

Two runtime-substrate features ARE supported here so the default
runtime behaves like the durable ones within a single process:

* **Signals** — ``signal()`` enqueues into a per-session FIFO mailbox;
  ``wait_for_signal()`` parks until a matching signal arrives and
  ``poll_signal()`` pops non-blocking. Session mailboxes are discarded
  when the session's last open context exits, so the sessions dict
  cannot grow without bound across runs.
* **Checkpoints** — ``put_checkpoint`` / ``get_checkpoint`` /
  ``get_latest_checkpoint`` / ``list_checkpoints`` backed by an
  in-memory store, enabling resume-within-process. Lost on exit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from ._signals import DEFAULT_MAX_PENDING, DEFAULT_PENDING_TTL_S, SessionTable, SignalMailbox
from .journal import Checkpoint, CheckpointMeta, InMemoryJournalStore


class InProcSession:
    """Trivial session: the session ID plus a signal mailbox."""

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self._mailbox = SignalMailbox()

    async def deliver(self, name: str, payload: Any) -> None:
        self._mailbox.deliver(name, payload)

    async def wait_signal(self, name: str) -> Any:
        return await self._mailbox.wait(name)

    def poll_signal(self, name: str) -> Any | None:
        return self._mailbox.poll(name)


class InProcRuntime:
    """No durability. Each step runs immediately."""

    name = "inproc"

    def __init__(
        self,
        *,
        max_checkpoints_per_session: int = 20,
        max_pending_signal_sessions: int | None = DEFAULT_MAX_PENDING,
        pending_signal_ttl_s: float | None = DEFAULT_PENDING_TTL_S,
    ) -> None:
        self._sessions: SessionTable[InProcSession] = SessionTable(
            InProcSession,
            max_pending=max_pending_signal_sessions,
            pending_ttl_s=pending_signal_ttl_s,
        )
        # Reuse the in-memory journal store purely for its checkpoint
        # shelf — step journaling stays disabled (steps run directly).
        self._checkpoints = InMemoryJournalStore(
            max_checkpoints_per_session=max_checkpoints_per_session
        )

    async def step(
        self,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return await fn(*args, **kwargs)

    def stream_step(
        self,
        name: str,
        fn: Callable[..., AsyncIterator[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        return fn(*args, **kwargs)

    # ---- session lifecycle -------------------------------------------------

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[InProcSession]:
        """Open (or re-enter) a session; its in-memory state — the
        signal mailbox — is discarded when the last open context for
        this ``session_id`` exits (checkpoints are kept)."""
        s = self._sessions.acquire(session_id)
        try:
            yield s
        finally:
            self._sessions.release(session_id, keep_signals=False)

    # ---- signals -------------------------------------------------------------

    async def signal(self, session_id: str, name: str, payload: Any) -> None:
        """Deliver a named signal to a session's FIFO mailbox.

        Signals sent while the session isn't open are queued for it —
        up to ``max_pending_signal_sessions`` sessions, for
        ``pending_signal_ttl_s`` — and dropped when its last open
        context exits.
        """
        await self._sessions.for_signal(session_id).deliver(name, payload)

    async def wait_for_signal(
        self, session_id: str, name: str, *, timeout_s: float | None = None
    ) -> Any:
        """Park until a matching signal arrives; return its payload.
        ``TimeoutError`` after ``timeout_s`` seconds (default: wait)."""
        return await self._sessions.wait(session_id, name, timeout_s)

    def poll_signal(self, session_id: str, name: str) -> Any | None:
        """Pop a queued signal if present; never blocks."""
        sess = self._sessions.get(session_id)
        return None if sess is None else sess.poll_signal(name)

    # ---- checkpoints -----------------------------------------------------------

    async def put_checkpoint(self, cp: Checkpoint) -> None:
        await self._checkpoints.put_checkpoint(cp)

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Checkpoint | None:
        return await self._checkpoints.get_checkpoint(
            session_id, checkpoint_id
        )

    async def get_latest_checkpoint(
        self, session_id: str
    ) -> Checkpoint | None:
        return await self._checkpoints.get_latest_checkpoint(session_id)

    async def list_checkpoints(
        self, session_id: str, limit: int = 50
    ) -> list[CheckpointMeta]:
        return await self._checkpoints.list_checkpoints(
            session_id, limit=limit
        )
