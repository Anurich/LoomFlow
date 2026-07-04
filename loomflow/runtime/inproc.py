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

from ._signals import SignalMailbox
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

    def __init__(self, *, max_checkpoints_per_session: int = 20) -> None:
        self._sessions: dict[str, InProcSession] = {}
        self._session_refs: dict[str, int] = {}
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
        s = self._get_session(session_id)
        self._session_refs[session_id] = (
            self._session_refs.get(session_id, 0) + 1
        )
        try:
            yield s
        finally:
            remaining = self._session_refs.get(session_id, 1) - 1
            if remaining <= 0:
                self._session_refs.pop(session_id, None)
                self._sessions.pop(session_id, None)
            else:
                self._session_refs[session_id] = remaining

    def _get_session(self, session_id: str) -> InProcSession:
        return self._sessions.setdefault(
            session_id, InProcSession(session_id)
        )

    # ---- signals -------------------------------------------------------------

    async def signal(self, session_id: str, name: str, payload: Any) -> None:
        """Deliver a named signal to a session's FIFO mailbox.

        Signals sent before the session is opened are queued and
        survive until the session's last open context exits.
        """
        await self._get_session(session_id).deliver(name, payload)

    async def wait_for_signal(self, session_id: str, name: str) -> Any:
        """Park until a matching signal arrives; return its payload."""
        return await self._get_session(session_id).wait_signal(name)

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
