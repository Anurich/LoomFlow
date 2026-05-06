"""In-process runtime: no durability, no journal.

Every step just runs. Used in dev, tests, and demos. Production users
swap in :class:`DBOSRuntime` or :class:`TemporalRuntime` (Phase 5).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any


class InProcSession:
    """Trivial session: just a holder for the session ID and signals."""

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self._signals: dict[str, Any] = {}

    async def deliver(self, name: str, payload: Any) -> None:
        self._signals[name] = payload


class InProcRuntime:
    """No durability. Each step runs immediately."""

    name = "inproc"

    def __init__(self) -> None:
        self._sessions: dict[str, InProcSession] = {}

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

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[InProcSession]:
        s = self._sessions.setdefault(session_id, InProcSession(session_id))
        try:
            yield s
        finally:
            pass  # no persistence; keep in-memory for the process lifetime

    async def signal(self, session_id: str, name: str, payload: Any) -> None:
        s = self._sessions.get(session_id)
        if s is not None:
            await s.deliver(name, payload)
