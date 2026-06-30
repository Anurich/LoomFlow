"""Journal-based durable runtime.

Wraps a :class:`JournalStore` with the :class:`Runtime` protocol. The
contract: every ``step()`` and ``stream_step()`` call inside an open
``session(session_id)`` context records its result. On a subsequent
call with the same ``(session_id, journal_key)``, the cached result is
returned without re-executing the underlying function.

The journal key is normally the positional ``name`` (e.g.
``tool_call_<turn>_<slot>``). When :meth:`step` is passed an
``idempotency_key`` — as the ReAct loop does for every tool call,
deriving it from the tool name + args — that key takes over instead,
namespaced under ``idem:`` so a content-hash can never collide with a
positional step name. This is what makes two *identical* tool calls in
different turns dedupe to a single side-effecting execution within a
session, rather than re-running because their positional names differ.

Session tracking uses :class:`contextvars.ContextVar`. anyio's
structured concurrency propagates contextvars to spawned tasks, so
parallel tool dispatches under ``_dispatch_tools`` still see the right
session id without explicit threading.

When ``step()`` is called outside any open session, the journal is
bypassed and the function runs directly — the runtime degrades
gracefully into the same behavior as :class:`InProcRuntime`.
"""

from __future__ import annotations

import contextvars
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from .journal import InMemoryJournalStore, JournalStore

_current_session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_jeeves_runtime_session",
    default=None,
)


def _journal_key(name: str, idempotency_key: str | None) -> str:
    """Resolve the journal key for a step.

    When ``idempotency_key`` is provided it wins, namespaced under
    ``idem:`` so a content-hash (the ReAct loop derives it from the
    tool name + args) can never collide with a positional step name
    like ``tool_call_<turn>_<slot>``. Otherwise fall back to the
    positional ``name``.
    """
    if idempotency_key is not None:
        return f"idem:{idempotency_key}"
    return name


class JournaledSession:
    """The handle yielded by :meth:`JournaledRuntime.session`."""

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self._signals: dict[str, Any] = {}

    async def deliver(self, name: str, payload: Any) -> None:
        self._signals[name] = payload


class JournaledRuntime:
    """Runtime that journals every step's result for replay.

    Pass any :class:`JournalStore` (in-memory for tests, sqlite for
    durable single-process use, future Postgres/DBOS adapters for
    multi-process / multi-host).
    """

    name = "journaled"

    def __init__(self, store: JournalStore | None = None) -> None:
        self._store: JournalStore = store if store is not None else InMemoryJournalStore()
        self._sessions: dict[str, JournaledSession] = {}

    @property
    def store(self) -> JournalStore:
        return self._store

    # ---- session lifecycle ----------------------------------------------

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[JournaledSession]:
        token = _current_session_var.set(session_id)
        sess = self._sessions.setdefault(
            session_id, JournaledSession(session_id)
        )
        try:
            yield sess
        finally:
            _current_session_var.reset(token)

    async def signal(self, session_id: str, name: str, payload: Any) -> None:
        sess = self._sessions.get(session_id)
        if sess is not None:
            await sess.deliver(name, payload)

    # ---- step ------------------------------------------------------------

    async def step(
        self,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run ``fn`` once per ``(session, journal_key)`` and journal it.

        When ``idempotency_key`` is supplied, it becomes the journal
        key (under the ``idem:`` namespace) so that two calls with the
        same key dedupe to a single execution regardless of their
        positional ``name``; otherwise ``name`` is the key.

        Caveat — this is the *defined* meaning of an idempotency key
        here: a tool whose result varies per call with identical args
        (a clock, a counter, ``random()``, a live API read) will
        **replay** its first recorded result within the session rather
        than re-execute. That determinism is exactly what a durable,
        crash-resumable runtime requires; pass no ``idempotency_key``
        for steps that must genuinely re-run on every distinct call.
        """
        session_id = _current_session_var.get()
        if session_id is None:
            return await fn(*args, **kwargs)

        key = _journal_key(name, idempotency_key)
        cached = await self._store.get_step(session_id, key)
        if cached is not None:
            return cached.value

        result = await fn(*args, **kwargs)
        await self._store.put_step(session_id, key, result)
        return result

    # ---- stream_step -----------------------------------------------------

    def stream_step(
        self,
        name: str,
        fn: Callable[..., AsyncIterator[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        return self._stream_replay_or_record(name, fn, *args, **kwargs)

    async def _stream_replay_or_record(
        self,
        name: str,
        fn: Callable[..., AsyncIterator[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        session_id = _current_session_var.get()
        if session_id is None:
            async for chunk in fn(*args, **kwargs):
                yield chunk
            return

        cached = await self._store.get_stream(session_id, name)
        if cached is not None:
            for chunk in cached:
                yield chunk
            return

        recorded: list[Any] = []
        async for chunk in fn(*args, **kwargs):
            recorded.append(chunk)
            yield chunk
        await self._store.put_stream(session_id, name, recorded)
