"""Journal-based durable runtime.

Wraps a :class:`JournalStore` with the :class:`Runtime` protocol. The
contract: every ``step()`` and ``stream_step()`` call inside an open
``session(session_id)`` context records its result under a journal key
built from the step *name* plus a **fingerprint of the step's inputs**
(``"<name>@<sha256[:16]>"``). On a subsequent call with the same
session, name AND inputs, the cached result is returned without
re-executing the underlying function. If the inputs differ — e.g.
``run()`` is invoked again against an old ``session_id`` with a new
prompt — the fingerprint differs, so the step executes fresh: input
changed → cache invalidated. (The stale entry stays in the store until
:meth:`JournalStore.prune`.)

Fingerprints are computed by stable serialization of the positional and
keyword arguments: Pydantic values via ``model_dump_json``, containers
recursively (mappings key-sorted), scalars via ``repr``, and a
``repr``-with-memory-addresses-stripped fallback for everything else.

When :meth:`step` is passed an ``idempotency_key`` — as the ReAct loop
does for every tool call, deriving it from the tool name + args — that
key takes over as the whole journal key, namespaced under ``idem:`` so
a content-hash can never collide with a positional step name like
``tool_call_<turn>_<slot>``. This is what makes two *identical* tool
calls in different turns (or retried calls that differ only in a fresh
per-attempt ``call_id``) dedupe to a single side-effecting execution
within a session, rather than re-running because their positional
names differ.

Delivery semantics — **at-least-once**, not exactly-once:
    A step's side effect completes *before* its journal entry is
    written. A crash in the window between side-effect completion and
    the journal write re-executes the step on resume. Steps with
    non-idempotent side effects must tolerate duplication (or be given
    an ``idempotency_key`` honoured by the downstream system).

Streaming granularity:
    ``stream_step()`` buffers chunks and journals them only after the
    underlying stream is fully drained. Resume therefore works
    *between* steps, not mid-stream: a run that dies mid-stream replays
    that stream from the start. If the consumer abandons the stream
    before draining it (or the producer raises), nothing is journaled —
    partial streams are never recorded as complete.

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
import hashlib
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel

from ._signals import SignalMailbox
from .journal import (
    Checkpoint,
    CheckpointMeta,
    InMemoryJournalStore,
    JournalStore,
)

_current_session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_jeeves_runtime_session",
    default=None,
)

# CPython default reprs embed the object's memory address
# ("<Foo object at 0x104f3b010>"), which changes every process. Strip
# it so fallback tokens stay stable across a crash-resume boundary.
_MEM_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _stable_token(value: Any) -> str:
    """Deterministic, process-stable string form of a step argument."""
    if isinstance(value, BaseModel):
        return f"{type(value).__name__}:{value.model_dump_json()}"
    if isinstance(value, Mapping):
        items = sorted(
            (_stable_token(k), _stable_token(v)) for k, v in value.items()
        )
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_stable_token(v) for v in value) + "]"
    if isinstance(value, set | frozenset):
        return "set[" + ",".join(sorted(_stable_token(v) for v in value)) + "]"
    if value is None or isinstance(value, str | bytes | bool | int | float):
        return repr(value)
    return _MEM_ADDR_RE.sub("", repr(value))


def _step_key(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    idempotency_key: str | None = None,
) -> str:
    """Journal key: explicit idempotency key, or name + fingerprint.

    An explicit ``idempotency_key`` becomes the whole key (under the
    ``idem:`` namespace) so logically identical steps dedupe across
    turns regardless of their positional ``name``. Otherwise the key
    is the name plus a stable hash of the inputs, so the same step
    name with changed inputs re-executes instead of replaying stale
    results.
    """
    if idempotency_key is not None:
        return f"idem:{idempotency_key}"
    payload = _stable_token(args) + "|" + _stable_token(kwargs)
    digest = hashlib.sha256(
        payload.encode("utf-8", "surrogatepass")
    ).hexdigest()
    return f"{name}@{digest[:16]}"


class JournaledSession:
    """The handle yielded by :meth:`JournaledRuntime.session`.

    Carries a per-session :class:`SignalMailbox`: ``deliver`` enqueues
    a named signal, ``wait_signal`` parks until one arrives, and
    ``poll_signal`` is the non-blocking pop. The mailbox lives only in
    process memory — signals are not journaled and are dropped when
    the session's last open context exits.
    """

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self._mailbox = SignalMailbox()

    async def deliver(self, name: str, payload: Any) -> None:
        self._mailbox.deliver(name, payload)

    async def wait_signal(self, name: str) -> Any:
        return await self._mailbox.wait(name)

    def poll_signal(self, name: str) -> Any | None:
        return self._mailbox.poll(name)


class JournaledRuntime:
    """Runtime that journals every step's result for replay.

    Pass any :class:`JournalStore` (in-memory for tests, sqlite for
    durable single-process use, future Postgres/DBOS adapters for
    multi-process / multi-host).

    ``max_checkpoints_per_session`` only applies when no ``store`` is
    given (it configures the default :class:`InMemoryJournalStore`);
    an explicit store carries its own retention setting.
    """

    name = "journaled"

    def __init__(
        self,
        store: JournalStore | None = None,
        *,
        max_checkpoints_per_session: int = 20,
    ) -> None:
        self._store: JournalStore = (
            store
            if store is not None
            else InMemoryJournalStore(
                max_checkpoints_per_session=max_checkpoints_per_session
            )
        )
        self._sessions: dict[str, JournaledSession] = {}
        self._session_refs: dict[str, int] = {}

    @property
    def store(self) -> JournalStore:
        return self._store

    # ---- session lifecycle ----------------------------------------------

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[JournaledSession]:
        """Open (or re-enter) a session; in-memory session state — the
        signal mailbox — is discarded when the last open context for
        this ``session_id`` exits. Journal/checkpoint data lives in
        the store and is unaffected."""
        token = _current_session_var.set(session_id)
        sess = self._get_session(session_id)
        self._session_refs[session_id] = (
            self._session_refs.get(session_id, 0) + 1
        )
        try:
            yield sess
        finally:
            _current_session_var.reset(token)
            remaining = self._session_refs.get(session_id, 1) - 1
            if remaining <= 0:
                self._session_refs.pop(session_id, None)
                self._sessions.pop(session_id, None)
            else:
                self._session_refs[session_id] = remaining

    def _get_session(self, session_id: str) -> JournaledSession:
        return self._sessions.setdefault(
            session_id, JournaledSession(session_id)
        )

    # ---- signals ----------------------------------------------------------

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

    # ---- checkpoints -------------------------------------------------------

    async def put_checkpoint(self, cp: Checkpoint) -> None:
        """Persist a transcript snapshot; store retention prunes the
        session's oldest beyond its configured limit."""
        await self._store.put_checkpoint(cp)

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Checkpoint | None:
        return await self._store.get_checkpoint(session_id, checkpoint_id)

    async def get_latest_checkpoint(
        self, session_id: str
    ) -> Checkpoint | None:
        return await self._store.get_latest_checkpoint(session_id)

    async def list_checkpoints(
        self, session_id: str, limit: int = 50
    ) -> list[CheckpointMeta]:
        return await self._store.list_checkpoints(session_id, limit=limit)

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

        The journal key is ``name`` plus a fingerprint of
        ``args``/``kwargs``: same session + name + inputs ⇒ cached
        result is returned without re-executing; changed inputs ⇒ the
        step runs fresh and records a new entry. When
        ``idempotency_key`` is supplied it becomes the whole key
        (under the ``idem:`` namespace) so two calls with the same key
        dedupe to a single execution regardless of their positional
        ``name``.

        Caveat — this is the *defined* meaning of an idempotency key
        here: a tool whose result varies per call with identical args
        (a clock, a counter, ``random()``, a live API read) will
        **replay** its first recorded result within the session rather
        than re-execute. That determinism is exactly what a durable,
        crash-resumable runtime requires; pass no ``idempotency_key``
        for steps that must genuinely re-run on every distinct call.
        At-least-once: a crash after ``fn`` completes but before the
        journal write re-executes ``fn`` on resume.
        """
        session_id = _current_session_var.get()
        if session_id is None:
            return await fn(*args, **kwargs)

        key = _step_key(name, args, kwargs, idempotency_key)
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
        """Replay a journaled stream or record a fresh one.

        Chunks are buffered and journaled in one write *after* the
        underlying stream is fully drained — resume granularity is
        per-step, never mid-stream. If the consumer abandons the
        stream early (``GeneratorExit`` at a ``yield``) or the
        producer raises, the ``put_stream`` below is never reached,
        so a partial stream is never recorded as complete.
        """
        session_id = _current_session_var.get()
        if session_id is None:
            async for chunk in fn(*args, **kwargs):
                yield chunk
            return

        key = _step_key(name, args, kwargs)
        cached = await self._store.get_stream(session_id, key)
        if cached is not None:
            for chunk in cached:
                yield chunk
            return

        recorded: list[Any] = []
        async for chunk in fn(*args, **kwargs):
            recorded.append(chunk)
            yield chunk
        await self._store.put_stream(session_id, key, recorded)
