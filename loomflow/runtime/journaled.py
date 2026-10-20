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

When :meth:`step` is passed an ``idempotency_key`` it takes over as
the whole journal key, namespaced under ``idem:``. The ReAct loop
passes the call's position and content (``tool_call_<turn>_<slot>``
plus a hash of the tool name and args), so on resume the same call
replays — even when its ``call_id`` differs because the model was
asked again — while a retry in a later turn, or an intentional second
identical call, runs.

Runs and epochs:
    A run's steps replay only into that same run resumed: re-running
    with the same session and inputs after a crash. When a run ends
    (its outermost ``session`` context exits without an exception)
    the session's *epoch* advances, and later runs on the session key
    their steps under the new epoch — so a finished run's results are
    never replayed into the next request, even an identical one.
    :func:`resuming_last_run` (what ``Agent.resume`` uses without a
    checkpoint) continues the most recent run instead, finished or not.
    A run opened inside another run — a sub-agent, a debater, a
    workflow node — keys its steps under the enclosing run (and how
    many times that run has opened the sub-session), so resuming the
    outer run replays the sub-runs' work as well, and two calls to one
    persistent worker in a run stay distinct.
    Each ``(session, user)`` runs one outermost run at a time in this
    process: a second concurrent run waits for the first to end.

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
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import anyio
from pydantic import BaseModel

from ..core.context import get_run_context
from ._signals import SignalMailbox
from .journal import (
    Checkpoint,
    CheckpointMeta,
    InMemoryJournalStore,
    JournalStore,
    journal_scope,
)

_current_session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_jeeves_runtime_session",
    default=None,
)

#: Journal key holding a scope's current run epoch.
EPOCH_KEY = "__run_epoch__"
#: Journal key holding the epoch of the scope's most recent run.
LAST_RUN_KEY = "__last_run_epoch__"

_resume_last_run_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_loom_resume_last_run", default=False
)


@contextmanager
def resuming_last_run() -> Iterator[None]:
    """Sessions opened inside this context continue the scope's most
    recent run — replaying its steps even if it finished — instead of
    starting a new one. ``Agent.resume`` uses it when it has no
    checkpoint to resume from."""
    token = _resume_last_run_var.set(True)
    try:
        yield
    finally:
        _resume_last_run_var.reset(token)


class _OpenRun:
    """A run in progress: the key namespace its steps use."""

    def __init__(
        self,
        scope: str,
        session_id: str,
        ns: str,
        parent: _OpenRun | None,
        *,
        epoch: int = 0,
        next_epoch: int = 0,
    ) -> None:
        self.scope = scope
        self.session_id = session_id
        self.ns = ns
        self.parent = parent
        self.epoch = epoch
        self.next_epoch = next_epoch
        self.children: dict[str, int] = {}
        self.touched = False

    def ancestors(self) -> Iterator[_OpenRun]:
        run: _OpenRun | None = self
        while run is not None:
            yield run
            run = run.parent


#: The innermost run open in this context (any runtime instance).
_open_run_var: contextvars.ContextVar[_OpenRun | None] = contextvars.ContextVar(
    "_loom_open_run", default=None
)


class _ScopeLock:
    def __init__(self) -> None:
        self.lock = anyio.Lock()
        self.users = 0


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


def _scope(session_id: str) -> str:
    """Journal scope for ``session_id`` under the current run's user."""
    return journal_scope(session_id, get_run_context().user_id)


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
        self._scope_locks: dict[str, _ScopeLock] = {}

    @property
    def store(self) -> JournalStore:
        return self._store

    # ---- session lifecycle ----------------------------------------------

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[JournaledSession]:
        """Open (or re-enter) a session for one run.

        The outermost run on a ``(session, current user)`` scope waits
        for any other run on it, and on a clean exit advances the
        scope's epoch. A run opened inside another run (a sub-agent)
        keys its steps under the parent run instead — so resuming the
        parent replays the sub-agent's work too — and doesn't advance
        anything. See the module docstring. In-memory session state —
        the signal mailbox — is discarded when the last open context
        for this ``session_id`` exits. Journal/checkpoint data lives
        in the store and is unaffected."""
        scope = _scope(session_id)
        parent = _open_run_var.get()
        if parent is not None and parent.scope == scope and parent.session_id == session_id:
            # Re-entering the open run (nested contexts on one session).
            async with self._enter(session_id) as sess:
                yield sess
            return
        if parent is not None:
            k = parent.children.get(scope, 0) + 1
            parent.children[scope] = k
            ns = hashlib.sha256(f"{parent.ns}|{scope}|{k}".encode()).hexdigest()[:16]
            run = _OpenRun(scope, session_id, ns, parent)
            if any(r.scope == scope for r in parent.ancestors()):
                # An ancestor holds this scope's lock: don't wait on it.
                async with self._run(run, session_id) as sess:
                    yield sess
                return

        holder = self._scope_locks.setdefault(scope, _ScopeLock())
        holder.users += 1
        try:
            async with holder.lock:
                if parent is None:
                    entry = await self._store.get_step(scope, EPOCH_KEY)
                    next_epoch = entry.value if entry is not None else 0
                    epoch = next_epoch
                    if _resume_last_run_var.get():
                        last = await self._store.get_step(scope, LAST_RUN_KEY)
                        if last is not None:
                            epoch = last.value
                    run = _OpenRun(
                        scope,
                        session_id,
                        str(epoch) if epoch else "",
                        None,
                        epoch=epoch,
                        next_epoch=next_epoch,
                    )
                async with self._run(run, session_id) as sess:
                    yield sess
        finally:
            holder.users -= 1
            if holder.users == 0:
                self._scope_locks.pop(scope, None)

    @asynccontextmanager
    async def _run(self, run: _OpenRun, session_id: str) -> AsyncIterator[JournaledSession]:
        token = _open_run_var.set(run)
        clean = False
        try:
            async with self._enter(session_id) as sess:
                yield sess
            clean = True
        finally:
            _open_run_var.reset(token)
            if clean and run.touched and run.parent is None:
                with anyio.CancelScope(shield=True):
                    await self._store.put_step(
                        run.scope, EPOCH_KEY, max(run.next_epoch, run.epoch + 1)
                    )

    @asynccontextmanager
    async def _enter(self, session_id: str) -> AsyncIterator[JournaledSession]:
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

    async def _run_key(self, scope: str, key: str) -> str:
        """``key`` within the open run's namespace (an outermost run's
        epoch 0 is unprefixed, so journals written before epochs
        existed still resume)."""
        run = _open_run_var.get()
        if run is None or run.scope != scope:
            return key
        if not run.touched:
            run.touched = True
            if run.parent is None:
                await self._store.put_step(scope, LAST_RUN_KEY, run.epoch)
        return f"{run.ns}/{key}" if run.ns else key

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
        """Run ``fn`` once per ``(session, user, journal_key)`` and journal it.

        Steps are stored under :func:`~loomflow.runtime.journal.journal_scope`
        of the open session and the current run's ``user_id``, so one
        user's results never replay into another user's run.

        The journal key is ``name`` plus a fingerprint of
        ``args``/``kwargs``: same session + name + inputs ⇒ cached
        result is returned without re-executing; changed inputs ⇒ the
        step runs fresh and records a new entry. When
        ``idempotency_key`` is supplied it becomes the whole key
        (under the ``idem:`` namespace) so two calls with the same key
        dedupe to a single execution regardless of their positional
        ``name``. Either way the key is scoped to the current run
        (see the module docstring): a finished run never replays into
        a later one.

        At-least-once: a crash after ``fn`` completes but before the
        journal write re-executes ``fn`` on resume.
        """
        session_id = _current_session_var.get()
        if session_id is None:
            return await fn(*args, **kwargs)

        scope = _scope(session_id)
        key = await self._run_key(scope, _step_key(name, args, kwargs, idempotency_key))
        cached = await self._store.get_step(scope, key)
        if cached is not None:
            return cached.value

        result = await fn(*args, **kwargs)
        await self._store.put_step(scope, key, result)
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

        scope = _scope(session_id)
        key = await self._run_key(scope, _step_key(name, args, kwargs))
        cached = await self._store.get_stream(scope, key)
        if cached is not None:
            for chunk in cached:
                yield chunk
            return

        recorded: list[Any] = []
        async for chunk in fn(*args, **kwargs):
            recorded.append(chunk)
            yield chunk
        await self._store.put_stream(scope, key, recorded)
