"""Auto-extract wrapper — runs the :class:`Consolidator` on every
remembered episode so the bot extracts and stores structured facts
*automatically* as conversations happen.

The wrapper is what turns ``Agent(memory="sqlite:./bot.db")`` into a
"my bot just remembers things" experience: the user says
"I prefer dark mode" and a turn later the framework has a
``Fact(subject="alice", predicate="prefers", object="dark_mode")``
in its store, partitioned by ``user_id``, ready to surface in
future runs via ``recall_facts``.

Wiring:

* Wraps any :class:`Memory` whose ``.facts`` is not ``None``.
* On every ``remember(episode)`` call: writes the episode through,
  then runs the configured :class:`Consolidator` on JUST that
  episode (single-episode batch), letting the consolidator append
  any extracted facts to ``inner.facts``.
* Extraction is **best-effort**: a failing extract (model error,
  malformed JSON, rate limit) NEVER breaks the run. The wrapper
  logs and moves on; the underlying episode write already
  succeeded.
* Every other Memory protocol method forwards straight through to
  the inner backend.

The :class:`Agent` builds one of these automatically when
``auto_extract=True`` (the default) and the resolved memory has a
fact store. Users who want the today's behaviour pass
``auto_extract=False`` to ``Agent(...)`` — the wrapper simply
isn't applied.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from ..core.protocols import Memory, Telemetry
from ..core.types import (
    Episode,
    EpisodeMatch,
    Fact,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
    Message,
)
from ._hybrid import default_recall_scored
from .consolidator import Consolidator

__all__ = ["AutoExtractMemory"]


_log = logging.getLogger("loomflow.memory.auto_extract")

# Process-wide flag — the "default-on" startup notice only fires
# once no matter how many AutoExtractMemory instances are created
# (multi-Agent processes don't spam the log).
_DEFAULT_ON_NOTICE_EMITTED = False


def _maybe_emit_default_on_notice() -> None:
    """Log a one-shot info-level notice when auto-extract gets
    turned on by default. Production-deployment readers needed
    this — it was happening silently before and ops only noticed
    when fact extraction calls showed up in their LLM bills.

    Idempotent across the whole process; safe to call from every
    AutoExtractMemory constructor on the default-picked path.
    """
    global _DEFAULT_ON_NOTICE_EMITTED
    if _DEFAULT_ON_NOTICE_EMITTED:
        return
    _DEFAULT_ON_NOTICE_EMITTED = True
    _log.info(
        "AutoExtractMemory enabled by default for this model class. "
        "Each remembered episode triggers a small extraction call "
        "to pull (subject, predicate, object) facts. Pass "
        "Agent(auto_extract=False) to disable, or "
        "Agent(auto_extract=True) to silence this notice."
    )


class AutoExtractMemory:
    """Wraps a :class:`Memory` and runs auto fact extraction on every
    ``remember`` call.

    Construct via the :class:`Agent` ``auto_extract=`` kwarg; this
    class isn't normally instantiated by user code. The wrapped
    memory must expose a ``.facts`` attribute (a :class:`FactStore`)
    for extraction to do anything — when ``inner.facts is None``,
    the wrapper still installs cleanly but every extraction is a
    no-op.
    """

    def __init__(
        self,
        inner: Memory,
        consolidator: Consolidator,
        *,
        on_extract_error: Callable[[BaseException], Awaitable[None]] | None = None,
        telemetry: Telemetry | None = None,
        auto_picked: bool = False,
        background: bool = True,
    ) -> None:
        self._inner = inner
        self._consolidator = consolidator
        self._on_extract_error = on_extract_error
        self._telemetry = telemetry
        # ``background=True`` (default in 0.10.20+): ``remember``
        # schedules the LLM fact-extraction as a fire-and-forget
        # task so the caller's ``Agent.run()`` returns the moment
        # the episode is persisted, instead of blocking for 3-10s
        # on the model round-trip. This fixes the "next prompt
        # comes back late" UX cost (observed in interactive REPLs
        # like loom-code: every turn paid a hidden round-trip
        # between the visible response and the next input).
        #
        # ``background=False`` (legacy + tests): ``remember``
        # awaits extraction synchronously. Used by tests that need
        # deterministic completion + by callers that explicitly
        # need facts populated before the caller proceeds.
        self._background = background
        # Pending fire-and-forget extraction tasks tracked so
        # ``aclose()`` can drain them on process shutdown. Using
        # ``asyncio.create_task`` (not anyio's task group) is the
        # right tool here despite loomflow's "anyio everywhere"
        # rule — the rule's rationale is *cancellation
        # propagation*, but fire-and-forget extraction explicitly
        # MUST NOT cancel when the caller's task group does
        # (otherwise we lose facts every time Agent.run() returns).
        # asyncio.create_task attaches to the running event loop
        # at the *loop* lifetime, not the caller's scope — which
        # is exactly what we want here.
        self._pending: set[asyncio.Task[None]] = set()
        if auto_picked:
            _maybe_emit_default_on_notice()

    @property
    def inner(self) -> Memory:
        """The wrapped backend. Power-user introspection — most call
        sites just use the protocol methods."""
        return self._inner

    @property
    def facts(self) -> Any:
        """Forward the inner backend's fact store. Reading this gives
        callers the same access to the bi-temporal store the
        consolidator writes into."""
        return getattr(self._inner, "facts", None)

    # ---- Memory protocol -------------------------------------------------

    async def working(
        self, *, user_id: str | None = None
    ) -> list[MemoryBlock]:
        result: list[MemoryBlock] = await self._inner.working(user_id=user_id)
        return result

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        await self._inner.update_block(name, content, user_id=user_id)

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        await self._inner.append_block(name, content, user_id=user_id)

    async def remember(self, episode: Episode) -> str:
        """Persist the episode, then run auto-extraction.

        The episode write happens first and is the contract — the
        function returns its id even when extraction fails. So the
        consolidator's fragility never leaks into the agent's own
        durability guarantees.

        With ``background=True`` (default), extraction is scheduled
        as a fire-and-forget task and ``remember`` returns the
        moment the inner write completes — caller doesn't pay the
        LLM round-trip latency. With ``background=False``,
        extraction is awaited inline (legacy behaviour, deterministic
        for tests).
        """
        result: str = await self._inner.remember(episode)
        if self._background:
            try:
                task = asyncio.create_task(
                    self._maybe_extract(episode)
                )
                self._pending.add(task)
                # Auto-discard the task from the tracking set when
                # it finishes — keeps the set bounded across long-
                # lived processes (REPL sessions accumulate
                # hundreds of episodes).
                task.add_done_callback(self._pending.discard)
            except RuntimeError:
                # No running loop (defensive — we're inside an
                # async method so this shouldn't happen, but if
                # the caller invoked ``remember`` from a context
                # without a loop, fall back to inline extraction
                # rather than dropping the episode silently).
                await self._maybe_extract(episode)
        else:
            await self._maybe_extract(episode)
        return result

    async def aclose(
        self,
        *,
        # ``timeout`` IS the drain semantic — caller picks how
        # long to wait at shutdown. This is NOT a per-operation
        # deadline that should propagate via cancellation (which
        # is the general case ASYNC109 warns against).
        timeout: float = 30.0,  # noqa: ASYNC109
    ) -> int:
        """Drain pending fire-and-forget extractions before shutdown.

        Call this from your process's shutdown path (REPL exit,
        SIGTERM handler, ``__aexit__`` of a long-lived context) to
        give in-flight fact extractions a bounded chance to
        complete before the event loop closes.

        ``timeout`` is a deliberate API parameter here — the caller
        chooses how long to wait. This is the shutdown-drain
        contract, NOT a per-operation deadline that should
        propagate through cancellation (which is what the ASYNC109
        lint is warning against in the general case).

        Returns the count of tasks still in flight when the
        timeout fired (``0`` = clean drain). No-op when
        ``background=False`` or no extractions are pending.
        """
        if not self._pending:
            return 0
        # Snapshot — _pending is mutated by add_done_callback as
        # tasks finish; iterating it directly would race.
        pending = list(self._pending)
        # ``asyncio.wait`` (not ``wait_for``) is deliberate — wait_for
        # would CANCEL the pending tasks on timeout, which
        # contradicts our "extractions keep running until the loop
        # actually closes" contract. ``wait`` just observes; the
        # tasks live on, and the returned ``not_done`` count tells
        # the caller how many were still in flight at the deadline.
        _, not_done = await asyncio.wait(pending, timeout=timeout)
        return len(not_done)

    async def _maybe_extract(self, episode: Episode) -> None:
        """Run the consolidator on a single episode. Catches every
        exception — auto-extract is a best-effort enhancement, never
        a critical-path dependency.

        Emits two telemetry signals when telemetry is configured:

        * ``jeeves.auto_extract.duration_ms`` (histogram) — wall time
          spent inside the consolidator, in milliseconds. Tagged with
          ``user_id`` and ``status`` (ok / error) so dashboards can
          slice by tenant or by failure rate.
        * ``jeeves.auto_extract.invocations`` (counter) — incremented
          once per extraction attempt, with the same tags.
        """
        store = self.facts
        if store is None:
            return
        started = time.perf_counter()
        status = "ok"
        try:
            await self._consolidator.consolidate([episode], store=store)
        except Exception as exc:  # noqa: BLE001 — best-effort by design
            status = "error"
            _log.warning(
                "auto-extract failed for episode %s (user_id=%s): %s",
                episode.id,
                episode.user_id,
                exc,
            )
            if self._on_extract_error is not None:
                try:
                    await self._on_extract_error(exc)
                except Exception:  # noqa: BLE001
                    # Even the error-callback is best-effort. We
                    # already logged; don't cascade.
                    pass
        finally:
            if self._telemetry is not None:
                duration_ms = (time.perf_counter() - started) * 1000.0
                # Telemetry emit is best-effort too — a broken
                # exporter must not turn a successful extract into a
                # failed remember(). Swallow and log instead.
                try:
                    await self._telemetry.emit_metric(
                        "loom.auto_extract.duration_ms",
                        duration_ms,
                        user_id=episode.user_id,
                        status=status,
                    )
                    await self._telemetry.emit_metric(
                        "loom.auto_extract.invocations",
                        1,
                        user_id=episode.user_id,
                        status=status,
                    )
                except Exception:  # noqa: BLE001
                    pass

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
    ) -> list[Episode]:
        result: list[Episode] = await self._inner.recall(
            query,
            kind=kind,
            limit=limit,
            time_range=time_range,
            user_id=user_id,
        )
        return result

    async def recall_scored(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
        alpha: float = 0.5,
    ) -> list[EpisodeMatch]:
        # Wrapper — pass through to the inner backend's scored
        # recall when it has one (preserves component scores);
        # otherwise wrap raw recall results with neutral scores.
        inner_scored = getattr(self._inner, "recall_scored", None)
        if inner_scored is not None:
            scored: list[EpisodeMatch] = await inner_scored(
                query,
                kind=kind,
                limit=limit,
                time_range=time_range,
                user_id=user_id,
                alpha=alpha,
            )
            return scored
        eps = await self._inner.recall(
            query,
            kind=kind,
            limit=limit,
            time_range=time_range,
            user_id=user_id,
        )
        return default_recall_scored(eps)

    async def recall_facts(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
        user_id: str | None = None,
    ) -> list[Fact]:
        result: list[Fact] = await self._inner.recall_facts(
            query, limit=limit, valid_at=valid_at, user_id=user_id
        )
        return result

    async def session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        result: list[Message] = await self._inner.session_messages(
            session_id, user_id=user_id, limit=limit
        )
        return result

    async def consolidate(self) -> None:
        await self._inner.consolidate()

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        result: MemoryProfile = await self._inner.profile(user_id=user_id)
        return result

    async def forget(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
    ) -> int:
        result: int = await self._inner.forget(
            user_id=user_id, session_id=session_id, before=before
        )
        return result

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        result: MemoryExport = await self._inner.export(user_id=user_id)
        return result
