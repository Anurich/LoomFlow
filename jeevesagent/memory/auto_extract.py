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

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from ..core.protocols import Memory, Telemetry
from ..core.types import (
    Episode,
    Fact,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
    Message,
)
from .consolidator import Consolidator

__all__ = ["AutoExtractMemory"]


_log = logging.getLogger("jeevesagent.memory.auto_extract")

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
    ) -> None:
        self._inner = inner
        self._consolidator = consolidator
        self._on_extract_error = on_extract_error
        self._telemetry = telemetry
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
        """
        result: str = await self._inner.remember(episode)
        await self._maybe_extract(episode)
        return result

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
                        "jeeves.auto_extract.duration_ms",
                        duration_ms,
                        user_id=episode.user_id,
                        status=status,
                    )
                    await self._telemetry.emit_metric(
                        "jeeves.auto_extract.invocations",
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
