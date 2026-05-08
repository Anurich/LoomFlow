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
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from ..core.protocols import Memory
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
    ) -> None:
        self._inner = inner
        self._consolidator = consolidator
        self._on_extract_error = on_extract_error

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
        a critical-path dependency."""
        store = self.facts
        if store is None:
            return
        try:
            await self._consolidator.consolidate([episode], store=store)
        except Exception as exc:  # noqa: BLE001 — best-effort by design
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
