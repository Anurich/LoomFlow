"""Lazy-construction wrapper for async-connect :class:`Memory` backends.

The framework's resolver lets users write::

    Agent("...", model="...", memory="postgres://prod-db/agent")

But ``PostgresMemory.connect(...)`` and ``RedisMemory.connect(...)``
are async — they open a pool / client on the wire. The
:class:`Agent` constructor is synchronous, so we need a way to defer
the connection until the agent loop is actually running.

:class:`LazyMemory` is that bridge. It:

* takes an **async builder coroutine** that returns the real backend
  (e.g. ``lambda: PostgresMemory.connect(dsn)``)
* holds it un-called until the first protocol method is invoked
* connects exactly once, caches the instance, then proxies every
  subsequent call straight through

So users see a regular ``Memory`` from the constructor onward; the
network round-trip happens on the first ``agent.run`` (where any
connection error surfaces as :class:`MemoryStoreError`, not a sync
exception in user-side construction code).

Construction is also safe under structured concurrency: the first-
use path is wrapped in an ``anyio.Lock`` so concurrent ``agent.run``
calls don't open the pool twice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import anyio

from ..core.errors import MemoryStoreError
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

__all__ = ["LazyMemory"]


class LazyMemory:
    """Defer construction of an async-built :class:`Memory` until first
    use.

    Users rarely instantiate this directly — it's what the
    :func:`_resolve_memory` resolver returns when given a
    ``postgres://`` or ``redis://`` URL. Pass a zero-arg async
    callable that builds the real backend; everything else
    (working / remember / recall / facts / session_messages /
    consolidate) is forwarded once that callable resolves.
    """

    def __init__(
        self,
        builder: Callable[[], Awaitable[Any]],
        *,
        description: str = "memory",
    ) -> None:
        self._builder = builder
        self._description = description
        self._inner: Any | None = None
        self._lock = anyio.Lock()

    @property
    def is_ready(self) -> bool:
        """``True`` once the backend has been constructed and cached."""
        return self._inner is not None

    @property
    def description(self) -> str:
        """Human-readable label (e.g. ``"postgres://prod-db/agent"``)
        — used in error messages so users can tell which Memory
        failed to connect."""
        return self._description

    async def _resolve(self) -> Any:
        """Build the inner backend if needed, return the cached
        instance otherwise. Connection errors are wrapped in
        :class:`MemoryStoreError` so callers don't have to catch
        backend-specific exceptions."""
        if self._inner is not None:
            return self._inner
        async with self._lock:
            # Re-check inside the lock — another waiter may have just
            # finished while we waited for it. mypy's narrowing
            # doesn't follow async lock semantics so we silence the
            # "unreachable" complaint here.
            if self._inner is not None:  # type: ignore[unreachable]
                return self._inner  # type: ignore[unreachable]
            try:
                self._inner = await self._builder()
            except Exception as exc:  # noqa: BLE001 — wrap any backend exc
                raise MemoryStoreError(
                    f"failed to connect lazy memory ({self._description}): "
                    f"{exc}"
                ) from exc
        return self._inner

    # ---- Memory protocol -------------------------------------------------
    #
    # Each method awaits the backend resolution then forwards. Doing
    # this by hand (rather than ``__getattr__``) keeps mypy happy and
    # keeps the protocol-conformance checker honest — if Memory grows
    # a new method, this file needs an explicit update.

    async def working(
        self, *, user_id: str | None = None
    ) -> list[MemoryBlock]:
        inner = await self._resolve()
        result: list[MemoryBlock] = await inner.working(user_id=user_id)
        return result

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        inner = await self._resolve()
        await inner.update_block(name, content, user_id=user_id)

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        inner = await self._resolve()
        await inner.append_block(name, content, user_id=user_id)

    async def remember(self, episode: Episode) -> str:
        inner = await self._resolve()
        result: str = await inner.remember(episode)
        return result

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
    ) -> list[Episode]:
        inner = await self._resolve()
        result: list[Episode] = await inner.recall(
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
        # Pass through to the wrapped backend's hybrid recall when
        # available; otherwise wrap raw recall results with neutral
        # scores. Lazy resolution still happens in either path.
        inner = await self._resolve()
        inner_scored = getattr(inner, "recall_scored", None)
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
        eps = await inner.recall(
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
        inner = await self._resolve()
        result: list[Fact] = await inner.recall_facts(
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
        inner = await self._resolve()
        result: list[Message] = await inner.session_messages(
            session_id, user_id=user_id, limit=limit
        )
        return result

    async def consolidate(self) -> None:
        inner = await self._resolve()
        await inner.consolidate()

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        inner = await self._resolve()
        result: MemoryProfile = await inner.profile(user_id=user_id)
        return result

    async def forget(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
    ) -> int:
        inner = await self._resolve()
        result: int = await inner.forget(
            user_id=user_id, session_id=session_id, before=before
        )
        return result

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        inner = await self._resolve()
        result: MemoryExport = await inner.export(user_id=user_id)
        return result

    @property
    def facts(self) -> Any | None:
        """Direct access to the inner backend's fact store (if any).

        Reading this BEFORE the backend has connected returns
        ``None`` — the connection deliberately hasn't happened yet.
        Once the backend is resolved (after the first ``agent.run``
        or an explicit ``await mem._resolve()``), this returns the
        live ``FactStore``. Power-user escape hatch; most callers
        go through :meth:`recall_facts`.
        """
        if self._inner is None:
            return None
        return getattr(self._inner, "facts", None)

    async def aclose(self) -> None:
        """Close the inner backend if it was constructed.

        Safe to call when the backend was never resolved (no-op).
        """
        if self._inner is None:
            return
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()
