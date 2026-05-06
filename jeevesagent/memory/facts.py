"""Bi-temporal fact store.

The store holds :class:`Fact` instances — semantic ``(subject,
predicate, object)`` claims extracted from episodes by a
:class:`Consolidator`.

Bi-temporal contract:

* ``valid_from`` / ``valid_until`` are when the fact was true *in the
  world*. ``valid_until = None`` means "still valid now".
* ``recorded_at`` is when *we* learned the fact (when the consolidator
  ran).

On :meth:`InMemoryFactStore.append`, conflicts are resolved by
*supersession*: if there's an existing currently-valid fact with the
same ``(subject, predicate)`` but different ``object``, its
``valid_until`` is set to the new fact's ``valid_from``. This is the
Zep-style temporal graph behaviour — old beliefs aren't deleted, they
get "closed off" so we can still reason about what was true at any
historical moment.

Today's only backend is :class:`InMemoryFactStore`. Postgres / sqlite
fact stores are a follow-up — the protocol is stable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

import anyio

from ..core.protocols import Embedder
from ..core.types import Fact


@runtime_checkable
class FactStore(Protocol):
    """Storage surface for bi-temporal facts."""

    async def append(self, fact: Fact) -> str: ...

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
    ) -> list[Fact]: ...

    async def recall_text(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
    ) -> list[Fact]: ...

    async def all_facts(self) -> list[Fact]: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryFactStore:
    """Dict-backed bi-temporal fact store.

    All operations are coordinated by an :class:`anyio.Lock` so
    concurrent appends from the consolidator and reads from the agent
    loop don't tear the index.

    When an ``embedder`` is supplied, every appended fact's triple
    (``"subject predicate object"``) is embedded and stored alongside
    the fact, and :meth:`recall_text` ranks by cosine similarity
    against the query's embedding. When no embedder is given,
    :meth:`recall_text` falls back to token-overlap matching.
    """

    def __init__(self, *, embedder: Embedder | None = None) -> None:
        self._facts: dict[str, Fact] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._embedder = embedder
        self._lock = anyio.Lock()

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    # ---- mutation --------------------------------------------------------

    async def append(self, fact: Fact) -> str:
        """Append a fact, invalidating any superseded predecessors.

        Supersession rule: any existing fact with matching subject +
        predicate, currently valid (``valid_until is None``), and a
        different ``object`` gets its ``valid_until`` set to the new
        fact's ``valid_from``.
        """
        # Embed outside the lock — embedders may make network calls.
        embedding: list[float] | None = None
        if self._embedder is not None:
            embedding = await self._embedder.embed(_triple_text(fact))

        async with self._lock:
            for existing_id, existing in list(self._facts.items()):
                if existing.subject != fact.subject:
                    continue
                if existing.predicate != fact.predicate:
                    continue
                if existing.valid_until is not None:
                    continue  # already superseded
                if existing.object == fact.object:
                    continue  # same claim — don't invalidate
                # Close off the old fact's validity window.
                self._facts[existing_id] = existing.model_copy(
                    update={"valid_until": fact.valid_from}
                )
            self._facts[fact.id] = fact
            if embedding is not None:
                self._embeddings[fact.id] = embedding
            return fact.id

    async def append_many(self, facts: Iterable[Fact]) -> list[str]:
        ids: list[str] = []
        for f in facts:
            ids.append(await self.append(f))
        return ids

    # ---- queries ---------------------------------------------------------

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
    ) -> list[Fact]:
        async with self._lock:
            results = list(self._facts.values())

        if subject is not None:
            results = [f for f in results if f.subject == subject]
        if predicate is not None:
            results = [f for f in results if f.predicate == predicate]
        if object_ is not None:
            results = [f for f in results if f.object == object_]
        if valid_at is not None:
            results = [f for f in results if _is_valid_at(f, valid_at)]

        # Most recently recorded first.
        results.sort(key=lambda f: f.recorded_at, reverse=True)
        return results[:limit]

    async def recall_text(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
    ) -> list[Fact]:
        """Rank facts against ``query``.

        With an embedder configured: cosine-similarity over the query's
        embedding vs each fact triple's stored embedding. Without one:
        token-overlap with a small stop-word list (longer overlaps
        win, ties break by shorter haystack = more specific match).
        """
        async with self._lock:
            facts = list(self._facts.values())
            embeddings = dict(self._embeddings)

        if valid_at is not None:
            facts = [f for f in facts if _is_valid_at(f, valid_at)]

        if self._embedder is not None:
            return await self._recall_text_embedding(
                query, facts, embeddings, limit
            )
        return self._recall_text_tokens(query, facts, limit)

    async def _recall_text_embedding(
        self,
        query: str,
        facts: list[Fact],
        embeddings: dict[str, list[float]],
        limit: int,
    ) -> list[Fact]:
        if not facts:
            return []
        assert self._embedder is not None
        query_emb = await self._embedder.embed(query)
        scored: list[tuple[float, Fact]] = []
        for f in facts:
            emb = embeddings.get(f.id)
            if emb is None:
                continue
            scored.append((_cosine(query_emb, emb), f))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [f for _, f in scored[:limit]]

    def _recall_text_tokens(
        self,
        query: str,
        facts: list[Fact],
        limit: int,
    ) -> list[Fact]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            facts.sort(key=lambda f: f.recorded_at, reverse=True)
            return facts[:limit]

        scored: list[tuple[int, int, Fact]] = []
        for f in facts:
            haystack = f"{f.subject} {f.predicate} {f.object}"
            haystack_tokens = _tokenize(haystack)
            overlap = sum(1 for t in query_tokens if t in haystack_tokens)
            if overlap > 0:
                # Higher overlap first; shorter haystack second.
                scored.append((-overlap, len(haystack), f))
        scored.sort()
        return [f for _, _, f in scored[:limit]]

    async def all_facts(self) -> list[Fact]:
        async with self._lock:
            return list(self._facts.values())

    async def aclose(self) -> None:
        return None

    # ---- introspection (test helper) ------------------------------------

    def snapshot(self) -> dict[str, Fact]:
        return dict(self._facts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_at(fact: Fact, when: datetime) -> bool:
    if when < fact.valid_from:
        return False
    if fact.valid_until is None:
        return True
    return when < fact.valid_until


def _triple_text(fact: Fact) -> str:
    """Canonical string for embedding: ``subject predicate object``."""
    return f"{fact.subject} {fact.predicate} {fact.object}"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _tokenize(text: str) -> set[str]:
    """Lowercase, alpha-numeric token set; underscores split too.

    Tokens shorter than 2 characters and a small stop-word list are
    dropped so naive queries like ``"the user's name"`` still surface
    the right facts.
    """
    out: set[str] = set()
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                token = "".join(buf)
                if len(token) >= 2 and token not in _STOP_WORDS:
                    out.add(token)
            buf = []
    if buf:
        token = "".join(buf)
        if len(token) >= 2 and token not in _STOP_WORDS:
            out.add(token)
    return out


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "this",
        "that",
        "what",
        "tell",
        "you",
        "are",
        "is",
        "be",
        "of",
        "to",
        "in",
        "on",
        "an",
        "or",
        "me",
        "my",
        "us",
        "our",
        "by",
        "as",
        "at",
        "it",
        "its",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
    }
)
