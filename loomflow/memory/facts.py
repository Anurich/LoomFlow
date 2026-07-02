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

import warnings
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import anyio

from ..core.protocols import Embedder
from ..core.types import Fact, _normalize_predicate
from ._embedding_util import cosine as _cosine


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
        user_id: str | None = None,
    ) -> list[Fact]: ...

    async def recall_text(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
        user_id: str | None = None,
    ) -> list[Fact]: ...

    async def all_facts(self) -> list[Fact]: ...

    async def delete(
        self,
        *,
        user_id: str | None = None,
        before: datetime | None = None,
    ) -> int: ...

    async def count(self, *, user_id: str | None = None) -> int: ...

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
                # Supersession is namespace-scoped — alice's facts never
                # invalidate bob's, anonymous facts never invalidate
                # named-user facts.
                if existing.user_id != fact.user_id:
                    continue
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
        """Append a batch of facts. Embedder calls are coalesced via
        :meth:`Embedder.embed_batch` when an embedder is configured —
        one network round-trip for the batch instead of N.
        """
        materialised = list(facts)
        if not materialised:
            return []

        # Single batch embedding for all triples up front.
        embeddings: list[list[float] | None]
        if self._embedder is not None:
            triples = [_triple_text(f) for f in materialised]
            embeddings = list(await self._embedder.embed_batch(triples))
        else:
            embeddings = [None] * len(materialised)

        ids: list[str] = []
        for fact, emb in zip(materialised, embeddings, strict=True):
            ids.append(await self._append_with_embedding(fact, emb))
        return ids

    async def _append_with_embedding(
        self,
        fact: Fact,
        embedding: list[float] | None,
    ) -> str:
        """Append using a pre-computed embedding (skip the per-fact
        ``embed()`` call). Same supersession rules as :meth:`append`.
        """
        async with self._lock:
            for existing_id, existing in list(self._facts.items()):
                # Supersession is namespace-scoped (same rule as
                # ``append``): alice's facts never invalidate bob's,
                # anonymous facts never invalidate named-user facts.
                if existing.user_id != fact.user_id:
                    continue
                if existing.subject != fact.subject:
                    continue
                if existing.predicate != fact.predicate:
                    continue
                if existing.valid_until is not None:
                    continue
                if existing.object == fact.object:
                    continue
                self._facts[existing_id] = existing.model_copy(
                    update={"valid_until": fact.valid_from}
                )
            self._facts[fact.id] = fact
            if embedding is not None:
                self._embeddings[fact.id] = embedding
            return fact.id

    # ---- queries ---------------------------------------------------------

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[Fact]:
        async with self._lock:
            results = list(self._facts.values())

        # Hard namespace partition by ``user_id``.
        results = [f for f in results if f.user_id == user_id]
        if subject is not None:
            results = [f for f in results if f.subject == subject]
        if predicate is not None:
            # Match the canonical form stored on Fact.predicate so a
            # query for "Name_Is" finds facts stored as "name_is".
            canonical = _normalize_predicate(predicate)
            results = [f for f in results if f.predicate == canonical]
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
        user_id: str | None = None,
    ) -> list[Fact]:
        """Rank facts against ``query``.

        With an embedder configured: cosine-similarity over the query's
        embedding vs each fact triple's stored embedding. Without one:
        token-overlap with a small stop-word list (longer overlaps
        win, ties break by shorter haystack = more specific match).

        ``user_id`` partitions the candidate set as a hard namespace
        boundary — see :class:`Fact` for semantics.
        """
        async with self._lock:
            facts = list(self._facts.values())
            embeddings = dict(self._embeddings)

        # Footgun protection — see ``InMemoryMemory.recall``.
        if user_id is None and any(f.user_id is not None for f in facts):
            from ..core.context import IsolationWarning
            warnings.warn(
                "FactStore.recall_text called without user_id, but the "
                "store contains facts for one or more named users. The "
                "anonymous bucket is partitioned from named-user "
                "buckets, so this query will only see anonymous facts. "
                "Did you forget to pass user_id=?",
                IsolationWarning,
                stacklevel=3,
            )

        # Hard namespace partition by ``user_id``.
        facts = [f for f in facts if f.user_id == user_id]
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

    async def delete(
        self,
        *,
        user_id: str | None = None,
        before: datetime | None = None,
    ) -> int:
        """Delete every fact in the ``user_id`` partition (optionally
        only those recorded before ``before``). Returns the number of
        facts actually removed. This is the GDPR-forget surface —
        memory backends delegate here instead of poking store
        internals."""
        async with self._lock:
            to_delete = [
                fid
                for fid, f in self._facts.items()
                if f.user_id == user_id
                and (before is None or f.recorded_at < before)
            ]
            for fid in to_delete:
                self._facts.pop(fid, None)
                self._embeddings.pop(fid, None)
            return len(to_delete)

    async def count(self, *, user_id: str | None = None) -> int:
        """Number of facts in the ``user_id`` partition."""
        async with self._lock:
            return sum(
                1 for f in self._facts.values() if f.user_id == user_id
            )

    async def aclose(self) -> None:
        return None

    # ---- introspection (test helper) ------------------------------------

    def snapshot(self) -> dict[str, Fact]:
        return dict(self._facts)


# ---------------------------------------------------------------------------
# Cross-backend GDPR helpers
# ---------------------------------------------------------------------------


async def count_facts(store: Any, *, user_id: str | None) -> int:
    """Count facts in a store's ``user_id`` partition via the
    :meth:`FactStore.count` method when the store implements it,
    falling back to a bounded ``query`` scan for third-party stores
    that predate the protocol method."""
    counter = getattr(store, "count", None)
    if callable(counter):
        return int(await counter(user_id=user_id))
    return len(await store.query(user_id=user_id, limit=100_000))


async def delete_facts(
    store: Any,
    *,
    user_id: str | None,
    before: datetime | None = None,
) -> int:
    """Delete facts in a store's ``user_id`` partition via the
    :meth:`FactStore.delete` method. Stores that don't implement
    ``delete`` (third-party, pre-protocol) return ``0`` — memory
    backends no longer poke store privates on their behalf."""
    deleter = getattr(store, "delete", None)
    if callable(deleter):
        return int(await deleter(user_id=user_id, before=before))
    return 0


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
