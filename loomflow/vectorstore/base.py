"""VectorStore protocol + shared types and helpers.

Every concrete vector store implements the :class:`VectorStore`
protocol — a small async surface (add / delete / search /
search_by_vector / count / get_by_ids). Backends differ in storage
and ANN algorithm, but the interface is identical so swapping
``InMemoryVectorStore`` for ``ChromaVectorStore`` / etc. is a
one-line change.

# Filtering

The ``filter`` argument to :meth:`search` is a Mongo-style query
expression — see :mod:`loomflow.vectorstore._filter` for the
operator reference. Common shapes::

    {"source": "report.pdf"}                 # equality shorthand
    {"page": {"$gte": 5}}                    # range
    {"tag": {"$in": ["draft", "final"]}}     # membership
    {"$and": [{"a": 1}, {"b": 2}]}           # composition

# Diversity (MMR)

:meth:`search` accepts ``diversity: float | None`` in [0, 1] for
Maximal Marginal Relevance reranking. ``None`` (default) gives
plain top-k by similarity. ``0.0`` is identical to ``None``;
``1.0`` is maximum diversity. Most users want ``0.3``..``0.5``
when they want diversity at all.

We picked the 0..1 diversity scale (rather than LangChain's
inverted ``lambda_mult``) because "more diverse → bigger number"
is intuitive and "fully relevant" is the natural zero state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..loader.base import Chunk
from ._filter import evaluate_filter


@dataclass
class SearchResult:
    """One hit from :meth:`VectorStore.search`.

    * ``chunk`` — the matched chunk (with its full metadata).
    * ``score`` — similarity in [-1, 1] for cosine; backend-
      specific for other distance metrics. Higher = more similar.
    * ``id`` — the store-assigned id (so callers can ``delete()``
      or ``get_by_ids()`` later).
    """

    chunk: Chunk
    score: float
    id: str


@runtime_checkable
class VectorStore(Protocol):
    """Async protocol for vector stores.

    Six methods cover the lifecycle: add (embed + store), delete,
    search (by query string), search_by_vector (precomputed),
    count, get_by_ids.

    Backends that aren't natively async (FAISS, Chroma) wrap their
    sync calls in :func:`anyio.to_thread.run_sync` so they don't
    block the event loop.
    """

    async def add(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed + store ``chunks``. Returns the assigned ids
        (caller-provided or generated)."""
        ...

    async def delete(self, ids: list[str]) -> None:
        """Remove the named chunks. Unknown ids are silently
        skipped (idempotent)."""
        ...

    async def search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
        """Embed ``query`` and return the top-``k`` chunks ranked
        by similarity. ``filter`` (optional) restricts candidates
        by metadata. ``diversity`` (optional, 0..1) enables MMR
        reranking for varied results."""
        ...

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
        """Same as :meth:`search` but with a precomputed query
        vector."""
        ...

    async def count(self) -> int:
        """Number of chunks currently in the store."""
        ...

    async def get_by_ids(
        self, ids: list[str]
    ) -> list[Chunk]:
        """Fetch chunks by id, in the same order as ``ids``.
        Unknown ids are skipped (the result may be shorter than
        the input)."""
        ...


# ---------------------------------------------------------------------------
# Backwards-compat helper (kept for the existing test file's import path).
# Delegates to :func:`evaluate_filter` so old callers transparently get the
# expanded operator support.
# ---------------------------------------------------------------------------


def matches_filter(
    metadata: Mapping[str, Any],
    filter: Mapping[str, Any] | None,
) -> bool:
    """Return True if ``metadata`` satisfies ``filter``.

    Thin wrapper around :func:`evaluate_filter` with the argument
    order our existing tests expect.
    """
    return evaluate_filter(filter, metadata)


# ---------------------------------------------------------------------------
# Helpers shared by per-backend factory methods
# ---------------------------------------------------------------------------


def _chunks_from_texts(
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> list[Chunk]:
    """Convert a list of raw text strings into :class:`Chunk`
    instances, validating that ``metadatas`` (when supplied) has
    matching length. Used by every backend's :meth:`from_texts`."""
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError(
            f"metadatas length ({len(metadatas)}) must match "
            f"texts length ({len(texts)})"
        )
    return [
        Chunk(
            content=text,
            metadata=(
                dict(metadatas[i]) if metadatas is not None else {}
            ),
        )
        for i, text in enumerate(texts)
    ]
