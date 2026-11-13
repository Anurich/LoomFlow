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

    The contract (every bundled backend enforces it, so swapping one
    for another doesn't change results):

    * **add upserts.** Adding an id that already exists replaces its
      content, metadata and vector — never a duplicate, never the old
      content. An id repeated inside one ``add`` call keeps the last
      occurrence. ``ids`` must be as long as ``chunks``.
    * **k is a positive int.** ``k <= 0`` (or a non-int) raises
      :class:`ValueError` from ``search`` / ``search_by_vector`` /
      ``search_hybrid``. Asking for more than the store holds returns
      what there is.
    * **Query vectors are finite.** A NaN / infinite / empty query
      vector raises :class:`ValueError`; so does one whose length
      differs from the stored vectors (a different embedder).
    * **Score** is cosine similarity, higher-is-better, in
      ``[-1, 1]`` — comparable across backends.
    * **Filters** use one Mongo-style language with one set of value
      semantics — see :mod:`loomflow.vectorstore._filter`. In short:
      types are strict (``2020`` never equals ``"2020"``, ``1`` never
      equals ``True``; ``1 == 1.0``), ``None`` matches a missing or
      ``None`` value, a stored list matches by element (``$eq`` /
      ``$in``), and ``$gt``-style comparisons work number-to-number
      or string-to-string (code-point order).

    Known remaining differences:

    * **Chroma stores scalars only.** ``None`` values are stored as
      an absent key (so ``$exists`` is false for them); lists / dicts
      are stored as JSON text and decoded back on read. Filters on
      them, on ``None``, on string ordering and with ``$exists`` /
      ``$not`` are re-checked in Python over an over-fetched pool
      (widened a few times until ``k`` rows match or the
      collection is exhausted). Whether an ``int`` operand matches a
      stored ``float`` (``5`` vs ``5.0``) is up to the installed
      Chroma version; every other backend treats them as equal.
    * **Postgres** needs filterable metadata keys made of letters,
      digits, ``_``, ``.``, ``-`` and spaces (others raise
      ``FilterError``), and stores non-JSON values as their ``str``.
    * **FAISS post-filters** an over-fetched candidate pool, so a
      very tight filter can return fewer than ``k`` there (InMemory
      scans everything; Postgres filters in the query).
    * ``search_hybrid`` exists on InMemory and Postgres only.
    """

    async def add(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed + store ``chunks``. Returns the assigned ids
        (caller-provided or generated). An existing id is replaced
        (upsert)."""
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
        reranking for varied results. ``k <= 0`` raises
        :class:`ValueError`.

        Where the filter runs: Postgres pushes it into the query and
        scans until ``k`` rows match (pgvector 0.8+ iterative index
        scans; older pgvector filters after the index scan, so a
        tight filter can return fewer). Chroma pushes what its engine
        can express and re-checks the rest (see the class docstring).
        FAISS POST-filters an over-fetched candidate pool, so a very
        tight filter on a large store may return FEWER than ``k``
        even when more matches exist — best-effort, not
        guaranteed-``k``. Prefer Postgres when an exact-``k``
        filtered result matters. A missing key counts as "not equal"
        for ``$ne`` / ``$nin`` / ``$not`` everywhere.
        """
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
        vector (which must be finite and of the store's dimension)."""
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
