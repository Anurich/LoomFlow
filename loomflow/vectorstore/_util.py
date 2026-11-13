"""Shared helpers for vector-store backends.

Every backend previously carried private copies of the same
snippets: cosine similarity, the ``embed_batch``-with-per-item
fallback loop, the ids/chunks length validation and the argument
checks of the shared contract (see :class:`~loomflow.vectorstore.
base.VectorStore`). One copy here keeps the four backends from
drifting apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypeVar

from ..core.ids import new_id
from ..core.protocols import Embedder

__all__ = [
    "cosine",
    "dedupe_last",
    "embed_all",
    "embedder_identity",
    "resolve_ids",
    "validate_k",
    "validate_query_vector",
]

_T = TypeVar("_T")


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, range ``[-1, 1]``.

    Returns ``0.0`` on zero-norm inputs (which would otherwise be
    ``nan``); raises on mismatched lengths (``zip(strict=True)``) —
    inside a single store every vector comes from the same embedder,
    so a length mismatch is a bug worth surfacing.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def embed_all(
    embedder: Embedder, texts: list[str]
) -> list[list[float]]:
    """Embed ``texts`` via :meth:`Embedder.embed_batch`, falling back
    to a per-item ``embed`` loop for embedders without batch support.

    "Without batch support" means the method is absent or raises
    :class:`NotImplementedError`. An :class:`AttributeError` raised
    *inside* a real ``embed_batch`` is a bug in that embedder and
    propagates (it used to be swallowed and silently retried one
    text at a time). A result of the wrong length raises too, so a
    short batch can never mis-pair vectors with chunks.

    Splitting into provider-sized requests is the embedder's job
    (see ``batch_size`` on the bundled embedders).
    """
    batch = getattr(embedder, "embed_batch", None)
    vectors: list[list[float]]
    if batch is None:
        vectors = [await embedder.embed(t) for t in texts]
    else:
        try:
            vectors = [list(v) for v in await batch(texts)]
        except NotImplementedError:
            vectors = [await embedder.embed(t) for t in texts]
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for "
            f"{len(texts)} texts"
        )
    return vectors


def embedder_identity(embedder: Any) -> tuple[str | None, int | None]:
    """``(name, dimensions)`` of an embedder, ``None`` for whatever a
    custom embedder doesn't declare."""
    name = getattr(embedder, "name", None)
    dims = getattr(embedder, "dimensions", None)
    return (
        name if isinstance(name, str) and name else None,
        dims if isinstance(dims, int) and not isinstance(dims, bool) else None,
    )


def resolve_ids(ids: list[str] | None, n_chunks: int) -> list[str]:
    """Validate caller-supplied ``ids`` against the chunk count (or
    generate fresh ids when ``None``). Returns the assigned ids."""
    if ids is not None:
        if len(ids) != n_chunks:
            raise ValueError(
                f"ids length ({len(ids)}) must match chunks "
                f"length ({n_chunks})"
            )
        return list(ids)
    return [new_id("vec") for _ in range(n_chunks)]


def dedupe_last(ids: Sequence[str], items: Sequence[_T]) -> tuple[list[str], list[_T]]:
    """Drop all but the LAST occurrence of each id (an id repeated in
    one ``add`` call upserts onto itself, so the last one wins).
    Order of first appearance is kept."""
    if len(set(ids)) == len(ids):
        return list(ids), list(items)
    last = {cid: i for i, cid in enumerate(ids)}
    keep = [i for i, cid in enumerate(ids) if last[cid] == i]
    return [ids[i] for i in keep], [items[i] for i in keep]


def validate_k(k: Any, *, name: str = "k") -> int:
    """The shared contract: a result count must be a positive int."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"{name} must be a positive integer, got {k!r}")
    if k <= 0:
        raise ValueError(f"{name} must be >= 1, got {k}")
    return int(k)


def validate_query_vector(
    vector: Sequence[float], *, dimensions: int | None = None
) -> None:
    """Reject query vectors that can only produce garbage: empty,
    NaN / infinite components, or (when the store's dimension is
    known) the wrong length."""
    if len(vector) == 0:
        raise ValueError("query vector is empty")
    if not all(math.isfinite(x) for x in vector):
        raise ValueError(
            "query vector contains NaN or infinite values"
        )
    if dimensions is not None and len(vector) != dimensions:
        raise ValueError(
            f"query vector has {len(vector)} dimensions but the store "
            f"holds {dimensions}-dimensional vectors. Query with the "
            "embedder the store was built with, or re-index the "
            "documents with the new embedder."
        )
