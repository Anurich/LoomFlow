"""Shared helpers for vector-store backends.

Every backend previously carried private copies of the same three
snippets: cosine similarity, the ``embed_batch``-with-per-item
fallback loop, and the ids/chunks length validation. One copy here
keeps the four backends from drifting apart.
"""

from __future__ import annotations

import math

from ..core.ids import new_id
from ..core.protocols import Embedder

__all__ = ["cosine", "embed_all", "resolve_ids"]


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
    """
    try:
        return list(await embedder.embed_batch(texts))
    except (AttributeError, NotImplementedError):
        return [await embedder.embed(t) for t in texts]


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
