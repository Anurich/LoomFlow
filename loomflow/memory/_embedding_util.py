"""Internal helpers for binary embedding storage.

float32 packing is the wire format used by both ``RedisMemory`` and
``SqliteFactStore`` (and any future blob-storage backend). One copy
here keeps the two backends from drifting apart.
"""

from __future__ import annotations

import math
import struct
from typing import Any


class EmbedderMismatchError(ValueError):
    """Stored vectors and the configured embedder don't belong
    together (different model or different dimension). Comparing them
    would return confident nonsense, so this is raised instead."""


def embedder_identity(embedder: Any) -> tuple[str | None, int | None]:
    """``(name, dimensions)`` of an embedder; ``None`` for whatever a
    custom embedder doesn't declare."""
    name = getattr(embedder, "name", None)
    dims = getattr(embedder, "dimensions", None)
    return (
        name if isinstance(name, str) and name else None,
        dims if isinstance(dims, int) and not isinstance(dims, bool) else None,
    )


def embedder_mismatch_message(
    where: str,
    stored: tuple[str | None, int | None],
    current: tuple[str | None, int | None],
) -> str:
    """The one error text for "this store was written with another
    embedder" — what was found, and the ways out."""

    def _describe(identity: tuple[str | None, int | None]) -> str:
        name, dims = identity
        label = repr(name) if name else "an unnamed embedder"
        return f"{label} ({dims} dimensions)" if dims else label

    return (
        f"{where} was written with embedder {_describe(stored)} but is "
        f"being opened with {_describe(current)}. Vectors from different "
        "embedders are not comparable, so recall would silently return "
        "wrong results. Fix it by passing the original embedder "
        "(embedder=..., or LOOMFLOW_EMBEDDER / OPENAI_API_KEY as they "
        "were when the data was written), or start a fresh store for "
        "the new embedder and re-add the data."
    )


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, range ``[-1, 1]``.

    Returns ``0.0`` for empty or zero-norm inputs (which would
    otherwise produce ``nan``). Vectors of different lengths come from
    different embedders and have no similarity at all: that raises
    :class:`EmbedderMismatchError` rather than scoring ``0.0`` and
    quietly ranking everything as irrelevant. Shared by the native
    hybrid ``recall_scored`` implementations across the persistent
    memory backends so they don't each carry their own copy.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise EmbedderMismatchError(
            f"cannot compare a {len(a)}-dimensional vector with a "
            f"{len(b)}-dimensional one: the stored embeddings were "
            "produced by a different embedder than the one in use. "
            "Use the embedder the data was written with, or re-embed "
            "the stored data with the new one."
        )
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


def pack_float32(values: list[float]) -> bytes:
    """Pack a list of floats into a contiguous float32 byte string.

    Empty input returns ``b""`` rather than zero-length pack so the
    distinction between "no embedding" (``b""``) and "all-zero
    embedding" stays visible.
    """
    if not values:
        return b""
    return struct.pack(f"{len(values)}f", *values)


def unpack_float32(blob: bytes) -> list[float]:
    """Unpack a float32 byte string into a Python list."""
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def vector_to_list(value: Any) -> list[float] | None:
    """An embedding read back from a database as ``list[float]``:
    pgvector's ``Vector`` (``to_list``), a numpy array (``tolist``),
    pgvector's text form (``"[1,2,3]"``) or any sequence."""
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    for method in ("to_list", "tolist"):
        convert = getattr(value, method, None)
        if callable(convert):
            return [float(x) for x in convert()]
    if isinstance(value, str):
        text = value.strip().removeprefix("[").removesuffix("]")
        return [float(x) for x in text.split(",")] if text else []
    return [float(x) for x in value]
