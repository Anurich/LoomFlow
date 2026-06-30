"""Internal helpers for binary embedding storage.

float32 packing is the wire format used by both ``RedisMemory`` and
``SqliteFactStore`` (and any future blob-storage backend). One copy
here keeps the two backends from drifting apart.
"""

from __future__ import annotations

import math
import struct


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, range ``[-1, 1]``.

    Returns ``0.0`` for empty, mismatched-length, or zero-norm inputs
    (which would otherwise produce ``nan``). Shared by the native
    hybrid ``recall_scored`` implementations across the persistent
    memory backends so they don't each carry their own copy.
    """
    if not a or not b or len(a) != len(b):
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
