"""Internal helpers for binary embedding storage.

float32 packing is the wire format used by both ``RedisMemory`` and
``SqliteFactStore`` (and any future blob-storage backend). One copy
here keeps the two backends from drifting apart.
"""

from __future__ import annotations

import struct


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
