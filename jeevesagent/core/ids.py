"""ID and hash helpers used across the harness.

ULIDs are preferred over UUIDs because they are time-sortable, which makes
journal scans, audit log queries, and episode timelines cheap.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ulid import ULID


def new_id(prefix: str = "") -> str:
    """Return a fresh ULID, optionally prefixed for readability.

    >>> new_id("ep").startswith("ep_")
    True
    """
    raw = str(ULID())
    return f"{prefix}_{raw}" if prefix else raw


def deterministic_hash(*parts: Any) -> str:
    """Stable hash of arbitrary JSON-serializable parts.

    Used as an idempotency key for journaled steps. The hash is stable
    across processes and Python versions because the input is canonicalised
    via ``json.dumps(..., sort_keys=True)``.
    """
    payload = json.dumps(parts, sort_keys=True, default=_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default(obj: Any) -> Any:
    # Pydantic models, datetimes, sets — make them representable.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=str)
    return repr(obj)
