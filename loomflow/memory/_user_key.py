"""Shared on-wire encoding for the anonymous ``user_id`` bucket.

Python code models "no user" as ``user_id=None``, but most storage
backends can't carry ``None``/NULL in the places we need it (Postgres
primary keys, Chroma metadata values, Redis hash fields). Every
backend therefore needs a non-NULL wire representation for the
anonymous bucket.

Earlier versions used the empty string ``""`` — which silently
collides with a caller passing ``""`` as a *real* user_id. The
Postgres backend replaced it with a reserved sentinel; this module
extracts that sentinel so **every** backend encodes the anonymous
bucket the same way.

Backward compatibility: data written by older versions stores the
anonymous bucket as ``""``. :func:`decode_legacy_user_id` (used by
the Chroma / Redis backends, which historically wrote ``""``)
accepts **both** forms and maps them to ``None``. New writes always
use the sentinel. Consequence: on those backends an empty-string
"real" user_id is indistinguishable from the legacy anonymous bucket
on read — don't use ``""`` as a user_id.

:func:`decode_user_id` is the strict inverse (only the sentinel maps
back to ``None``); Postgres uses it because its schema migration
rewrites legacy ``""`` rows to the sentinel, freeing ``""`` up as a
real value there.
"""

from __future__ import annotations

ANON_USER_ID = "__jeeves_anon_user__"
"""Reserved wire value for the anonymous (``user_id=None``) bucket.

The double-underscore-jeeves prefix is a hard rule callers must not
violate; :func:`encode_user_id` raises if they try.
"""

_LEGACY_ANON_USER_ID = ""
"""Pre-sentinel wire encoding of the anonymous bucket (still accepted
on read by :func:`decode_legacy_user_id` for old stored data)."""


def encode_user_id(user_id: str | None) -> str:
    """Map a Python ``user_id`` (``None`` for anonymous) to its
    on-wire representation.

    Rejects callers who try to use the sentinel value as a real
    user_id — that would let one user silently impersonate the
    anonymous bucket.
    """
    if user_id == ANON_USER_ID:
        raise ValueError(
            f"user_id {ANON_USER_ID!r} is reserved by Loom for "
            "the anonymous bucket; choose a different identifier."
        )
    return user_id if user_id is not None else ANON_USER_ID


def decode_user_id(wire_value: str) -> str | None:
    """Strict inverse of :func:`encode_user_id`: only the sentinel
    maps back to ``None``. Used by backends (Postgres) whose schema
    migration rewrote legacy ``""`` rows to the sentinel."""
    return None if wire_value == ANON_USER_ID else wire_value


def decode_legacy_user_id(wire_value: str) -> str | None:
    """Lenient inverse for backends that historically encoded the
    anonymous bucket as ``""`` (Chroma, Redis). Accepts both the
    legacy empty string and the new sentinel and maps either to
    ``None`` so pre-existing stored data keeps decoding correctly."""
    if wire_value in (ANON_USER_ID, _LEGACY_ANON_USER_ID):
        return None
    return wire_value


def user_id_wire_values(user_id: str | None) -> list[str]:
    """Every wire encoding that should MATCH ``user_id`` on the read
    side. For the anonymous bucket that's both the sentinel (new
    writes) and the legacy empty string (old rows); for a named user
    it's just the encoded id. Backends with native filters (e.g.
    Chroma ``where``) use this to build backward-compatible clauses.
    """
    if user_id is None:
        return [ANON_USER_ID, _LEGACY_ANON_USER_ID]
    return [encode_user_id(user_id)]


def user_id_where_clause(user_id: str | None) -> dict[str, object]:
    """Mongo-style equality/``$in`` clause over the ``user_id``
    metadata field, matching every wire encoding from
    :func:`user_id_wire_values`. Used by the Chroma backends' native
    ``where`` filters so legacy empty-string anonymous rows keep
    matching after the sentinel migration."""
    values = user_id_wire_values(user_id)
    if len(values) == 1:
        return {"user_id": values[0]}
    return {"user_id": {"$in": values}}


__all__ = [
    "ANON_USER_ID",
    "decode_legacy_user_id",
    "decode_user_id",
    "encode_user_id",
    "user_id_where_clause",
    "user_id_wire_values",
]
