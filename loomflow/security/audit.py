"""Append-only audit log.

Every meaningful event in the loop — run start/finish, tool dispatch,
tool result, permission decision — gets a signed entry on the audit
log. Two backends ship:

* :class:`InMemoryAuditLog` — list-backed, fast, used in tests and dev.
* :class:`FileAuditLog` — JSONL append on disk, durable across
  process restarts.

Both compute an HMAC-SHA256 ``signature`` over a canonicalised
representation of the entry's content fields, keyed by a per-log
``secret``. The signature lets compliance tooling detect tampering;
:func:`verify_signature` recomputes it and compares.

The log is conceptually monotonic: ``seq`` is per-log and never
re-used. :class:`FileAuditLog` recovers the highest seq from the file
on startup so multiple processes can append in turn.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio

from ..core.types import AuditEntry


@runtime_checkable
class AuditLog(Protocol):
    """The append-only signed log surface.

    ``user_id`` (M9) is a top-level field on every entry, populated
    from the live :class:`~loomflow.RunContext`. Backends MUST
    accept the kwarg on ``append`` and the ``query`` filter so
    multi-tenant audit queries work without payload-digging.
    """

    async def append(
        self,
        *,
        session_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
        user_id: str | None = None,
    ) -> AuditEntry: ...

    async def query(
        self,
        *,
        session_id: str | None = None,
        action: str | None = None,
        user_id: str | None = None,
    ) -> list[AuditEntry]: ...


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def _canonical_payload(
    *,
    seq: int,
    timestamp: datetime,
    session_id: str,
    actor: str,
    action: str,
    payload: dict[str, Any],
    user_id: str | None = None,
) -> bytes:
    """Stable byte representation used for HMAC signing.

    Sorted-keys JSON keeps the signature stable across processes and
    Python releases. ``user_id`` is included so a tampered entry
    that swaps the user_id alongside the payload won't verify.
    """
    blob = {
        "seq": seq,
        "timestamp": timestamp.isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "actor": actor,
        "action": action,
        "payload": payload,
    }
    return json.dumps(blob, sort_keys=True, default=str).encode("utf-8")


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(
        secret.encode("utf-8") or b"",
        body,
        hashlib.sha256,
    ).hexdigest()


def verify_signature(entry: AuditEntry, secret: str) -> bool:
    """Recompute the HMAC and compare against the stored signature."""
    expected = _sign(
        secret,
        _canonical_payload(
            seq=entry.seq,
            timestamp=entry.timestamp,
            session_id=entry.session_id,
            user_id=entry.user_id,
            actor=entry.actor,
            action=entry.action,
            payload=entry.payload,
        ),
    )
    return hmac.compare_digest(expected, entry.signature)


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryAuditLog:
    """List-backed signed audit log."""

    def __init__(self, *, secret: str = "") -> None:
        self._secret = secret
        self._entries: list[AuditEntry] = []
        self._seq = 0
        self._lock = anyio.Lock()

    async def append(
        self,
        *,
        session_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
        user_id: str | None = None,
    ) -> AuditEntry:
        async with self._lock:
            self._seq += 1
            timestamp = datetime.now(UTC)
            body = _canonical_payload(
                seq=self._seq,
                timestamp=timestamp,
                session_id=session_id,
                user_id=user_id,
                actor=actor,
                action=action,
                payload=payload,
            )
            entry = AuditEntry(
                seq=self._seq,
                timestamp=timestamp,
                session_id=session_id,
                user_id=user_id,
                actor=actor,
                action=action,
                payload=payload,
                signature=_sign(self._secret, body),
            )
            self._entries.append(entry)
            return entry

    async def query(
        self,
        *,
        session_id: str | None = None,
        action: str | None = None,
        user_id: str | None = None,
    ) -> list[AuditEntry]:
        async with self._lock:
            entries = list(self._entries)
        return _filter_entries(
            entries,
            session_id=session_id,
            action=action,
            user_id=user_id,
        )

    async def all_entries(self) -> list[AuditEntry]:
        async with self._lock:
            return list(self._entries)


# ---------------------------------------------------------------------------
# JSONL file backend
# ---------------------------------------------------------------------------


class FileAuditLog:
    """JSONL append-only audit log with HMAC signatures.

    On construction we read any pre-existing entries to recover the
    highest seq, so a process restart picks up where the last one left
    off.
    """

    def __init__(self, path: str | Path, *, secret: str = "") -> None:
        self._path = Path(path)
        self._secret = secret
        self._seq = 0
        self._lock = anyio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._seq = self._scan_max_seq()

    @property
    def path(self) -> Path:
        return self._path

    def _scan_max_seq(self) -> int:
        max_seq = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq = obj.get("seq")
                if isinstance(seq, int) and seq > max_seq:
                    max_seq = seq
        return max_seq

    async def append(
        self,
        *,
        session_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
        user_id: str | None = None,
    ) -> AuditEntry:
        async with self._lock:
            self._seq += 1
            timestamp = datetime.now(UTC)
            body = _canonical_payload(
                seq=self._seq,
                timestamp=timestamp,
                session_id=session_id,
                user_id=user_id,
                actor=actor,
                action=action,
                payload=payload,
            )
            entry = AuditEntry(
                seq=self._seq,
                timestamp=timestamp,
                session_id=session_id,
                user_id=user_id,
                actor=actor,
                action=action,
                payload=payload,
                signature=_sign(self._secret, body),
            )
            await anyio.to_thread.run_sync(self._write_line, entry)
            return entry

    def _write_line(self, entry: AuditEntry) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.model_dump(mode="json")))
            fh.write("\n")

    async def query(
        self,
        *,
        session_id: str | None = None,
        action: str | None = None,
        user_id: str | None = None,
    ) -> list[AuditEntry]:
        entries = await anyio.to_thread.run_sync(self._read_entries)
        return _filter_entries(
            entries,
            session_id=session_id,
            action=action,
            user_id=user_id,
        )

    def _read_entries(self) -> list[AuditEntry]:
        if not self._path.exists():
            return []
        out: list[AuditEntry] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    out.append(AuditEntry.model_validate(obj))
                except Exception:  # noqa: BLE001 — skip corrupt entries
                    continue
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_entries(
    entries: list[AuditEntry],
    *,
    session_id: str | None,
    action: str | None,
    user_id: str | None = None,
) -> list[AuditEntry]:
    if session_id is not None:
        entries = [e for e in entries if e.session_id == session_id]
    if action is not None:
        entries = [e for e in entries if e.action == action]
    if user_id is not None:
        entries = [e for e in entries if e.user_id == user_id]
    return entries


async def stream_entries(log: AuditLog) -> AsyncIterator[AuditEntry]:
    """Yield every entry currently in ``log`` in seq order.

    A polling helper for compliance tooling. Doesn't tail — that comes
    later when we add an on-write notification stream.
    """
    entries = await log.query()
    entries.sort(key=lambda e: e.seq)
    for entry in entries:
        yield entry
