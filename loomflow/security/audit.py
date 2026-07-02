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

The tamper-evidence is only as strong as the key: with the default
``secret=""`` the HMAC is computed over a well-known (empty) key,
so anyone who can edit the log can also recompute a valid
signature — :func:`verify_signature` then proves nothing. Both
backends emit a one-shot :class:`UserWarning` when constructed
without a real secret.

The log is conceptually monotonic: ``seq`` is per-log and never
re-used. :class:`FileAuditLog` recovers the highest seq from the file
on startup so multiple processes can append in turn.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import warnings
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Union, runtime_checkable

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


# One-shot flag: warn about empty signing secrets at most once per
# process, not once per log construction (tests and dev tooling
# build many unsigned logs deliberately).
_EMPTY_SECRET_WARNED = False


def _warn_if_empty_secret(secret: str, cls_name: str) -> None:
    """Emit a one-shot warning when a log is built without a real
    signing key — its HMACs are then forgeable by anyone who can
    write the log, so the advertised tamper-evidence is void."""
    global _EMPTY_SECRET_WARNED
    if secret or _EMPTY_SECRET_WARNED:
        return
    _EMPTY_SECRET_WARNED = True
    warnings.warn(
        f"{cls_name} constructed with an empty signing secret: entry "
        "signatures are HMACs over a well-known key, so anyone who can "
        "edit the log can recompute them and verify_signature() "
        "provides NO tamper-evidence. Pass secret=<random key> (e.g. "
        "secrets.token_hex(32)) for real integrity guarantees.",
        UserWarning,
        stacklevel=3,
    )


def verify_signature(entry: AuditEntry, secret: str) -> bool:
    """Recompute the HMAC and compare against the stored signature.

    Only meaningful when the log was built with a real (non-empty,
    unguessable) ``secret``: with the default ``secret=""`` the
    HMAC key is public knowledge and a forger can produce entries
    that verify, so a ``True`` return proves nothing.
    """
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
        _warn_if_empty_secret(secret, "InMemoryAuditLog")
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
        _warn_if_empty_secret(secret, "FileAuditLog")
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


# ---------------------------------------------------------------------------
# Full-transcript wrapper
# ---------------------------------------------------------------------------


def wants_full_transcripts(log: AuditLog | None) -> bool:
    """True when ``log`` opted into full-content capture by setting
    ``full_transcripts = True``.

    Framework call sites (agent loop, react tool dispatch) use this
    to decide whether to send full prompts / outputs / tool result
    bodies into the audit payload or stick to the safe-by-default
    summary fields. ``None`` and any log without the attribute return
    ``False`` — the default stays compliance-friendly.
    """
    return bool(getattr(log, "full_transcripts", False))


class FullTranscriptAuditLog:
    """Wraps any :class:`AuditLog` to capture full prompts, outputs,
    and tool-result bodies — not just the summary fields.

    The default audit log truncates prompts to 500 chars, omits the
    model's final output, and stores only ``ok`` / ``denied`` /
    ``error`` / ``reason`` for tool results. That's the right
    default for compliance regimes that prohibit logging customer
    PII verbatim.

    For **debugging**, **post-incident replay**, or **internal
    investigations** ("what did my agent actually say to that
    user?") the framework checks
    :func:`wants_full_transcripts` on the wired audit log and emits
    full content into the payload when this wrapper is present.

    Usage::

        from loomflow.security import (
            FullTranscriptAuditLog,
            FileAuditLog,
        )

        agent = Agent(
            "...",
            audit_log=FullTranscriptAuditLog(
                FileAuditLog("./audit.jsonl", secret="...")
            ),
        )

    Forwards every call to the wrapped log unchanged — same seq
    numbers, same HMAC signatures, same ``query`` semantics. Only
    difference: the agent / architecture layer fills the
    ``payload`` dict with full content before calling ``append``.

    The opt-in lives on the wrapper, not on a constructor flag, so
    threat-modelling is explicit: ``isinstance(log,
    FullTranscriptAuditLog)`` or ``log.full_transcripts is True``
    is the audit reviewer's signal that PII may be in the log.
    """

    # Duck-typed marker checked by :func:`wants_full_transcripts`.
    # Keeping it as a class attribute (not a constructor arg) means
    # the type itself is the contract — wrapping = opt-in.
    full_transcripts: bool = True

    def __init__(self, inner: AuditLog) -> None:
        self._inner = inner

    @property
    def inner(self) -> AuditLog:
        """The wrapped audit log. Useful for tests + introspection."""
        return self._inner

    async def append(
        self,
        *,
        session_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
        user_id: str | None = None,
    ) -> AuditEntry:
        return await self._inner.append(
            session_id=session_id,
            actor=actor,
            action=action,
            payload=payload,
            user_id=user_id,
        )

    async def query(
        self,
        *,
        session_id: str | None = None,
        action: str | None = None,
        user_id: str | None = None,
    ) -> list[AuditEntry]:
        return await self._inner.query(
            session_id=session_id,
            action=action,
            user_id=user_id,
        )

    async def all_entries(self) -> list[AuditEntry]:
        # Forward when the inner log exposes the list helper —
        # InMemoryAuditLog and FileAuditLog both do. Custom impls
        # without it just won't have the helper either.
        fn = getattr(self._inner, "all_entries", None)
        if fn is None:
            return await self._inner.query()
        result: list[AuditEntry] = await fn()
        return result


# ---------------------------------------------------------------------------
# Resolver — turn the ``audit_log=`` constructor arg into an AuditLog
# ---------------------------------------------------------------------------


# Public type alias — the surface accepted by ``Agent(audit_log=...)`` and
# ``Workflow(audit_log=...)``.
AuditLogSpec = Union[
    "AuditLog",
    str,
    Path,
    dict[str, Any],
    None,
]


def resolve_audit_log(spec: AuditLogSpec) -> AuditLog | None:
    """Normalise the ``audit_log=`` constructor argument.

    Accepted forms:

    * ``None`` — no audit log; pass-through.
    * ``str`` / :class:`pathlib.Path` — sugar for
      :class:`FileAuditLog` at that path. Use this when you just
      want JSONL on disk with no signing key and summary-level
      capture.
    * Any :class:`AuditLog` instance — used as-is. Lets callers
      hand-construct a :class:`FileAuditLog` with a signing key,
      wrap one in :class:`FullTranscriptAuditLog`, or plug in a
      custom backend.
    * ``dict`` — config-friendly form. Recognised keys:

        * ``"name"`` (``str`` / ``Path``, optional) — file path.
          When omitted, an :class:`InMemoryAuditLog` is built.
        * ``"scope_full"`` (``bool``, default ``False``) — when
          ``True``, the resulting log is wrapped with
          :class:`FullTranscriptAuditLog` so prompts, model
          outputs, and tool result bodies land in the audit
          payload verbatim instead of being summarised.
        * ``"secret"`` (``str``, optional) — HMAC signing key
          passed through to the underlying log.

      Example: ``audit_log={"name": "audit.log", "scope_full": True}``
      gets you a file-backed log that captures everything.

    Anything else raises :class:`TypeError` with the list of valid
    options so wiring mistakes surface at construction time, not
    deep in the runtime.
    """
    if spec is None:
        return None

    if isinstance(spec, (str, Path)):
        return FileAuditLog(spec)

    if isinstance(spec, dict):
        return _resolve_from_dict(spec)

    # Real AuditLog instance — runtime-checkable Protocol covers
    # the framework's two backends + ``FullTranscriptAuditLog`` +
    # any custom impl with the right ``append`` + ``query`` shape.
    if isinstance(spec, AuditLog):
        return spec

    raise TypeError(
        f"audit_log= must be an AuditLog instance, a path (str / "
        f"pathlib.Path), a dict, or None; got "
        f"{type(spec).__name__}: {spec!r}.\n"
        f"Valid options:\n"
        f"  • InMemoryAuditLog() — keep entries in memory\n"
        f"  • FileAuditLog('run.log') — JSONL on disk\n"
        f"  • 'run.log' or Path('run.log') — sugar for FileAuditLog\n"
        f"  • {{'name': 'run.log', 'scope_full': True}} — config-style "
        f"with optional full-transcript capture\n"
        f"  • None — disable audit logging"
    )


def _resolve_from_dict(spec: dict[str, Any]) -> AuditLog:
    """Build an audit log from the config-dict form."""
    allowed = {"name", "scope_full", "secret"}
    extras = set(spec) - allowed
    if extras:
        raise TypeError(
            f"audit_log= dict has unknown key(s): "
            f"{sorted(extras)}. Allowed keys: {sorted(allowed)}."
        )

    name = spec.get("name")
    secret = spec.get("secret", "")
    scope_full = bool(spec.get("scope_full", False))

    inner: AuditLog
    if name is None:
        inner = InMemoryAuditLog(secret=secret)
    else:
        if not isinstance(name, (str, Path)):
            raise TypeError(
                f"audit_log['name'] must be a str or pathlib.Path; "
                f"got {type(name).__name__}: {name!r}."
            )
        inner = FileAuditLog(name, secret=secret)

    if scope_full:
        return FullTranscriptAuditLog(inner)
    return inner
