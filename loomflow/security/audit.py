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

Each entry also carries ``prev_hash`` — the previous entry's
signature — inside its own signed content, so the entries form a
hash chain: :func:`verify_chain` (and ``verify()`` on both backends)
detects an edited, deleted, inserted or reordered entry, not just an
edited one. Cutting entries off the *end* leaves a valid chain; to
detect that, record the ``head`` a verification reports somewhere the
log's writer can't change, and pass it back as ``expected_head``.

The tamper-evidence is only as strong as the key: with the default
``secret=""`` the HMAC is computed over a well-known (empty) key,
so anyone who can edit the log can also recompute a valid
signature — :func:`verify_signature` then proves nothing. Both
backends emit a one-shot :class:`UserWarning` when constructed
without a real secret.

``seq`` is per-log, starts at 1 and never repeats. :class:`FileAuditLog`
takes an exclusive file lock for each append and continues from the
file's last entry, so several processes (or several log objects) can
append to one file without duplicating a seq or forking the chain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import warnings
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Protocol, Union, runtime_checkable

import anyio
from pydantic_core import to_json

from ..core.types import AuditEntry

_LOG = logging.getLogger("loomflow.security.audit")

#: ``prev_hash`` of a log's first entry.
GENESIS_HASH = "0" * 64


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
    prev_hash: str = "",
) -> bytes:
    """Stable byte representation used for HMAC signing.

    Sorted-keys JSON keeps the signature stable across processes and
    Python releases. ``user_id`` is included so a tampered entry
    that swaps the user_id alongside the payload won't verify.
    ``prev_hash`` is left out when empty so entries written before
    the hash chain still verify.
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
    if prev_hash:
        blob["prev_hash"] = prev_hash
    return json.dumps(blob, sort_keys=True, default=str).encode("utf-8")


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(
        secret.encode("utf-8") or b"",
        body,
        hashlib.sha256,
    ).hexdigest()


def _json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """``payload`` as the JSON it is stored as (datetimes as ISO
    strings, sets and tuples as lists, other objects via ``str``).

    The entry is signed over, and holds, this form — so an entry read
    back from storage verifies exactly like the one ``append``
    returned."""
    out: dict[str, Any] = json.loads(to_json(payload, fallback=str))
    return out


def _new_entry(
    secret: str,
    *,
    seq: int,
    prev_hash: str,
    session_id: str,
    actor: str,
    action: str,
    payload: dict[str, Any],
    user_id: str | None,
) -> AuditEntry:
    timestamp = datetime.now(UTC)
    payload = _json_payload(payload)
    body = _canonical_payload(
        seq=seq,
        timestamp=timestamp,
        session_id=session_id,
        user_id=user_id,
        actor=actor,
        action=action,
        payload=payload,
        prev_hash=prev_hash,
    )
    return AuditEntry(
        seq=seq,
        timestamp=timestamp,
        session_id=session_id,
        user_id=user_id,
        actor=actor,
        action=action,
        payload=payload,
        signature=_sign(secret, body),
        prev_hash=prev_hash,
    )


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
    that verify, so a ``True`` return proves nothing. Checks one
    entry; :func:`verify_chain` also checks that none were removed,
    added or reordered.
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
            prev_hash=entry.prev_hash,
        ),
    )
    return hmac.compare_digest(expected, entry.signature)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditProblem:
    """One thing wrong with a log. ``line`` is the 1-based line in a
    :class:`FileAuditLog` file (``None`` elsewhere)."""

    reason: str
    seq: int | None = None
    line: int | None = None


@dataclass(frozen=True)
class AuditVerification:
    """The result of checking a whole log.

    ``head`` is the last entry's signature: store it outside the log
    and pass it back as ``expected_head`` to detect entries cut off
    the end later.
    """

    ok: bool
    entries: int
    head: str | None
    problems: tuple[AuditProblem, ...]


def verify_chain(
    entries: Iterable[AuditEntry],
    secret: str,
    *,
    expected_head: str | None = None,
) -> AuditVerification:
    """Check a whole log's entries, in stored order: every signature,
    a seq that starts at 1 and goes up by one, and each ``prev_hash``
    linking to the entry before it. ``expected_head`` (a ``head``
    recorded earlier) must still be in the log.

    Entries written before the hash chain (``prev_hash == ""``) are
    checked for their signature and seq only.
    """
    return _verify([(None, e) for e in entries], secret, expected_head, [])


def _verify(
    rows: list[tuple[int | None, AuditEntry]],
    secret: str,
    expected_head: str | None,
    problems: list[AuditProblem],
) -> AuditVerification:
    prev: AuditEntry | None = None
    signatures: set[str] = set()
    for line, entry in rows:
        found: list[str] = []
        if not verify_signature(entry, secret):
            found.append("signature does not match the entry's content")
        expected_seq = 1 if prev is None else prev.seq + 1
        if prev is None and entry.prev_hash not in ("", GENESIS_HASH):
            found.append("entries before this one are missing")
        elif entry.seq != expected_seq:
            found.append(f"seq {entry.seq} where {expected_seq} was expected")
        if prev is not None:
            if entry.prev_hash and entry.prev_hash != prev.signature:
                found.append("prev_hash does not match the entry before it")
            elif not entry.prev_hash and prev.prev_hash:
                found.append("unchained entry after chained ones")
        problems.extend(AuditProblem(r, seq=entry.seq, line=line) for r in found)
        signatures.add(entry.signature)
        prev = entry
    if expected_head is not None and expected_head not in signatures:
        problems.append(
            AuditProblem("the expected head is not in the log (entries were removed)")
        )
    return AuditVerification(
        ok=not problems,
        entries=len(rows),
        head=prev.signature if prev is not None else None,
        problems=tuple(problems),
    )


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
            entry = _new_entry(
                self._secret,
                seq=self._seq + 1,
                prev_hash=(
                    self._entries[-1].signature if self._entries else GENESIS_HASH
                ),
                session_id=session_id,
                user_id=user_id,
                actor=actor,
                action=action,
                payload=payload,
            )
            self._seq = entry.seq
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

    async def verify(self, *, expected_head: str | None = None) -> AuditVerification:
        """Check the whole log with this log's secret (see
        :func:`verify_chain`)."""
        return verify_chain(
            await self.all_entries(), self._secret, expected_head=expected_head
        )


# ---------------------------------------------------------------------------
# JSONL file backend
# ---------------------------------------------------------------------------


@contextmanager
def _file_lock(fh: IO[bytes], *, shared: bool = False) -> Iterator[None]:
    """Hold an advisory lock on ``fh`` (``flock`` on POSIX, a byte-range
    lock on Windows) so appends from several processes serialise."""
    if sys.platform == "win32":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _parse_entry(raw: bytes) -> AuditEntry | None:
    if not raw.strip():
        return None
    try:
        return AuditEntry.model_validate_json(raw)
    except ValueError:
        return None


@dataclass
class _Tail:
    size: int  # file size this state was read at
    seq: int
    head: str
    ends_with_newline: bool


class FileAuditLog:
    """JSONL append-only audit log with HMAC signatures.

    Each append takes an exclusive lock on the file and continues the
    seq and the hash chain from the file's last entry, so a process
    restart, or another process appending to the same file, picks up
    where the last writer left off. The file is first read on the
    first ``append`` (in a worker thread, off the event loop), not in
    ``__init__``, and only its tail is read — again only when another
    writer has appended since.

    ``query`` skips (and logs) lines that are not valid entries;
    ``verify`` reports them with their line numbers.
    """

    def __init__(self, path: str | Path, *, secret: str = "") -> None:
        _warn_if_empty_secret(secret, "FileAuditLog")
        self._path = Path(path)
        self._secret = secret
        self._tail: _Tail | None = None
        self._lock = anyio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def _recover_tail(self, fh: IO[bytes], size: int) -> _Tail:
        """Seq and signature of the file's last valid entry, read
        backwards from the end."""
        if size == 0:
            return _Tail(0, 0, GENESIS_HASH, True)
        fh.seek(size - 1)
        ends_with_newline = fh.read(1) == b"\n"
        block = 1 << 16
        while True:
            start = max(0, size - block)
            fh.seek(start)
            lines = fh.read(size - start).split(b"\n")
            if start > 0:
                lines = lines[1:]  # the first piece may be a partial line
            for raw in reversed(lines):
                entry = _parse_entry(raw)
                if entry is not None:
                    return _Tail(size, entry.seq, entry.signature, ends_with_newline)
            if start == 0:
                return _Tail(size, 0, GENESIS_HASH, ends_with_newline)
            block *= 4

    async def append(
        self,
        *,
        session_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
        user_id: str | None = None,
    ) -> AuditEntry:
        def write() -> AuditEntry:
            with self._path.open("a+b") as fh, _file_lock(fh):
                size = os.fstat(fh.fileno()).st_size
                tail = self._tail
                if tail is None or tail.size != size:
                    tail = self._recover_tail(fh, size)
                entry = _new_entry(
                    self._secret,
                    seq=tail.seq + 1,
                    prev_hash=tail.head,
                    session_id=session_id,
                    user_id=user_id,
                    actor=actor,
                    action=action,
                    payload=payload,
                )
                line = json.dumps(entry.model_dump(mode="json")).encode("utf-8") + b"\n"
                if not tail.ends_with_newline:
                    line = b"\n" + line  # don't glue onto a torn last line
                fh.write(line)
                fh.flush()
                self._tail = _Tail(size + len(line), entry.seq, entry.signature, True)
                return entry

        async with self._lock:
            return await anyio.to_thread.run_sync(write)

    def _read_lines(self) -> list[bytes]:
        if not self._path.exists():
            return []
        with self._path.open("rb") as fh, _file_lock(fh, shared=True):
            return fh.read().split(b"\n")

    def _read_entries(self) -> list[AuditEntry]:
        out: list[AuditEntry] = []
        bad: list[int] = []
        for number, raw in enumerate(self._read_lines(), start=1):
            entry = _parse_entry(raw)
            if entry is not None:
                out.append(entry)
            elif raw.strip():
                bad.append(number)
        if bad:
            _LOG.warning(
                "audit log %s: skipped %d unreadable line(s) (lines %s); "
                "run verify() for details",
                self._path,
                len(bad),
                ", ".join(map(str, bad[:10])) + (", ..." if len(bad) > 10 else ""),
            )
        return out

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

    async def all_entries(self) -> list[AuditEntry]:
        return await self.query()

    async def verify(self, *, expected_head: str | None = None) -> AuditVerification:
        """Check the whole file with this log's secret: unreadable
        lines, signatures, seq order and the hash chain (see
        :func:`verify_chain`)."""

        def check() -> AuditVerification:
            rows: list[tuple[int | None, AuditEntry]] = []
            problems: list[AuditProblem] = []
            for number, raw in enumerate(self._read_lines(), start=1):
                entry = _parse_entry(raw)
                if entry is not None:
                    rows.append((number, entry))
                elif raw.strip():
                    problems.append(AuditProblem("not a valid entry", line=number))
            return _verify(rows, self._secret, expected_head, problems)

        return await anyio.to_thread.run_sync(check)


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

    async def verify(self, *, expected_head: str | None = None) -> AuditVerification:
        """Forward to the wrapped log's ``verify``."""
        fn = getattr(self._inner, "verify", None)
        if fn is None:
            raise TypeError(f"{type(self._inner).__name__} has no verify()")
        result: AuditVerification = await fn(expected_head=expected_head)
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
