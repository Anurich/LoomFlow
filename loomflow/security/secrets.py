"""Concrete :class:`~loomflow.core.protocols.Secrets`
implementations.

Two ship in the framework, neither requiring extra dependencies:

* :class:`EnvSecrets` — reads from ``os.environ``. Default for
  :class:`~loomflow.Agent` so today's behaviour is preserved
  (API keys come from environment variables) without callers
  having to wire anything.
* :class:`DictSecrets` — explicit in-memory dict, useful in tests
  and for callers who load secrets from a config file or a
  vault-fetch-once-at-startup script.

Production users running on AWS / GCP / Vault should write a
custom :class:`Secrets` adapter that calls their secret manager
inside ``resolve()`` and caches into a local dict for
``lookup_sync()``. The framework only requires
``lookup_sync()`` to return synchronously (it's called from
inside Agent / model-adapter constructors); ``resolve()`` /
``store()`` can do whatever async work you need.

Redaction lives here too. :func:`redact_secrets` scrubs (a) every
secret VALUE a :class:`Secrets` backend in this process has stored
or handed out, and (b) common key shapes (``sk-...``, ``Bearer
...``, AWS / GitHub / Slack / Google keys, ``FOO_API_KEY=...``).
The shipped audit logs (:mod:`loomflow.security.audit`) run every
payload through it before the entry is signed and written, so
callers who don't wire a vault still get safe-by-default audit log
behaviour.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Final

__all__ = [
    "DictSecrets",
    "EnvSecrets",
    "redact_payload",
    "redact_secrets",
]

REDACTED: Final = "[REDACTED]"

# Secret VALUES known to this process: everything a Secrets backend
# has stored or returned. Pattern matching can't catch a key with no
# recognisable shape (a database password, a webhook secret); these
# are scrubbed verbatim. Values shorter than ``_MIN_KNOWN_LEN`` are
# not tracked — replacing every "dev" or "1234" in a log would make
# it unreadable without protecting anything.
_MIN_KNOWN_LEN: Final = 8
_known_values: set[str] = set()
_known_lock = threading.Lock()
_known_pattern: re.Pattern[str] | None = None


def _remember(value: str | None) -> None:
    """Record a secret value so redaction scrubs it verbatim."""
    global _known_pattern  # noqa: PLW0603
    if not isinstance(value, str) or len(value) < _MIN_KNOWN_LEN:
        return
    with _known_lock:
        if value not in _known_values:
            _known_values.add(value)
            _known_pattern = None  # rebuilt lazily


def _known_values_pattern() -> re.Pattern[str] | None:
    global _known_pattern  # noqa: PLW0603
    with _known_lock:
        if _known_pattern is None and _known_values:
            # Longest first, so a value that contains another is
            # replaced whole.
            ordered = sorted(_known_values, key=len, reverse=True)
            _known_pattern = re.compile("|".join(re.escape(v) for v in ordered))
        return _known_pattern


# Patterns we redact by default, as (pattern, replacement) pairs.
# Conservative — anchored to well-known key prefixes / shapes so
# ordinary prose doesn't match, but false-positives are preferable
# to leaking a real key into an audit log. Production users
# override this in subclasses.
_REDACTION_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED]"),  # OpenAI-style
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"), "[REDACTED]"),  # Anthropic-style
    # AWS access key ids — long-lived (AKIA) and session / temporary
    # (ASIA) credentials share the 16-char uppercase suffix shape.
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), "[REDACTED]"),  # GitHub PAT
    # Slack tokens: xoxb- (bot), xoxa-, xoxp- (user), xoxr-, xoxs-.
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "[REDACTED]"),
    # Google API keys — fixed "AIza" prefix + 35 more chars.
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED]"),
    # Authorization headers: "Bearer <token>". Keeps the scheme so
    # the redacted log still shows WHAT was sent, not the credential.
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
        "Bearer [REDACTED]",
    ),
    # Generic env-style assignments: FOO_API_KEY=..., BAR_TOKEN=....
    # Anchored on the =-assignment shape + a minimum value length so
    # prose ("the token is short-lived") never matches; the variable
    # NAME is kept so logs stay debuggable.
    (
        re.compile(
            r"\b([A-Za-z][A-Za-z0-9_]*(?:_API_KEY|_TOKEN))"
            r"(\s*=\s*)(['\"]?)[^\s'\"]{8,}\3"
        ),
        r"\1\2[REDACTED]",
    ),
)


def _apply_redaction(text: str) -> str:
    out = text
    known = _known_values_pattern()
    if known is not None:
        out = known.sub(REDACTED, out)
    for pat, replacement in _REDACTION_PATTERNS:
        out = pat.sub(replacement, out)
    return out


def redact_secrets(text: str) -> str:
    """Scrub secrets from ``text``: every value a :class:`Secrets`
    backend in this process has stored or handed out, plus common
    credential shapes (``sk-...``, ``Bearer <token>``, AWS / GitHub /
    Slack / Google keys, ``NAME_API_KEY=...`` assignments)."""
    return _apply_redaction(text)


def redact_payload(value: Any, redact: Any = redact_secrets) -> Any:
    """Apply ``redact`` to every string inside a JSON-like value
    (dict values AND keys, list items, nested to any depth); other
    scalars pass through. Returns a new structure.

    This is what the shipped audit logs run on a payload before
    signing it — call it from a custom :class:`AuditLog` to get the
    same behaviour.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {
            (redact(k) if isinstance(k, str) else k): redact_payload(v, redact)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_payload(v, redact) for v in value]
    return value


class EnvSecrets:
    """Reads secrets from ``os.environ``.

    The default :class:`Secrets` impl wired by :class:`Agent` when
    the caller doesn't pass an explicit one. Behaviour matches the
    pre-M10 framework: API keys are looked up as the corresponding
    environment variable name (``OPENAI_API_KEY``,
    ``ANTHROPIC_API_KEY``, etc.).
    """

    async def resolve(self, ref: str) -> str:
        value = os.environ.get(ref)
        if value is None:
            raise KeyError(f"environment variable {ref!r} is not set")
        _remember(value)
        return value

    async def store(self, ref: str, value: str) -> None:
        # We don't write to ``os.environ`` from a Secrets impl —
        # mutating the environment of a running process is rude
        # to other code running inside it. Use :class:`DictSecrets`
        # for an in-process writable store, or write a custom
        # impl that hits your real secret manager.
        raise NotImplementedError(
            "EnvSecrets is read-only; use DictSecrets or a custom "
            "Secrets backend for write access."
        )

    def redact(self, text: str) -> str:
        return _apply_redaction(text)

    def lookup_sync(self, ref: str) -> str | None:
        value = os.environ.get(ref)
        _remember(value)
        return value


class DictSecrets:
    """In-process :class:`Secrets` backed by an explicit dict.

    Useful in tests and for callers that fetch secrets once at
    startup (from a config file, a one-shot Vault read, etc.) and
    want to make them available to the framework without leaking
    them into ``os.environ``.

    Mutable: ``store()`` updates the in-process map. Not durable
    across process restarts.
    """

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._values: dict[str, str] = dict(initial or {})
        for value in self._values.values():
            _remember(value)

    async def resolve(self, ref: str) -> str:
        try:
            return self._values[ref]
        except KeyError as exc:
            raise KeyError(f"secret {ref!r} not present") from exc

    async def store(self, ref: str, value: str) -> None:
        self._values[ref] = value
        _remember(value)

    def redact(self, text: str) -> str:
        return _apply_redaction(text)

    def lookup_sync(self, ref: str) -> str | None:
        return self._values.get(ref)
