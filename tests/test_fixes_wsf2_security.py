"""Regression tests for the security-layer review fixes (WSF2 batch).

Covers:

* Audit logs warn (once per process) when constructed with an
  empty signing secret — an empty-key HMAC is forgeable, so
  verify_signature would be meaningless.
* Broadened redact() defaults: AWS session keys, Slack tokens,
  Google API keys, Bearer headers, generic *_API_KEY / *_TOKEN
  assignments — without over-matching prose.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import loomflow.security.audit as audit_mod
from loomflow.security import DictSecrets
from loomflow.security.audit import FileAuditLog, InMemoryAuditLog

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Empty-secret warning
# ---------------------------------------------------------------------------


def test_inmemory_empty_secret_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_mod, "_EMPTY_SECRET_WARNED", False)
    with pytest.warns(UserWarning, match="empty signing secret"):
        InMemoryAuditLog()


def test_file_empty_secret_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audit_mod, "_EMPTY_SECRET_WARNED", False)
    with pytest.warns(UserWarning, match="empty signing secret"):
        FileAuditLog(tmp_path / "audit.jsonl")


def test_empty_secret_warning_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_mod, "_EMPTY_SECRET_WARNED", False)
    with pytest.warns(UserWarning):
        InMemoryAuditLog()
    # Second construction stays silent — the warning fired already.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InMemoryAuditLog()


def test_real_secret_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audit_mod, "_EMPTY_SECRET_WARNED", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InMemoryAuditLog(secret="real-key")
        FileAuditLog(tmp_path / "audit.jsonl", secret="real-key")


async def test_signed_entries_still_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warning must not change signing behaviour."""
    monkeypatch.setattr(audit_mod, "_EMPTY_SECRET_WARNED", True)
    log = InMemoryAuditLog(secret="k1")
    entry = await log.append(
        session_id="s", actor="a", action="run.start", payload={}
    )
    assert audit_mod.verify_signature(entry, "k1")
    assert not audit_mod.verify_signature(entry, "k2")


# ---------------------------------------------------------------------------
# redact() — broadened defaults
# ---------------------------------------------------------------------------


def _redact(text: str) -> str:
    return DictSecrets().redact(text)


def test_redact_aws_session_key() -> None:
    out = _redact("temp creds: ASIAIOSFODNN7EXAMPLE in use")
    assert "ASIAIOSFODNN7EXAMPLE" not in out
    assert "[REDACTED]" in out


def test_redact_aws_access_key_still_masked() -> None:
    out = _redact("key AKIAIOSFODNN7EXAMPLE here")
    assert "AKIAIOSFODNN7EXAMPLE" not in out


def test_redact_slack_tokens() -> None:
    out = _redact("slack: xoxb-123456789012-abcdefghijkl done")
    assert "xoxb-123456789012" not in out
    assert "[REDACTED]" in out


def test_redact_google_api_key() -> None:
    key = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"[:35]
    assert len(key) == 39
    out = _redact(f"maps key {key} end")
    assert key not in out
    assert "[REDACTED]" in out


def test_redact_bearer_header_keeps_scheme() -> None:
    out = _redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def123456")
    assert "eyJhbGciOiJIUzI1NiJ9" not in out
    assert "Bearer [REDACTED]" in out


def test_redact_generic_api_key_assignment_keeps_name() -> None:
    out = _redact("export MYSERVICE_API_KEY=super-secret-value-1234")
    assert "super-secret-value-1234" not in out
    assert "MYSERVICE_API_KEY=[REDACTED]" in out


def test_redact_generic_token_assignment() -> None:
    out = _redact('GITLAB_TOKEN="glpat-abcdef1234567890"')
    assert "glpat-abcdef1234567890" not in out
    assert "GITLAB_TOKEN=" in out


def test_redact_does_not_eat_prose() -> None:
    benign = (
        "The bearer of this message asked about tokens. "
        "Set your api key in the dashboard; the token is short."
    )
    assert _redact(benign) == benign


def test_redact_short_values_not_matched() -> None:
    # Values under 8 chars are below the generic pattern's floor —
    # placeholders like X_TOKEN=abc stay readable.
    text = "X_TOKEN=abc"
    assert _redact(text) == text
