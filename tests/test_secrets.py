"""M10.6 — Secrets protocol concrete impls + model-resolver wiring.

Locks in the contract:

* :class:`EnvSecrets` reads from ``os.environ`` (preserves the
  pre-M10 default behaviour for API keys).
* :class:`DictSecrets` reads from an explicit dict (test /
  vault-fetched-once-at-startup use case).
* Both impls implement ``lookup_sync`` — the constructor-time
  path the model adapters use when no ``api_key=`` is supplied.
* ``redact()`` masks common API-key shapes so audit logs don't
  leak credentials.
* ``Agent(secrets=...)`` flows through to model adapter
  construction — ``OpenAIModel(secrets=...)`` resolves
  ``OPENAI_API_KEY`` from the supplied backend, not from
  ``os.environ``.
"""

from __future__ import annotations

import pytest

from jeevesagent import DictSecrets, EnvSecrets

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# EnvSecrets
# ---------------------------------------------------------------------------


def test_env_secrets_lookup_sync_reads_from_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_KEY", "from-env")
    s = EnvSecrets()
    assert s.lookup_sync("EXAMPLE_KEY") == "from-env"


def test_env_secrets_lookup_sync_returns_none_for_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    assert EnvSecrets().lookup_sync("DEFINITELY_NOT_SET") is None


async def test_env_secrets_resolve_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASYNC_RESOLVED", "ok")
    assert await EnvSecrets().resolve("ASYNC_RESOLVED") == "ok"


async def test_env_secrets_store_raises() -> None:
    """Mutating ``os.environ`` from a Secrets impl is rude — the
    framework refuses."""
    with pytest.raises(NotImplementedError):
        await EnvSecrets().store("X", "y")


# ---------------------------------------------------------------------------
# DictSecrets
# ---------------------------------------------------------------------------


def test_dict_secrets_lookup_sync() -> None:
    s = DictSecrets({"K": "v"})
    assert s.lookup_sync("K") == "v"
    assert s.lookup_sync("MISSING") is None


async def test_dict_secrets_resolve_and_store() -> None:
    s = DictSecrets()
    await s.store("K", "v1")
    assert await s.resolve("K") == "v1"
    await s.store("K", "v2")
    assert await s.resolve("K") == "v2"


async def test_dict_secrets_resolve_raises_on_missing() -> None:
    with pytest.raises(KeyError):
        await DictSecrets().resolve("MISSING")


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redact_masks_openai_style_keys() -> None:
    redacted = EnvSecrets().redact(
        "log line containing sk-AbCdEf01234567890XYZabc as the key"
    )
    assert "sk-AbCdEf" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_masks_anthropic_style_keys() -> None:
    redacted = DictSecrets().redact(
        "key=sk-ant-api01-AbCdEfGhIj1234567890 in payload"
    )
    assert "sk-ant" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_passes_through_clean_text() -> None:
    benign = "no secrets in this string"
    assert EnvSecrets().redact(benign) == benign


# ---------------------------------------------------------------------------
# Model resolver integration
# ---------------------------------------------------------------------------


def test_openai_model_uses_secrets_for_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``secrets=`` is passed AND no explicit ``api_key=``,
    OpenAIModel reads from the supplied Secrets backend rather
    than ``os.environ``."""
    # Make sure the env var ISN'T set, so we know the secrets
    # lookup is what supplied the key.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured: dict[str, str | None] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    # Patch the openai module the adapter imports.
    import sys
    fake_openai = type(sys)("openai")
    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from jeevesagent.model.openai import OpenAIModel

    secrets = DictSecrets({"OPENAI_API_KEY": "sk-from-secrets"})
    OpenAIModel("gpt-4o", secrets=secrets)
    assert captured["api_key"] == "sk-from-secrets"


def test_explicit_api_key_takes_precedence_over_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``api_key=`` wins over ``secrets.lookup_sync``,
    which wins over ``os.environ``."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    captured: dict[str, str | None] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            captured["api_key"] = api_key

    import sys
    fake_openai = type(sys)("openai")
    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from jeevesagent.model.openai import OpenAIModel

    secrets = DictSecrets({"OPENAI_API_KEY": "from-secrets"})
    OpenAIModel("gpt-4o", api_key="explicit", secrets=secrets)
    assert captured["api_key"] == "explicit"
