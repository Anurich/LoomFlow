"""``telemetry=`` resolver tests.

Brings telemetry up to dict-form parity with model / memory /
runtime / audit_log.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ConfigError, NoTelemetry
from loomflow.observability import (
    ConsoleTelemetry,
    FileTelemetry,
    InMemoryTelemetry,
    resolve_telemetry,
)


def test_resolver_none_returns_no_telemetry() -> None:
    t = resolve_telemetry(None)
    assert isinstance(t, NoTelemetry)


def test_resolver_string_none() -> None:
    assert isinstance(resolve_telemetry("none"), NoTelemetry)
    assert isinstance(resolve_telemetry("noop"), NoTelemetry)


def test_resolver_string_console() -> None:
    assert isinstance(resolve_telemetry("console"), ConsoleTelemetry)


def test_resolver_string_memory() -> None:
    assert isinstance(resolve_telemetry("memory"), InMemoryTelemetry)
    assert isinstance(resolve_telemetry("inmemory"), InMemoryTelemetry)


def test_resolver_string_file(tmp_path: Path) -> None:
    out = tmp_path / "spans.jsonl"
    t = resolve_telemetry(f"file:{out}")
    assert isinstance(t, FileTelemetry)


def test_resolver_string_file_requires_path() -> None:
    with pytest.raises(ConfigError, match="needs a path"):
        resolve_telemetry("file:")


def test_resolver_dict_console() -> None:
    t = resolve_telemetry({"backend": "console"})
    assert isinstance(t, ConsoleTelemetry)


def test_resolver_dict_file(tmp_path: Path) -> None:
    out = tmp_path / "spans.jsonl"
    t = resolve_telemetry({"backend": "file", "path": str(out)})
    assert isinstance(t, FileTelemetry)


def test_resolver_dict_file_requires_path() -> None:
    with pytest.raises(ConfigError, match="requires 'path'"):
        resolve_telemetry({"backend": "file"})


def test_resolver_passes_through_instance() -> None:
    sink = InMemoryTelemetry()
    assert resolve_telemetry(sink) is sink


def test_resolver_rejects_unknown_string() -> None:
    with pytest.raises(ConfigError, match="unrecognised"):
        resolve_telemetry("smoke-signals")


def test_resolver_rejects_dict_without_backend() -> None:
    with pytest.raises(ConfigError, match="must include 'backend'"):
        resolve_telemetry({"path": "x"})


def test_resolver_dict_aliases_type_and_name_for_backend() -> None:
    assert isinstance(resolve_telemetry({"type": "console"}), ConsoleTelemetry)
    assert isinstance(resolve_telemetry({"name": "console"}), ConsoleTelemetry)


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_agent_accepts_telemetry_string() -> None:
    agent = Agent("hi", model="echo", telemetry="memory")
    assert isinstance(agent._telemetry, InMemoryTelemetry)


def test_agent_accepts_telemetry_dict(tmp_path: Path) -> None:
    out = tmp_path / "spans.jsonl"
    agent = Agent(
        "hi",
        model="echo",
        telemetry={"backend": "file", "path": str(out)},
    )
    assert isinstance(agent._telemetry, FileTelemetry)
