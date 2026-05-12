"""``runtime=`` resolver tests.

Covers the dict-form parity work that brings runtime up to the
same shape as model / memory / audit_log / telemetry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ConfigError
from loomflow.runtime import (
    InProcRuntime,
    SqliteRuntime,
    resolve_runtime,
)


def test_resolver_none_returns_inproc_default() -> None:
    rt = resolve_runtime(None)
    assert isinstance(rt, InProcRuntime)


def test_resolver_inproc_string() -> None:
    rt = resolve_runtime("inproc")
    assert isinstance(rt, InProcRuntime)


def test_resolver_sqlite_bare_returns_in_memory_journal() -> None:
    rt = resolve_runtime("sqlite")
    assert isinstance(rt, SqliteRuntime)
    assert str(rt.path) == ":memory:"


def test_resolver_sqlite_with_path(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    rt = resolve_runtime(f"sqlite:{db}")
    assert isinstance(rt, SqliteRuntime)
    assert rt.path == db


def test_resolver_dict_inproc() -> None:
    rt = resolve_runtime({"backend": "inproc"})
    assert isinstance(rt, InProcRuntime)


def test_resolver_dict_sqlite_with_path(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    rt = resolve_runtime({"backend": "sqlite", "path": str(db)})
    assert isinstance(rt, SqliteRuntime)
    assert rt.path == db


def test_resolver_dict_accepts_type_or_name_as_alias_for_backend() -> None:
    """TOML/YAML configs in the wild use any of ``backend``/``type``
    /``name``. Match the same flexibility as the model resolver."""
    assert isinstance(resolve_runtime({"type": "inproc"}), InProcRuntime)
    assert isinstance(resolve_runtime({"name": "inproc"}), InProcRuntime)


def test_resolver_passes_through_runtime_instance() -> None:
    rt = InProcRuntime()
    assert resolve_runtime(rt) is rt


def test_resolver_rejects_unknown_string() -> None:
    with pytest.raises(ConfigError, match="unrecognised"):
        resolve_runtime("badger")


def test_resolver_rejects_empty_string() -> None:
    with pytest.raises(ConfigError, match="empty string"):
        resolve_runtime("")


def test_resolver_rejects_dict_without_backend() -> None:
    with pytest.raises(ConfigError, match="must include 'backend'"):
        resolve_runtime({"path": "/tmp/x.db"})


def test_resolver_postgres_string_redirects_to_async_constructor() -> None:
    """Postgres needs an async connect, so the sync resolver refuses
    the URL form and tells the user how to wire it up properly."""
    with pytest.raises(ConfigError, match="async constructor"):
        resolve_runtime("postgres://user:pw@host/db")


def test_resolver_dict_postgres_redirects_to_async_constructor() -> None:
    with pytest.raises(ConfigError, match="async constructor"):
        resolve_runtime({"backend": "postgres", "url": "postgres://x"})


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_agent_accepts_runtime_string() -> None:
    agent = Agent("hi", model="echo", runtime="inproc")
    assert isinstance(agent.runtime, InProcRuntime)


def test_agent_accepts_runtime_dict(tmp_path: Path) -> None:
    db = tmp_path / "j.db"
    agent = Agent(
        "hi",
        model="echo",
        runtime={"backend": "sqlite", "path": str(db)},
    )
    assert isinstance(agent.runtime, SqliteRuntime)
    assert agent.runtime.path == db
