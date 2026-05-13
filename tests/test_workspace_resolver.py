"""``workspace=`` resolver tests — string / dict / instance / None."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import (
    Agent,
    ConfigError,
    InMemoryWorkspace,
    LocalDiskWorkspace,
    resolve_workspace,
)

# ---------------------------------------------------------------------------
# None / instance passthrough
# ---------------------------------------------------------------------------


def test_resolver_none_returns_none() -> None:
    """No workspace wired = no shared notebook. ``None`` is valid;
    the framework just skips the auto-wire path."""
    assert resolve_workspace(None) is None


def test_resolver_passes_through_instance() -> None:
    ws = InMemoryWorkspace()
    assert resolve_workspace(ws) is ws


# ---------------------------------------------------------------------------
# String forms
# ---------------------------------------------------------------------------


def test_resolver_string_memory() -> None:
    assert isinstance(resolve_workspace("memory"), InMemoryWorkspace)
    assert isinstance(resolve_workspace("inmemory"), InMemoryWorkspace)


def test_resolver_string_temp_returns_disk_workspace() -> None:
    ws = resolve_workspace("temp")
    assert isinstance(ws, LocalDiskWorkspace)
    # Temp dirs auto-cleanup on aclose; trust the typed flag rather
    # than poking the private attribute.
    import shutil
    shutil.rmtree(ws.root, ignore_errors=True)


def test_resolver_string_temp_with_prefix() -> None:
    ws = resolve_workspace("temp:my-research")
    assert isinstance(ws, LocalDiskWorkspace)
    assert "my-research" in str(ws.root)
    import shutil
    shutil.rmtree(ws.root, ignore_errors=True)


def test_resolver_temp_empty_prefix_rejected() -> None:
    with pytest.raises(ConfigError, match="needs a prefix"):
        resolve_workspace("temp:")


def test_resolver_string_path_opens_disk_workspace(tmp_path: Path) -> None:
    ws = resolve_workspace(str(tmp_path / "ws"))
    assert isinstance(ws, LocalDiskWorkspace)
    assert ws.root == (tmp_path / "ws").resolve()


def test_resolver_rejects_empty_string() -> None:
    with pytest.raises(ConfigError, match="empty string"):
        resolve_workspace("")


# ---------------------------------------------------------------------------
# Dict forms
# ---------------------------------------------------------------------------


def test_resolver_dict_memory_backend() -> None:
    ws = resolve_workspace({"backend": "memory"})
    assert isinstance(ws, InMemoryWorkspace)


def test_resolver_dict_disk_backend(tmp_path: Path) -> None:
    ws = resolve_workspace(
        {"backend": "disk", "path": str(tmp_path / "ws")}
    )
    assert isinstance(ws, LocalDiskWorkspace)


def test_resolver_dict_disk_aliases() -> None:
    """Backend names accept 'disk' / 'local' / 'filesystem' / 'fs'."""
    import tempfile

    for alias in ("disk", "local", "filesystem", "fs"):
        with tempfile.TemporaryDirectory() as raw:
            ws = resolve_workspace(
                {"backend": alias, "path": raw}
            )
            assert isinstance(ws, LocalDiskWorkspace), alias


def test_resolver_dict_disk_requires_path() -> None:
    with pytest.raises(ConfigError, match="requires a string 'path'"):
        resolve_workspace({"backend": "disk"})


def test_resolver_dict_temp_with_seed(tmp_path: Path) -> None:
    seed = tmp_path / "ref.md"
    seed.write_text("# Reference")
    ws = resolve_workspace(
        {"backend": "temp", "prefix": "x", "seed": [str(seed)]}
    )
    assert isinstance(ws, LocalDiskWorkspace)
    # The seed file landed under seeds/.
    seeded = ws.root / "seeds" / "ref.md"
    assert seeded.exists()
    import shutil
    shutil.rmtree(ws.root, ignore_errors=True)


def test_resolver_dict_aliases_type_and_name_for_backend() -> None:
    """TOML/YAML configs in the wild use any of ``backend`` /
    ``type`` / ``name`` for the discriminator. Match that."""
    assert isinstance(
        resolve_workspace({"type": "memory"}), InMemoryWorkspace
    )
    assert isinstance(
        resolve_workspace({"name": "memory"}), InMemoryWorkspace
    )


def test_resolver_dict_rejects_unknown_backend() -> None:
    with pytest.raises(ConfigError, match="not recognised"):
        resolve_workspace({"backend": "smoke-signals"})


def test_resolver_dict_rejects_missing_backend() -> None:
    with pytest.raises(ConfigError, match="must include 'backend'"):
        resolve_workspace({"path": "/tmp/ws"})


def test_resolver_dict_seed_must_be_string_or_list() -> None:
    with pytest.raises(ConfigError, match="must be a string"):
        resolve_workspace(
            {"backend": "temp", "seed": 42}  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_agent_accepts_workspace_string() -> None:
    agent = Agent("x", model="echo", workspace="memory")
    assert isinstance(agent._workspace, InMemoryWorkspace)  # noqa: SLF001


def test_agent_accepts_workspace_dict(tmp_path: Path) -> None:
    agent = Agent(
        "x",
        model="echo",
        workspace={"backend": "disk", "path": str(tmp_path / "ws")},
    )
    assert isinstance(agent._workspace, LocalDiskWorkspace)  # noqa: SLF001


def test_agent_accepts_workspace_instance() -> None:
    ws = InMemoryWorkspace()
    agent = Agent("x", model="echo", workspace=ws)
    assert agent._workspace is ws  # noqa: SLF001


def test_agent_without_workspace_has_none() -> None:
    agent = Agent("x", model="echo")
    assert agent._workspace is None  # noqa: SLF001
    assert agent._workspace_was_explicit is False  # noqa: SLF001
