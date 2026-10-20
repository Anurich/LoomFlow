"""Regression tests for the runtime / security validation findings.

Each test reproduces a finding from the validation pass and fails on
the code before its fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from loomflow.security import FilesystemSandbox
from loomflow.tools import InProcessToolHost, tool
from loomflow.tools.builtin import grep_tool

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# grep / FilesystemSandbox containment
# ---------------------------------------------------------------------------


async def test_grep_skips_symlinks_that_leave_the_workdir(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOKEN=hunter2\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / "link.txt").symlink_to(outside / "secret.txt")
    (work / "ok.txt").write_text("TOKEN=public\n")

    out = await grep_tool(work).execute({"pattern": "TOKEN"})

    assert "hunter2" not in out
    assert "public" in out


@tool
async def open_path(path: str) -> str:
    """Open a path."""
    return path


@tool
async def nested(options: dict[str, Any]) -> str:
    """Take nested options."""
    return "ok"


async def _call(sandbox: FilesystemSandbox, name: str, args: dict[str, Any]) -> Any:
    return await sandbox.call(name, args, call_id="c")


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "file:///%65tc/passwd",
        "~/.ssh/id_rsa",
        "../outside/secret.txt",
        "a\x00b",
    ],
)
async def test_filesystem_sandbox_refuses_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    # With the process inside the root, a file: URL read as a relative
    # path ("file:/etc/passwd" under the cwd) used to pass.
    monkeypatch.chdir(root)
    sandbox = FilesystemSandbox(InProcessToolHost([open_path]), roots=[root])
    result = await _call(sandbox, "open_path", {"path": value})
    assert not result.ok and result.denied, value


async def test_filesystem_sandbox_relative_paths_resolve_against_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path is checked where workdir-based tools open it —
    under the root — not against the process's working directory."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    sandbox = FilesystemSandbox(InProcessToolHost([open_path]), roots=[root])
    result = await _call(sandbox, "open_path", {"path": "sub/file.txt"})
    assert result.ok


async def test_filesystem_sandbox_refuses_arguments_nested_past_scan_depth(
    tmp_path: Path,
) -> None:
    value: Any = {"path": "/etc/passwd"}
    for _ in range(40):
        value = {"options": value}
    sandbox = FilesystemSandbox(InProcessToolHost([nested]), roots=[tmp_path])
    result = await _call(sandbox, "nested", {"options": value})
    assert not result.ok and result.denied
