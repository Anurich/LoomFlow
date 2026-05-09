"""Sandbox tests — pass-through and filesystem path validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from loomflow import Agent, NoSandbox, tool
from loomflow.core.types import ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.security import FilesystemSandbox
from loomflow.tools import InProcessToolHost

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# NoSandbox
# ---------------------------------------------------------------------------


async def test_no_sandbox_passes_through_list_and_call() -> None:
    @tool
    async def ping() -> str:
        """Return pong."""
        return "pong"

    inner = InProcessToolHost([ping])
    sandbox = NoSandbox(inner)

    defs = await sandbox.list_tools()
    assert {d.name for d in defs} == {"ping"}

    result = await sandbox.call("ping", {}, call_id="c1")
    assert result.ok
    assert result.output == "pong"
    assert result.call_id == "c1"


async def test_no_sandbox_works_inside_agent_loop() -> None:
    @tool
    async def echo(msg: str) -> str:
        """Return the message."""
        return msg

    sandbox = NoSandbox(InProcessToolHost([echo]))
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", tool="echo", args={"msg": "hi"})
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model, tools=sandbox)
    result = await agent.run("...")
    assert "done" in result.output


# ---------------------------------------------------------------------------
# FilesystemSandbox
# ---------------------------------------------------------------------------


async def test_filesystem_sandbox_allows_paths_inside_root(tmp_path: Path) -> None:
    @tool
    def read_file(path: str) -> str:
        """Read file contents."""
        return Path(path).read_text()

    target = tmp_path / "ok.txt"
    target.write_text("hello")

    sandbox = FilesystemSandbox(
        InProcessToolHost([read_file]),
        roots=[tmp_path],
    )
    result = await sandbox.call(
        "read_file", {"path": str(target)}, call_id="c1"
    )
    assert result.ok
    assert result.output == "hello"


async def test_filesystem_sandbox_blocks_paths_outside_root(tmp_path: Path) -> None:
    @tool
    async def read_file(path: str) -> str:
        """Read file contents."""
        raise AssertionError("must not run; sandbox should deny")

    sandbox = FilesystemSandbox(
        InProcessToolHost([read_file]),
        roots=[tmp_path],
    )
    # Try to read /etc/passwd — well outside tmp_path.
    result = await sandbox.call(
        "read_file", {"path": "/etc/passwd"}, call_id="c1"
    )
    assert not result.ok
    assert result.denied
    assert result.reason is not None
    assert "outside" in result.reason


async def test_filesystem_sandbox_resolves_symlink_escapes(
    tmp_path: Path,
) -> None:
    """Symlinks pointing outside the root are caught after resolution."""

    @tool
    async def read_file(path: str) -> str:
        """Read file contents."""
        raise AssertionError("symlink escape must be denied")

    # Create a symlink inside tmp_path that points to /etc.
    link = tmp_path / "etc"
    try:
        link.symlink_to("/etc")
    except OSError:
        pytest.skip("filesystem doesn't support symlinks")

    sandbox = FilesystemSandbox(
        InProcessToolHost([read_file]),
        roots=[tmp_path],
    )
    result = await sandbox.call(
        "read_file", {"path": str(link / "passwd")}, call_id="c1"
    )
    assert not result.ok
    assert result.denied


async def test_filesystem_sandbox_explicit_path_args_overrides_auto_detect(
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    @tool
    async def grep(query: str, where: str) -> str:
        """Search for a query in a directory."""
        captured["where"] = where
        return f"hits in {where}"

    # Auto-detect would skip ``where``; explicit path_args=["where"]
    # makes it required to be inside root.
    sandbox = FilesystemSandbox(
        InProcessToolHost([grep]),
        roots=[tmp_path],
        path_args=["where"],
    )

    inside = await sandbox.call(
        "grep",
        {"query": "needle", "where": str(tmp_path)},
        call_id="ok",
    )
    assert inside.ok

    outside = await sandbox.call(
        "grep",
        {"query": "needle", "where": "/var"},
        call_id="bad",
    )
    assert not outside.ok
    assert outside.denied


async def test_filesystem_sandbox_skips_non_path_string_arguments(
    tmp_path: Path,
) -> None:
    """Strings that don't look like paths and don't match a path-arg
    name pass through without validation."""

    @tool
    async def echo(msg: str) -> str:
        """Echo the message."""
        return msg

    sandbox = FilesystemSandbox(
        InProcessToolHost([echo]),
        roots=[tmp_path],
    )
    # ``msg`` isn't a path-named arg; value has no path separator.
    result = await sandbox.call(
        "echo", {"msg": "hello world"}, call_id="c1"
    )
    assert result.ok
    assert result.output == "hello world"


def test_filesystem_sandbox_requires_at_least_one_root() -> None:
    with pytest.raises(ValueError):
        FilesystemSandbox(InProcessToolHost([]), roots=[])


async def test_filesystem_sandbox_multiple_roots_all_allowed(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()

    @tool
    async def read_file(path: str) -> str:
        """Read file contents."""
        return f"got {path}"

    sandbox = FilesystemSandbox(
        InProcessToolHost([read_file]),
        roots=[root_a, root_b],
    )

    file_a = root_a / "x.txt"
    file_a.write_text("a")
    res_a = await sandbox.call("read_file", {"path": str(file_a)}, call_id="ca")
    assert res_a.ok

    file_b = root_b / "y.txt"
    file_b.write_text("b")
    res_b = await sandbox.call("read_file", {"path": str(file_b)}, call_id="cb")
    assert res_b.ok

    res_outside = await sandbox.call(
        "read_file", {"path": "/etc/passwd"}, call_id="cc"
    )
    assert not res_outside.ok


def test_filesystem_sandbox_introspection() -> None:
    inner = InProcessToolHost([])
    sandbox = FilesystemSandbox(inner, roots=["/tmp"])
    assert sandbox.inner is inner
    assert len(sandbox.roots) == 1
    assert sandbox.roots[0] == Path("/tmp").resolve()


# ---------------------------------------------------------------------------
# Stress: tilde expansion + relative paths
# ---------------------------------------------------------------------------


async def test_relative_path_resolved_against_cwd(tmp_path: Path) -> None:
    """A relative path is resolved before the containment check."""

    @tool
    def read_file(path: str) -> str:
        """Read file contents."""
        return Path(path).read_text()

    target = tmp_path / "rel.txt"
    target.write_text("ok")

    cwd_was = os.getcwd()
    os.chdir(tmp_path)
    try:
        sandbox = FilesystemSandbox(
            InProcessToolHost([read_file]),
            roots=[tmp_path],
        )
        result = await sandbox.call(
            "read_file", {"path": "rel.txt"}, call_id="c1"
        )
        assert result.ok
    finally:
        os.chdir(cwd_was)
