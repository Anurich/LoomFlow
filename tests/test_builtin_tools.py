"""Tests for the built-in filesystem + shell tools.

Covers:

* :func:`read_tool`: read existing file, line-numbered format,
  offset / limit paging, file-not-found, path-escape rejection,
  empty-file handling.
* :func:`write_tool`: create new file, overwrite existing, parent
  dir creation, path-escape rejection, ``create_parents=False``
  refusal.
* :func:`edit_tool`: exact-match replacement, multiple-match
  rejection without ``replace_all``, ``replace_all=True``, missing
  string error, file-not-found, path-escape rejection.
* :func:`bash_tool`: command runs, stdout/stderr capture, non-zero
  exit code, default deny-list rejection, ``allow_pattern``
  override, timeout, output truncation.
* :func:`filesystem_tools` bundle returns three tools.
* End-to-end: an Agent with all four tools registered actually
  uses them to produce a file via ScriptedModel-driven dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, Tool
from loomflow.core.types import ToolCall
from loomflow.tools import (
    PathEscapeError,
    bash_tool,
    default_workdir,
    edit_tool,
    filesystem_tools,
    find_tool,
    grep_tool,
    ls_tool,
    read_tool,
    write_tool,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# grep_tool / find_tool / ls_tool — the read-only navigation kernel
# ---------------------------------------------------------------------------


async def test_grep_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "b.py").write_text("x = 2\ndef world():\n    pass\n")
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": r"def \w+"})
    assert "a.py:1:" in out
    assert "b.py:2:" in out


async def test_grep_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("TARGET here\n")
    (tmp_path / "skip.txt").write_text("TARGET here\n")
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": "TARGET", "glob": "*.py"})
    assert "keep.py" in out
    assert "skip.txt" not in out


async def test_grep_skips_noise_dirs(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("NEEDLE\n")
    noise = tmp_path / "node_modules"
    noise.mkdir()
    (noise / "junk.py").write_text("NEEDLE\n")
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": "NEEDLE"})
    assert "src.py" in out
    assert "node_modules" not in out


async def test_grep_no_match(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("nothing here\n")
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": "absent_pattern"})
    assert "No matches" in out


async def test_grep_invalid_regex(tmp_path: Path) -> None:
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": "[unclosed"})
    assert "ERROR" in out
    assert "invalid regex" in out


async def test_grep_ignore_case(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("HELLO World\n")
    grep = grep_tool(tmp_path)
    sensitive = await grep.execute({"pattern": "hello"})
    insensitive = await grep.execute(
        {"pattern": "hello", "ignore_case": True}
    )
    assert "No matches" in sensitive
    assert "a.py:1:" in insensitive


async def test_grep_path_escape_rejected(tmp_path: Path) -> None:
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": "x", "path": "../../etc"})
    assert "ERROR" in out


async def test_find_by_glob(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("")
    (tmp_path / "test_main.py").write_text("")
    (tmp_path / "readme.md").write_text("")
    find = find_tool(tmp_path)
    py = await find.execute({"glob": "*.py"})
    assert "main.py" in py
    assert "test_main.py" in py
    assert "readme.md" not in py


async def test_find_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "deep.py").write_text("")
    find = find_tool(tmp_path)
    out = await find.execute({"glob": "*.py"})
    assert "pkg/sub/deep.py" in out


async def test_find_skips_noise(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("")
    find = find_tool(tmp_path)
    out = await find.execute({"glob": "*.py"})
    assert "real.py" in out
    assert ".venv" not in out


async def test_find_no_match(tmp_path: Path) -> None:
    find = find_tool(tmp_path)
    out = await find.execute({"glob": "*.nonexistent"})
    assert "No files matching" in out


async def test_ls_lists_directory(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("abc")
    (tmp_path / "subdir").mkdir()
    ls = ls_tool(tmp_path)
    out = await ls.execute({})
    assert "subdir/" in out
    assert "file.txt" in out


async def test_ls_dirs_first(tmp_path: Path) -> None:
    (tmp_path / "zzz_file.txt").write_text("")
    (tmp_path / "aaa_dir").mkdir()
    ls = ls_tool(tmp_path)
    out = await ls.execute({})
    # Directory should be listed before the file despite alpha order.
    assert out.index("aaa_dir/") < out.index("zzz_file.txt")


async def test_ls_empty_directory(tmp_path: Path) -> None:
    sub = tmp_path / "empty"
    sub.mkdir()
    ls = ls_tool(tmp_path)
    out = await ls.execute({"path": "empty"})
    assert "empty directory" in out


async def test_ls_not_a_directory(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    ls = ls_tool(tmp_path)
    out = await ls.execute({"path": "f.txt"})
    assert "ERROR" in out
    assert "not a directory" in out


async def test_filesystem_tools_bundle_now_six(tmp_path: Path) -> None:
    """filesystem_tools() now returns the 6-tool read-only +
    mutating kernel (read/write/edit/grep/find/ls), bash excluded."""
    tools = filesystem_tools(tmp_path)
    names = {t.name for t in tools}
    assert names == {
        "read", "write", "edit", "grep", "find", "ls",
    }


# ---------------------------------------------------------------------------
# read_tool
# ---------------------------------------------------------------------------


async def test_read_tool_returns_numbered_lines(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("first\nsecond\nthird\n")
    t = read_tool(tmp_path)
    out = await t.execute({"path": "hello.txt"})
    assert "first" in out
    assert "second" in out
    assert "third" in out
    # Line-numbered format: starts with whitespace + line num + tab
    assert "\t" in out
    assert "1" in out and "2" in out and "3" in out


async def test_read_tool_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("")
    t = read_tool(tmp_path)
    out = await t.execute({"path": "empty.txt"})
    assert "empty" in out.lower()


async def test_read_tool_offset_and_limit(tmp_path: Path) -> None:
    f = tmp_path / "many.txt"
    f.write_text("\n".join(f"line{i}" for i in range(10)))
    t = read_tool(tmp_path)
    out = await t.execute({"path": "many.txt", "offset": 3, "limit": 2})
    assert "line3" in out
    assert "line4" in out
    assert "line0" not in out
    assert "line5" not in out
    # Should hint at more lines available
    assert "more line" in out or "more line(s)" in out


async def test_read_tool_file_not_found(tmp_path: Path) -> None:
    t = read_tool(tmp_path)
    out = await t.execute({"path": "missing.txt"})
    assert out.startswith("ERROR: file not found")


async def test_read_tool_rejects_path_escape(tmp_path: Path) -> None:
    """Symlinks and ../.. must NOT escape the workdir."""
    sub = tmp_path / "sub"
    sub.mkdir()
    t = read_tool(sub)
    # Create a file ABOVE the workdir
    (tmp_path / "secret.txt").write_text("nope")
    out = await t.execute({"path": "../secret.txt"})
    assert out.startswith("ERROR:")
    assert "escapes" in out


async def test_read_tool_rejects_directory(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    t = read_tool(tmp_path)
    out = await t.execute({"path": "subdir"})
    assert out.startswith("ERROR:")
    assert "regular" in out


# ---------------------------------------------------------------------------
# write_tool
# ---------------------------------------------------------------------------


async def test_write_tool_creates_new_file(tmp_path: Path) -> None:
    t = write_tool(tmp_path)
    out = await t.execute({"path": "new.txt", "content": "hello world"})
    assert "wrote" in out
    assert "11 bytes" in out
    assert (tmp_path / "new.txt").read_text() == "hello world"


async def test_write_tool_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old content")
    t = write_tool(tmp_path)
    await t.execute({"path": "existing.txt", "content": "new content"})
    assert target.read_text() == "new content"


async def test_write_tool_creates_parent_dirs(tmp_path: Path) -> None:
    t = write_tool(tmp_path)
    await t.execute({"path": "a/b/c/deep.txt", "content": "x"})
    assert (tmp_path / "a" / "b" / "c" / "deep.txt").exists()


async def test_write_tool_refuses_missing_parents_when_disabled(
    tmp_path: Path,
) -> None:
    t = write_tool(tmp_path, create_parents=False)
    out = await t.execute({"path": "a/b/c.txt", "content": "x"})
    assert out.startswith("ERROR:")
    assert "parent" in out


async def test_write_tool_rejects_path_escape(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    t = write_tool(sub)
    out = await t.execute({"path": "../boom.txt", "content": "no"})
    assert out.startswith("ERROR:")
    assert "escapes" in out
    assert not (tmp_path / "boom.txt").exists()


async def test_write_tool_marked_destructive() -> None:
    """Permission policies inspect ``destructive`` to gate."""
    t = write_tool("/tmp")
    assert t.destructive is True


# ---------------------------------------------------------------------------
# edit_tool
# ---------------------------------------------------------------------------


async def test_edit_tool_replaces_unique_match(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("# Title\n\nold body\n\n## More\n")
    t = edit_tool(tmp_path)
    out = await t.execute({
        "path": "doc.md",
        "old_string": "old body",
        "new_string": "new body",
    })
    assert "replaced 1" in out
    assert target.read_text() == "# Title\n\nnew body\n\n## More\n"


async def test_edit_tool_rejects_when_not_found(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello")
    t = edit_tool(tmp_path)
    out = await t.execute({
        "path": "doc.md",
        "old_string": "missing",
        "new_string": "x",
    })
    assert out.startswith("ERROR:")
    assert "not found" in out
    # File unchanged
    assert target.read_text() == "hello"


async def test_edit_tool_rejects_multiple_matches_by_default(
    tmp_path: Path,
) -> None:
    target = tmp_path / "doc.md"
    target.write_text("foo bar foo baz foo")
    t = edit_tool(tmp_path)
    out = await t.execute({
        "path": "doc.md",
        "old_string": "foo",
        "new_string": "X",
    })
    assert out.startswith("ERROR:")
    assert "3 times" in out
    assert "replace_all" in out
    # Unchanged
    assert "foo" in target.read_text()


async def test_edit_tool_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("foo bar foo baz foo")
    t = edit_tool(tmp_path)
    out = await t.execute({
        "path": "doc.md",
        "old_string": "foo",
        "new_string": "X",
        "replace_all": True,
    })
    assert "replaced 3" in out
    assert target.read_text() == "X bar X baz X"


async def test_edit_tool_can_delete_via_empty_replacement(
    tmp_path: Path,
) -> None:
    target = tmp_path / "doc.md"
    target.write_text("keep this\nremove me\nkeep that\n")
    t = edit_tool(tmp_path)
    await t.execute({
        "path": "doc.md",
        "old_string": "remove me\n",
        "new_string": "",
    })
    assert "remove me" not in target.read_text()


async def test_edit_tool_file_not_found(tmp_path: Path) -> None:
    t = edit_tool(tmp_path)
    out = await t.execute({
        "path": "ghost.txt",
        "old_string": "x",
        "new_string": "y",
    })
    assert out.startswith("ERROR: file not found")


async def test_edit_tool_rejects_path_escape(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "outside.txt").write_text("hi")
    t = edit_tool(sub)
    out = await t.execute({
        "path": "../outside.txt",
        "old_string": "hi",
        "new_string": "x",
    })
    assert out.startswith("ERROR:")
    assert "escapes" in out


# ---------------------------------------------------------------------------
# bash_tool
# ---------------------------------------------------------------------------


async def test_bash_tool_runs_simple_command(tmp_path: Path) -> None:
    t = bash_tool(tmp_path, timeout=5.0)
    out = await t.execute({"command": "echo hello"})
    assert "hello" in out
    assert "exit=0" in out


async def test_bash_tool_captures_stderr(tmp_path: Path) -> None:
    t = bash_tool(tmp_path, timeout=5.0)
    out = await t.execute(
        {"command": "echo err 1>&2; echo out"}
    )
    assert "out" in out
    assert "err" in out
    assert "stderr" in out
    assert "stdout" in out


async def test_bash_tool_runs_in_workdir(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("here")
    t = bash_tool(tmp_path, timeout=5.0)
    out = await t.execute({"command": "ls"})
    assert "marker.txt" in out


async def test_bash_tool_nonzero_exit(tmp_path: Path) -> None:
    t = bash_tool(tmp_path, timeout=5.0)
    out = await t.execute({"command": "false"})
    assert "exit=1" in out


async def test_bash_tool_default_denylist_allows_normal_rm_rf(
    tmp_path: Path,
) -> None:
    """``rm -rf /tmp/whatever`` is normal cleanup, NOT a root wipe.
    The deny list should only fire on ``rm -rf /`` (with whitespace
    or end-of-line after the slash)."""
    t = bash_tool(tmp_path)
    out = await t.execute({"command": "rm -rf /tmp/whatever"})
    assert not out.startswith("ERROR: blocked")


async def test_bash_tool_blocks_dangerous_patterns(tmp_path: Path) -> None:
    t = bash_tool(tmp_path)
    for cmd in [
        "rm -rf /",
        "sudo rm anything",
        "mkfs.ext4 /dev/sda1",
    ]:
        out = await t.execute({"command": cmd})
        assert out.startswith("ERROR: blocked"), cmd


async def test_bash_tool_allow_pattern_overrides_denylist(
    tmp_path: Path,
) -> None:
    """When allow_pattern is supplied, it REPLACES the deny list."""
    t = bash_tool(
        tmp_path, allow_pattern=lambda cmd: cmd.startswith("echo ")
    )
    # echo is allowed
    out = await t.execute({"command": "echo ok"})
    assert "ok" in out
    # ls is NOT allowed (allow_pattern only allows echo)
    out = await t.execute({"command": "ls"})
    assert out.startswith("ERROR: blocked")
    assert "user-supplied" in out


async def test_bash_tool_timeout_kills_long_command(
    tmp_path: Path,
) -> None:
    t = bash_tool(tmp_path, timeout=0.5)
    out = await t.execute({"command": "sleep 5"})
    assert out.startswith("ERROR: command timed out")


async def test_bash_tool_marked_destructive() -> None:
    t = bash_tool("/tmp")
    assert t.destructive is True


# ---------------------------------------------------------------------------
# filesystem_tools bundle
# ---------------------------------------------------------------------------


def test_filesystem_tools_returns_kernel(tmp_path: Path) -> None:
    """``filesystem_tools()`` returns the 6-tool read-only +
    mutating kernel (v0.10.1: grew from 3 to 6 — added the
    grep / find / ls navigation tools). ``bash`` stays excluded."""
    tools = filesystem_tools(tmp_path)
    assert len(tools) == 6
    assert {t.name for t in tools} == {
        "read", "write", "edit", "grep", "find", "ls",
    }
    assert all(isinstance(t, Tool) for t in tools)


# ---------------------------------------------------------------------------
# PathEscapeError class
# ---------------------------------------------------------------------------


def test_path_escape_error_is_value_error_subclass() -> None:
    assert issubclass(PathEscapeError, ValueError)


# ---------------------------------------------------------------------------
# Default workdir (no-arg factory calls)
# ---------------------------------------------------------------------------


def test_default_workdir_lazy_creates_tempdir() -> None:
    """Calling default_workdir() lazily creates a tempdir under
    /tmp/jeeves_agent_*. Subsequent calls return the same path."""
    d1 = default_workdir()
    d2 = default_workdir()
    assert d1.exists()
    assert d1.is_dir()
    assert d1 == d2
    assert "jeeves_agent_" in str(d1)


async def test_no_arg_factories_share_default_workdir() -> None:
    """``read_tool()`` / ``write_tool()`` / ``edit_tool()`` /
    ``bash_tool()`` called without arguments all see the same
    tempdir, so write→read works without explicit wiring."""
    w = write_tool()
    r = read_tool()
    out = await w.execute(
        {"path": "shared_default_test.txt", "content": "hi from default"}
    )
    assert "wrote" in out

    out = await r.execute({"path": "shared_default_test.txt"})
    assert "hi from default" in out


async def test_filesystem_tools_no_arg_uses_default() -> None:
    tools = filesystem_tools()
    assert len(tools) == 6
    # Use them — write then read (read/write are the first two,
    # order preserved for back-compat).
    await tools[1].execute({"path": "fs_default.txt", "content": "x"})
    out = await tools[0].execute({"path": "fs_default.txt"})
    assert "x" in out


# ---------------------------------------------------------------------------
# End-to-end: agent uses the tools via tool dispatch
# ---------------------------------------------------------------------------


async def test_agent_can_use_filesystem_tools_end_to_end(
    tmp_path: Path,
) -> None:
    """Smoke test the full tool-dispatch path: an Agent registers
    write+read+edit, the model emits tool calls, the registry
    dispatches them, and the file actually changes on disk."""
    model = ScriptedModel([
        # Turn 1: write a file
        ScriptedTurn(
            tool_calls=[
                ToolCall(
                    id="c1",
                    tool="write",
                    args={
                        "path": "report.md",
                        "content": "# Title\n\nbody\n",
                    },
                )
            ]
        ),
        # Turn 2: edit it
        ScriptedTurn(
            tool_calls=[
                ToolCall(
                    id="c2",
                    tool="edit",
                    args={
                        "path": "report.md",
                        "old_string": "body",
                        "new_string": "polished body",
                    },
                )
            ]
        ),
        # Turn 3: read to verify
        ScriptedTurn(
            tool_calls=[
                ToolCall(
                    id="c3",
                    tool="read",
                    args={"path": "report.md"},
                )
            ]
        ),
        # Turn 4: final answer
        ScriptedTurn(text="Done — wrote, edited, verified."),
    ])

    agent = Agent(
        "test agent",
        model=model,
        tools=filesystem_tools(tmp_path),
    )
    result = await agent.run("create a report")
    assert "Done" in result.output
    final = (tmp_path / "report.md").read_text()
    assert "polished body" in final
    assert "body\n" not in final.replace("polished body", "")
