"""Built-in tools for filesystem + shell agents.

Four tools that any :class:`~loomflow.Agent` can register to
gain the canonical "Claude-Code-shaped" capability set:

* :func:`read_tool` — read a text file
* :func:`write_tool` — create or overwrite a file
* :func:`edit_tool` — find-and-replace inside an existing file
* :func:`bash_tool` — run a shell command, capture output

All four are **factory functions**: they take a ``workdir`` (and
in the case of ``bash_tool`` a few safety knobs) and return a
ready-to-register :class:`Tool` instance. The closure captures the
workdir, so the resulting tool is workdir-scoped — the model can't
escape it via ``../`` traversal or absolute paths.

Usage::

    from loomflow import (
        Agent, read_tool, write_tool, edit_tool, bash_tool,
    )

    agent = Agent(
        "You are a research agent.",
        model="gpt-4.1-mini",
        tools=[
            read_tool(workdir="/tmp/agent_work"),
            write_tool(workdir="/tmp/agent_work"),
            edit_tool(workdir="/tmp/agent_work"),
            bash_tool(workdir="/tmp/agent_work", timeout=30.0),
        ],
    )

Or as a bundle::

    from loomflow import filesystem_tools, bash_tool

    agent = Agent(
        "...",
        model="...",
        tools=filesystem_tools("/tmp/agent_work") + [bash_tool("/tmp/agent_work")],
    )

Safety
------

* **Workdir-scoped by default.** Read / write / edit refuse paths
  that resolve outside the workdir. ``bash_tool`` runs commands
  with the workdir as ``cwd``.
* **Timeout on bash.** Default 30 seconds; override via the
  ``timeout`` kwarg. Commands that exceed the timeout are killed.
* **Destructive-command denylist.** ``bash_tool`` rejects a small
  set of obviously-dangerous patterns (``rm -rf /``, ``sudo``,
  ``mkfs``, etc.) by default. Override via the ``allow_pattern``
  callable for advanced use.
* **Edit requires unique match.** ``edit_tool``'s ``old_string``
  must appear EXACTLY once in the file (unless ``replace_all=True``
  is passed in the call) — forces the model to provide enough
  context for unambiguous edits, the same approach Claude Code
  takes.

These will be the foundation of the upcoming Deep Agent architecture
(planner + filesystem state + subagent registry).
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import anyio

from .registry import Tool

# ---------------------------------------------------------------------------
# Default workdir — lazily created on first use, shared across all
# built-in tool factories that don't get an explicit workdir. So
# ``read_tool()`` + ``write_tool()`` + ``edit_tool()`` + ``bash_tool()``
# all see the same tempdir without the caller wiring it up.
# ---------------------------------------------------------------------------


_DEFAULT_WORKDIR: Path | None = None


def default_workdir() -> Path:
    """Return the framework's default workdir for built-in tools,
    creating it lazily on first call.

    The directory is a fresh tempdir under ``$TMPDIR/jeeves_agent_*``,
    created once per process. All built-in tool factories share it
    when called without an explicit ``workdir`` argument, so an
    Agent that registers ``read_tool()`` and ``write_tool()`` (no
    args) sees the same place.

    The directory is NOT auto-cleaned at process exit — leave that
    to the OS's tempdir cleanup so debug data survives a crash.
    """
    global _DEFAULT_WORKDIR
    if _DEFAULT_WORKDIR is None:
        _DEFAULT_WORKDIR = Path(
            tempfile.mkdtemp(prefix="jeeves_agent_")
        ).resolve()
    return _DEFAULT_WORKDIR


def _resolve_workdir(workdir: Path | str | None) -> Path:
    """Resolve an explicit workdir or fall back to the shared default."""
    if workdir is None:
        return default_workdir()
    return Path(workdir).resolve()

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _resolve_within(workdir: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``workdir``; raise if it escapes."""
    target = (workdir / rel).resolve()
    workdir_resolved = workdir.resolve()
    try:
        target.relative_to(workdir_resolved)
    except ValueError as exc:
        raise PathEscapeError(
            f"path {rel!r} escapes workdir {str(workdir_resolved)!r}"
        ) from exc
    return target


class PathEscapeError(ValueError):
    """Raised when a tool argument resolves outside its workdir."""


# ---------------------------------------------------------------------------
# read_tool
# ---------------------------------------------------------------------------


_DEFAULT_READ_LINE_LIMIT = 2000


def read_tool(
    workdir: Path | str | None = None,
    *,
    name: str = "read",
    line_limit: int = _DEFAULT_READ_LINE_LIMIT,
) -> Tool:
    """Build a :class:`Tool` that reads a text file under ``workdir``.

    The tool's signature seen by the model:
        ``read(path: str, offset: int = 0, limit: int | None = None)``

    Returns the file's text with line numbers prefixed (one
    line per output line), in the same format Claude Code's Read
    tool uses — that lets the ``edit`` tool work without ambiguity
    later. Long files are truncated to ``line_limit`` lines per
    call; pass ``offset`` / ``limit`` to read further chunks.

    Errors (file-not-found, path-escape) are returned as a string
    starting with ``"ERROR: "`` rather than raising — the model
    sees them as a tool result and can adjust.

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir (shared with the other built-in tools called without
    a workdir).
    """
    workdir_path = _resolve_workdir(workdir)

    async def _read(
        path: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        try:
            target = _resolve_within(workdir_path, path)
        except PathEscapeError as exc:
            return f"ERROR: {exc}"
        if not target.exists():
            return f"ERROR: file not found: {path}"
        if not target.is_file():
            return f"ERROR: not a regular file: {path}"

        text = await anyio.to_thread.run_sync(target.read_text)
        lines = text.splitlines()
        end = (
            min(offset + limit, len(lines))
            if limit is not None
            else min(offset + line_limit, len(lines))
        )
        chunk = lines[offset:end]
        numbered = "\n".join(
            f"{offset + i + 1:6d}\t{line}"
            for i, line in enumerate(chunk)
        )
        if end < len(lines):
            numbered += (
                f"\n... ({len(lines) - end} more line(s); "
                f"call read() again with offset={end})"
            )
        return numbered or "(empty file)"

    return Tool(
        name=name,
        description=(
            f"Read a text file under {str(workdir_path)}. Returns "
            "the contents with 1-indexed line numbers prefixed "
            "(format: '   N\\tline content'). Use these line "
            "numbers when planning edits. For long files, pass "
            "offset (default 0) and limit (default "
            f"{_DEFAULT_READ_LINE_LIMIT}) to page through."
        ),
        fn=_read,
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path relative to the workdir. Cannot "
                        "escape the workdir via '..' or absolute "
                        "paths."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Zero-indexed line offset to start reading "
                        "from. Default 0."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max lines to return. Default "
                        f"{_DEFAULT_READ_LINE_LIMIT}."
                    ),
                },
            },
            "required": ["path"],
        },
    )


# ---------------------------------------------------------------------------
# write_tool
# ---------------------------------------------------------------------------


def write_tool(
    workdir: Path | str | None = None,
    *,
    name: str = "write",
    create_parents: bool = True,
) -> Tool:
    """Build a :class:`Tool` that writes / overwrites a text file
    under ``workdir``.

    The tool's signature seen by the model:
        ``write(path: str, content: str)``

    Overwrites existing files. With ``create_parents=True`` (the
    default), missing parent directories are created automatically.

    Returns a confirmation string with the byte count, or an
    ``"ERROR: "``-prefixed message on failure.

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir (shared with the other built-in tools).
    """
    workdir_path = _resolve_workdir(workdir)

    async def _write(path: str, content: str) -> str:
        try:
            target = _resolve_within(workdir_path, path)
        except PathEscapeError as exc:
            return f"ERROR: {exc}"

        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        elif not target.parent.exists():
            return (
                f"ERROR: parent directory does not exist: "
                f"{target.parent.relative_to(workdir_path)}"
            )

        await anyio.to_thread.run_sync(target.write_text, content)
        return (
            f"wrote {len(content)} bytes to {path} "
            f"({content.count(chr(10)) + 1} line(s))"
        )

    return Tool(
        name=name,
        description=(
            f"Create or OVERWRITE a text file under {str(workdir_path)}. "
            "Use this for new files or full rewrites; use the edit "
            "tool for in-place modifications. Parent directories "
            "are created automatically. Returns byte count + line "
            "count on success."
        ),
        fn=_write,
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path relative to the workdir. Cannot "
                        "escape the workdir."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write.",
                },
            },
            "required": ["path", "content"],
        },
        destructive=True,  # overwrites — flag for permission policies
    )


# ---------------------------------------------------------------------------
# edit_tool
# ---------------------------------------------------------------------------


def edit_tool(
    workdir: Path | str | None = None,
    *,
    name: str = "edit",
) -> Tool:
    """Build a :class:`Tool` that does find-and-replace inside an
    existing file under ``workdir``.

    The tool's signature seen by the model:
        ``edit(path: str, old_string: str, new_string: str,
               replace_all: bool = False)``

    Behaviour matches Claude Code's Edit tool:

    * ``old_string`` must be EXACTLY present in the file. Mismatch
      (whitespace, indentation, line breaks) → error.
    * ``old_string`` must appear EXACTLY once in the file unless
      ``replace_all=True`` is passed — forces the model to give
      enough surrounding context for unambiguous matches.
    * ``new_string`` replaces ``old_string`` (or every occurrence
      if ``replace_all=True``).

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir (shared with the other built-in tools).
    """
    workdir_path = _resolve_workdir(workdir)

    async def _edit(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        try:
            target = _resolve_within(workdir_path, path)
        except PathEscapeError as exc:
            return f"ERROR: {exc}"
        if not target.exists():
            return f"ERROR: file not found: {path}"
        if not target.is_file():
            return f"ERROR: not a regular file: {path}"

        text = await anyio.to_thread.run_sync(target.read_text)
        count = text.count(old_string)
        if count == 0:
            return (
                f"ERROR: old_string not found in {path}. "
                "It must match EXACTLY (whitespace, indentation, "
                "line breaks)."
            )
        if count > 1 and not replace_all:
            return (
                f"ERROR: old_string appears {count} times in "
                f"{path}; pass replace_all=True or provide more "
                "surrounding context to make the match unique."
            )

        if replace_all:
            new_text = text.replace(old_string, new_string)
            replaced = count
        else:
            new_text = text.replace(old_string, new_string, 1)
            replaced = 1

        await anyio.to_thread.run_sync(target.write_text, new_text)
        return (
            f"edited {path}: replaced {replaced} occurrence(s) "
            f"({len(text)} → {len(new_text)} bytes)"
        )

    return Tool(
        name=name,
        description=(
            f"Modify an existing text file under {str(workdir_path)} "
            "by replacing an exact string. ``old_string`` must match "
            "the file's contents EXACTLY (including whitespace) and "
            "must be unique in the file (unless ``replace_all=True``). "
            "If you don't have enough context for a unique match, "
            "read the file first to grab surrounding lines. Returns "
            "the number of replacements + new byte count on success."
        ),
        fn=_edit,
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path relative to the workdir. File must exist."
                    ),
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "The exact string to replace, including "
                        "any surrounding whitespace needed for "
                        "uniqueness."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": (
                        "The replacement string. Can be empty to "
                        "delete the matched region."
                    ),
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "If true, replace every occurrence of "
                        "old_string. Default false (require "
                        "uniqueness)."
                    ),
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        destructive=True,  # mutates files — flag for permission policies
    )


# ---------------------------------------------------------------------------
# bash_tool
# ---------------------------------------------------------------------------

# Patterns that indicate a clearly-dangerous shell command. The
# default deny list is conservative — users can override entirely
# via ``allow_pattern``.
# Patterns that indicate a clearly-dangerous shell command. The
# ``rm -rf /`` pattern matches only when the trailing slash is
# directly followed by whitespace or end-of-line — so
# ``rm -rf /tmp/foo`` (a perfectly normal cleanup) is NOT flagged.
_DEFAULT_DENY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-r[fr]*\s+/(?:\s|$)"),  # rm -rf / (root only)
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if=.*of=/dev/"),
    re.compile(r":\(\)\{.*\}\s*;\s*:"),  # fork bomb
    re.compile(r">\s*/dev/sd"),  # write to raw disk
    re.compile(r"\bchmod\s+(?:777|-R\s+777)\s+/(?:\s|$)"),  # chmod 777 /
)


def bash_tool(
    workdir: Path | str | None = None,
    *,
    name: str = "bash",
    timeout: float = 30.0,
    allow_pattern: Callable[[str], bool] | None = None,
    extra_env: dict[str, str] | None = None,
) -> Tool:
    """Build a :class:`Tool` that runs a shell command with the
    workdir as the current working directory.

    Default safety:

    * Commands matching the built-in destructive patterns
      (``rm -rf /``, ``sudo``, ``mkfs``, fork bombs, ...) are
      rejected before being executed.
    * Commands run with a default ``timeout`` of 30 seconds; the
      subprocess is killed on timeout.
    * The shell is invoked via ``/bin/sh -c <command>``, so
      pipelines + redirections work the way you'd expect.

    Knobs:

    * ``allow_pattern`` — a callable that takes the command string
      and returns True if the command should run. When provided, it
      OVERRIDES the default deny list — you take full responsibility.
    * ``extra_env`` — extra environment variables merged into the
      subprocess env.
    * ``timeout`` — seconds before the command is killed.

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir (shared with the other built-in tools).
    """
    workdir_path = _resolve_workdir(workdir)

    def _is_allowed(command: str) -> tuple[bool, str | None]:
        if allow_pattern is not None:
            if allow_pattern(command):
                return True, None
            return False, "blocked by user-supplied allow_pattern"
        for pat in _DEFAULT_DENY_PATTERNS:
            if pat.search(command):
                return (
                    False,
                    f"blocked: matches denylist pattern {pat.pattern!r}",
                )
        return True, None

    async def _bash(command: str, timeout_sec: float | None = None) -> str:
        ok, reason = _is_allowed(command)
        if not ok:
            return f"ERROR: {reason}"

        effective_timeout = (
            float(timeout_sec) if timeout_sec is not None else timeout
        )

        # Build env: parent env + any extras the user passed in.
        import os as _os

        env = dict(_os.environ)
        if extra_env:
            env.update(extra_env)

        # Use anyio's subprocess support so we don't block the loop.
        try:
            with anyio.fail_after(effective_timeout):
                process = await anyio.run_process(
                    ["/bin/sh", "-c", command],
                    cwd=str(workdir_path),
                    env=env,
                    check=False,
                )
        except TimeoutError:
            return (
                f"ERROR: command timed out after "
                f"{effective_timeout:.1f}s: {command!r}"
            )

        stdout = (
            process.stdout.decode("utf-8", errors="replace")
            if process.stdout
            else ""
        )
        stderr = (
            process.stderr.decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        rc = process.returncode

        # Truncate huge outputs so the model isn't blown up
        max_len = 10_000
        if len(stdout) > max_len:
            stdout = (
                stdout[:max_len]
                + f"\n... (truncated, {len(stdout) - max_len} more bytes)"
            )
        if len(stderr) > max_len:
            stderr = (
                stderr[:max_len]
                + f"\n... (truncated, {len(stderr) - max_len} more bytes)"
            )

        sections = [f"$ {command}\n[exit={rc}]"]
        if stdout:
            sections.append("--- stdout ---\n" + stdout.rstrip())
        if stderr:
            sections.append("--- stderr ---\n" + stderr.rstrip())
        if not stdout and not stderr:
            sections.append("(no output)")
        return "\n".join(sections)

    return Tool(
        name=name,
        description=(
            f"Run a shell command with cwd={str(workdir_path)}. "
            f"Default timeout {timeout:.0f}s. Returns stdout + "
            "stderr + exit code in a structured block. The default "
            "deny list rejects obviously-dangerous patterns "
            "(rm -rf /, sudo, mkfs, fork bombs, ...). For "
            "long-running commands, pass timeout_sec to override."
        ),
        fn=_bash,
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to run via /bin/sh -c. "
                        "Pipelines and redirections work as expected."
                    ),
                },
                "timeout_sec": {
                    "type": "number",
                    "description": (
                        "Override the default timeout for this "
                        "call (seconds). Optional."
                    ),
                },
            },
            "required": ["command"],
        },
        destructive=True,  # arbitrary command execution
    )


# ---------------------------------------------------------------------------
# Bundle helper
# ---------------------------------------------------------------------------


def filesystem_tools(
    workdir: Path | str | None = None,
) -> list[Tool]:
    """Return all three filesystem tools (read + write + edit)
    bound to a single workdir. ``bash_tool`` is excluded — pair
    them only when you want shell access too.

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir (shared with bash_tool() called the same way)."""
    resolved = _resolve_workdir(workdir)
    return [
        read_tool(resolved),
        write_tool(resolved),
        edit_tool(resolved),
    ]


__all__ = [
    "PathEscapeError",
    "bash_tool",
    "default_workdir",
    "edit_tool",
    "filesystem_tools",
    "read_tool",
    "write_tool",
]


# Re-exports kept here so the module is independently importable
# (some users might prefer ``from loomflow.tools.builtin import
# read_tool``).
_ = shutil  # noqa: F401 — reserved for potential future use (cp helper)
