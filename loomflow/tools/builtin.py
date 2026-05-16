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
    * The shell is invoked via ``/bin/sh -c <command>`` on POSIX
      (macOS, Linux) and ``cmd.exe /c <command>`` on Windows. The
      command STRING is what the model emits — it must use the
      shell's native syntax for whichever host runs it. ``ls``
      and ``rm`` won't work in cmd.exe; ``dir`` and ``del`` won't
      work in sh. Surface the host platform to the model in the
      system prompt if you need cross-host commands.

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

        # Pick the host shell. ``/bin/sh`` doesn't exist on Windows
        # — invoking it there raises ``FileNotFoundError`` with the
        # confusing "[WinError 2] system cannot find the file
        # specified" message. ``cmd.exe`` is the cross-version
        # Windows fallback (PowerShell would also work but quoting
        # rules diverge more from sh, so cmd is the safer default).
        import sys as _sys
        if _sys.platform == "win32":
            argv = ["cmd.exe", "/c", command]
        else:
            argv = ["/bin/sh", "-c", command]

        # Use anyio's subprocess support so we don't block the loop.
        try:
            with anyio.fail_after(effective_timeout):
                process = await anyio.run_process(
                    argv,
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

    import sys as _sys_for_desc
    host_shell = "cmd.exe" if _sys_for_desc.platform == "win32" else "/bin/sh"
    return Tool(
        name=name,
        description=(
            f"Run a shell command with cwd={str(workdir_path)}. "
            f"Host shell: {host_shell} (use its native syntax — "
            f"sh on macOS/Linux, cmd on Windows). "
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
                        f"Shell command to run via {host_shell} "
                        "(use the host shell's native syntax). "
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
# grep_tool — search file contents
# ---------------------------------------------------------------------------


# Directories never worth walking — keeps grep / find / ls fast and
# their output relevant in real codebases.
_NOISE_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".loom", ".idea", ".vscode", ".tox",
})


def grep_tool(
    workdir: Path | str | None = None,
    *,
    name: str = "grep",
    max_results: int = 200,
) -> Tool:
    """Build a :class:`Tool` that searches file contents for a
    regex under ``workdir`` — the read-only "find me where X is"
    tool every coding agent needs.

    The tool's signature seen by the model:
        ``grep(pattern: str, path: str = ".", glob: str = "*",
               ignore_case: bool = False)``

    Returns matching lines prefixed ``relpath:lineno: content``,
    capped at ``max_results``. Skips noise dirs (``.git``,
    ``node_modules``, ``__pycache__``, ...) so output stays
    relevant in real repos. Non-text files are skipped silently.

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir.
    """
    workdir_path = _resolve_workdir(workdir)

    async def _grep(
        pattern: str,
        path: str = ".",
        glob: str = "*",
        ignore_case: bool = False,
    ) -> str:
        try:
            root = _resolve_within(workdir_path, path)
        except PathEscapeError as exc:
            return f"ERROR: {exc}"
        if not root.exists():
            return f"ERROR: path not found: {path}"
        try:
            flags = re.IGNORECASE if ignore_case else 0
            rx = re.compile(pattern, flags)
        except re.error as exc:
            return f"ERROR: invalid regex {pattern!r}: {exc}"

        def _search() -> list[str]:
            hits: list[str] = []
            candidates = (
                [root] if root.is_file() else sorted(root.rglob(glob))
            )
            for fp in candidates:
                if not fp.is_file():
                    continue
                if any(part in _NOISE_DIRS for part in fp.parts):
                    continue
                try:
                    text = fp.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue  # binary / unreadable — skip
                rel = fp.relative_to(workdir_path)
                for i, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{rel}:{i}: {line.rstrip()}")
                        if len(hits) >= max_results:
                            return hits
            return hits

        hits = await anyio.to_thread.run_sync(_search)
        if not hits:
            return f"No matches for {pattern!r} under {path}"
        out = "\n".join(hits)
        if len(hits) >= max_results:
            out += f"\n... (capped at {max_results} matches)"
        return out

    return Tool(
        name=name,
        description=(
            f"Search file contents for a regex under {str(workdir_path)}. "
            "Returns matching lines as 'relpath:lineno: content'. "
            "Skips .git / node_modules / __pycache__ and other noise "
            "dirs. Use this to locate where a symbol / string / "
            "pattern lives before reading or editing."
        ),
        fn=_grep,
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regex to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory (or single file) to search "
                        "under, relative to the workdir. Default '.'."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Filename glob to restrict the search "
                        "(e.g. '*.py'). Default '*' (all files)."
                    ),
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive match. Default false.",
                },
            },
            "required": ["pattern"],
        },
    )


# ---------------------------------------------------------------------------
# find_tool — locate files by name
# ---------------------------------------------------------------------------


def find_tool(
    workdir: Path | str | None = None,
    *,
    name: str = "find",
    max_results: int = 300,
) -> Tool:
    """Build a :class:`Tool` that finds files by name-glob under
    ``workdir``.

    The tool's signature seen by the model:
        ``find(glob: str, path: str = ".")``

    Returns matching paths (relative to the workdir), one per
    line, capped at ``max_results``. Skips noise dirs. Use this
    to answer "where is the file called X" / "what test files
    exist" without shelling out.

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir.
    """
    workdir_path = _resolve_workdir(workdir)

    async def _find(glob: str, path: str = ".") -> str:
        try:
            root = _resolve_within(workdir_path, path)
        except PathEscapeError as exc:
            return f"ERROR: {exc}"
        if not root.exists():
            return f"ERROR: path not found: {path}"

        def _walk() -> list[str]:
            out: list[str] = []
            for fp in sorted(root.rglob(glob)):
                if any(part in _NOISE_DIRS for part in fp.parts):
                    continue
                out.append(str(fp.relative_to(workdir_path)))
                if len(out) >= max_results:
                    break
            return out

        matches = await anyio.to_thread.run_sync(_walk)
        if not matches:
            return f"No files matching {glob!r} under {path}"
        result = "\n".join(matches)
        if len(matches) >= max_results:
            result += f"\n... (capped at {max_results})"
        return result

    return Tool(
        name=name,
        description=(
            f"Find files by name-glob under {str(workdir_path)}. "
            "Returns matching relative paths, one per line. Skips "
            ".git / node_modules / build dirs. Example globs: "
            "'*.py', 'test_*.py', '**/config.*'."
        ),
        fn=_find,
        input_schema={
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "description": (
                        "Filename glob, e.g. '*.py' or "
                        "'test_*.py'. Recursive by default."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to search under, relative to "
                        "the workdir. Default '.'."
                    ),
                },
            },
            "required": ["glob"],
        },
    )


# ---------------------------------------------------------------------------
# ls_tool — list a directory
# ---------------------------------------------------------------------------


def ls_tool(
    workdir: Path | str | None = None,
    *,
    name: str = "ls",
) -> Tool:
    """Build a :class:`Tool` that lists a directory's entries
    under ``workdir``.

    The tool's signature seen by the model:
        ``ls(path: str = ".")``

    Returns one entry per line, directories suffixed with ``/``,
    sorted dirs-first then files. Noise dirs are listed but not
    recursed (this is a single-level listing). Use this to orient
    in an unfamiliar part of the tree.

    ``workdir`` is optional; ``None`` uses the framework's default
    tempdir.
    """
    workdir_path = _resolve_workdir(workdir)

    async def _ls(path: str = ".") -> str:
        try:
            target = _resolve_within(workdir_path, path)
        except PathEscapeError as exc:
            return f"ERROR: {exc}"
        if not target.exists():
            return f"ERROR: path not found: {path}"
        if not target.is_dir():
            return f"ERROR: not a directory: {path}"

        def _list() -> str:
            entries = sorted(
                target.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
            lines: list[str] = []
            for e in entries:
                if e.is_dir():
                    lines.append(f"{e.name}/")
                else:
                    try:
                        size = e.stat().st_size
                        lines.append(f"{e.name}  ({size}B)")
                    except OSError:
                        lines.append(e.name)
            return "\n".join(lines) or "(empty directory)"

        return await anyio.to_thread.run_sync(_list)

    return Tool(
        name=name,
        description=(
            f"List a directory's entries under {str(workdir_path)}. "
            "Single-level (not recursive — use find for that). "
            "Directories are suffixed '/', files show their byte "
            "size. Sorted dirs-first."
        ),
        fn=_ls,
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to list, relative to the "
                        "workdir. Default '.'."
                    ),
                },
            },
            "required": [],
        },
    )


# ---------------------------------------------------------------------------
# Bundle helper
# ---------------------------------------------------------------------------


def filesystem_tools(
    workdir: Path | str | None = None,
) -> list[Tool]:
    """Return the read-only + mutating filesystem tools bound to a
    single workdir: ``read`` + ``write`` + ``edit`` + ``grep`` +
    ``find`` + ``ls``. ``bash_tool`` is excluded — pair it in only
    when you want shell access too.

    This is the "Claude-Code-shaped" / Pi-kernel tool set minus
    bash. ``workdir`` is optional; ``None`` uses the framework's
    default tempdir (shared with bash_tool() called the same way).
    """
    resolved = _resolve_workdir(workdir)
    return [
        read_tool(resolved),
        write_tool(resolved),
        edit_tool(resolved),
        grep_tool(resolved),
        find_tool(resolved),
        ls_tool(resolved),
    ]


__all__ = [
    "PathEscapeError",
    "bash_tool",
    "default_workdir",
    "edit_tool",
    "filesystem_tools",
    "find_tool",
    "grep_tool",
    "ls_tool",
    "read_tool",
    "write_tool",
]


# Re-exports kept here so the module is independently importable
# (some users might prefer ``from loomflow.tools.builtin import
# read_tool``).
_ = shutil  # noqa: F401 — reserved for potential future use (cp helper)
