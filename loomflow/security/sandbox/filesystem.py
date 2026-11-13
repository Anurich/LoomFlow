"""Path-aware sandbox.

Wraps a :class:`ToolHost` and rejects tool calls whose path-typed
arguments resolve outside a configured set of allowed roots. Detection
is configurable:

* Pass ``path_args=("path", "destination", ...)`` to validate exactly
  those argument names.
* Otherwise the sandbox auto-detects: any string argument whose name
  is in :data:`DEFAULT_PATH_ARG_NAMES`, or ends with a common
  path-ish suffix (``...path`` / ``...paths`` / ``...file`` /
  ``...dir`` etc.), *or* whose value contains a path separator
  (``/`` or ``\\``) is treated as a path.

Container arguments (lists / tuples / sets / dicts) are recursed
into and their string leaves validated with the same rules —
``{"paths": ["/etc/passwd"]}`` and nested dicts are checked, not
skipped. Leaves inherit the pathiness of the nearest enclosing
path-like name, so every string under a path-named argument is
validated regardless of separator presence.

Symlinks are resolved before the containment check so an attacker
can't bypass the sandbox by symlinking ``/etc/passwd`` into the
allowed root. ``file://`` URLs and ``~`` paths are checked as the
paths they name, relative paths resolve against ``base`` (default:
the first root, which is how workdir-based tools resolve them, not
the process's working directory), and anything the check can't
decide — a null byte, arguments nested deeper than it scans —
is refused.

**This is an argument check, not containment.** It looks at the
path-shaped *arguments* of a call; it has no say over what the tool
then does. For file tools (read / write / edit) the argument IS the
access, so the check is the boundary. For a shell or code-execution
tool it is not: ``bash(command="cat /etc/passwd")`` has no path
argument to refuse, and the command can read, write and delete
anywhere the process can. Wrapping such a tool therefore emits a
:class:`UserWarning` on its first call (``shell_tools="warn"``, the
default); pass ``shell_tools="deny"`` to refuse those calls, or
``"allow"`` to silence the warning when the shell is contained some
other way. To actually contain commands use
:class:`~loomflow.security.sandbox.OSSandbox` or run them through a
container executor (``bash_tool(executor=...)``).
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from ...core.protocols import ToolHost
from ...core.types import ToolDef, ToolEvent, ToolResult
from .base import register_inside

DEFAULT_PATH_ARG_NAMES: frozenset[str] = frozenset(
    {
        "path",
        "file",
        "filename",
        "filepath",
        "directory",
        "dir",
        "folder",
        "src",
        "source",
        "dst",
        "destination",
        "target",
    }
)

# Auto-detect also treats argument / nested-key names that ARE one
# of these, or end in one after a ``_`` / ``-`` / ``.`` separator
# ("paths", "output_path", "src-files", ...), as path-like, so
# their string leaves are validated even without a separator in
# the value. Delimited matching avoids false positives like
# ``resource`` (ends with "source" but isn't a path).
_PATH_NAME_SUFFIXES: tuple[str, ...] = (
    "path",
    "paths",
    "file",
    "files",
    "filename",
    "filenames",
    "dir",
    "dirs",
    "directory",
    "directories",
    "folder",
    "folders",
    "source",
    "destination",
    "target",
    "targets",
)

# Containers nest at most this deep before the scan gives up —
# tool args come from JSON and never legitimately nest this far;
# the bound just guards against pathological recursion.
_MAX_ARG_DEPTH = 16


# Tool names that mean "runs whatever text it is given" — plus any
# tool carrying the ``__loomflow_shell__`` marker (the built-in
# ``bash_tool`` does, whatever it was renamed to) or taking a
# ``command`` / ``cmd`` / ``script`` string argument.
SHELL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "bash",
        "sh",
        "shell",
        "zsh",
        "powershell",
        "terminal",
        "exec",
        "execute",
        "execute_command",
        "run_command",
        "run_shell",
        "shell_exec",
    }
)
_SHELL_ARG_NAMES: frozenset[str] = frozenset({"command", "cmd", "script"})

ShellToolPolicy = Literal["warn", "deny", "allow"]


class FilesystemSandbox:
    """Restrict a tool host's path-typed arguments to declared roots.

    Only arguments are checked. That contains file tools; it does
    NOT contain a shell / code-execution tool, whose command can
    touch any path without naming it in an argument the check
    understands. ``shell_tools`` says what to do when such a tool is
    called through the sandbox: ``"warn"`` (default) lets the call
    through with a one-time :class:`UserWarning`, ``"deny"`` refuses
    it, ``"allow"`` lets it through silently.
    """

    def __init__(
        self,
        inner: ToolHost,
        *,
        roots: Iterable[str | Path],
        path_args: Iterable[str] | None = None,
        auto_detect: bool = True,
        base: str | Path | None = None,
        shell_tools: ShellToolPolicy = "warn",
    ) -> None:
        if shell_tools not in ("warn", "deny", "allow"):
            raise ValueError(
                "shell_tools must be 'warn', 'deny' or 'allow', "
                f"got {shell_tools!r}"
            )
        self._shell_tools: ShellToolPolicy = shell_tools
        self._warned_shell_tools: set[str] = set()
        roots_list = [Path(r).resolve() for r in roots]
        if not roots_list:
            raise ValueError(
                "FilesystemSandbox requires at least one allowed root"
            )
        self._inner = inner
        self._roots: tuple[Path, ...] = tuple(roots_list)
        self._base = Path(base).resolve() if base is not None else roots_list[0]
        self._explicit_path_args: frozenset[str] | None = (
            frozenset(path_args) if path_args is not None else None
        )
        self._auto_detect = auto_detect

    # ---- introspection --------------------------------------------------

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    @property
    def inner(self) -> ToolHost:
        return self._inner

    def register(self, tool: Any) -> None:
        """Add ``tool`` under this sandbox (see :func:`register_inside`)."""
        register_inside(self, tool)

    # ---- ToolHost protocol ----------------------------------------------

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        return await self._inner.list_tools(query=query)

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        if self._shell_tools != "allow" and self._is_shell_tool(tool, args):
            if self._shell_tools == "deny":
                return ToolResult.denied_(
                    call_id,
                    f"FilesystemSandbox: {tool!r} runs arbitrary commands, "
                    "which a path-argument check cannot contain "
                    "(shell_tools='deny'). Use OSSandbox or a container "
                    "executor for shell tools.",
                )
            if tool not in self._warned_shell_tools:
                self._warned_shell_tools.add(tool)
                warnings.warn(
                    f"FilesystemSandbox does not contain the shell tool "
                    f"{tool!r}: it only checks path-like ARGUMENTS, and a "
                    "command can read or write any path the process can "
                    "(`cat /etc/passwd` passes). Use OSSandbox or "
                    "bash_tool(executor=<container executor>) to contain "
                    "commands; pass shell_tools='deny' to refuse shell "
                    "tools here, or shell_tools='allow' to silence this.",
                    UserWarning,
                    stacklevel=2,
                )
        violation = self._first_violation(args)
        if violation is not None:
            arg_name, arg_value = violation
            return ToolResult.denied_(
                call_id,
                f"FilesystemSandbox: argument {arg_name!r} resolves outside "
                f"the allowed roots ({arg_value!r})",
            )
        return await self._inner.call(tool, args, call_id=call_id)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for event in self._inner.watch():
            yield event

    # ---- shell-tool detection -------------------------------------------

    def _is_shell_tool(self, tool: str, args: Mapping[str, Any]) -> bool:
        """Best-effort: does ``tool`` execute free-form commands?"""
        if tool.lower() in SHELL_TOOL_NAMES:
            return True
        if any(
            isinstance(args.get(name), str) for name in _SHELL_ARG_NAMES
        ):
            return True
        # The built-in ``bash_tool`` marks its function, so a renamed
        # one is still recognised when the inner host can hand the
        # Tool back (``InProcessToolHost.get``).
        host: Any = self._inner
        for _ in range(8):  # walk nested sandboxes down to the host
            getter = getattr(host, "get", None)
            if callable(getter):
                try:
                    registered = getter(tool)
                except Exception:  # noqa: BLE001 — detection is best-effort
                    registered = None
                fn = getattr(registered, "fn", None)
                if getattr(fn, "__loomflow_shell__", False):
                    return True
            host = getattr(host, "inner", None) or getattr(host, "_inner", None)
            if host is None:
                break
        return False

    # ---- validation -----------------------------------------------------

    def _first_violation(
        self, args: Mapping[str, Any]
    ) -> tuple[str, str] | None:
        for name, value in args.items():
            leaf = self._scan_value(name, value, 0)
            if leaf is not None:
                return name, leaf
        return None

    def _scan_value(self, name: str, value: Any, depth: int) -> str | None:
        """Recursively scan ``value`` (str leaves, lists / tuples /
        sets / dicts) for the first path outside the roots. ``name``
        is the nearest enclosing argument / dict-key name; container
        items inherit it, so ``{"paths": ["/etc/passwd"]}`` is
        checked with the pathiness of ``paths``. Returns the
        offending leaf value, or ``None``."""
        if depth > _MAX_ARG_DEPTH:
            # Deeper than we scan: refuse rather than let it through.
            return f"<arguments nested deeper than {_MAX_ARG_DEPTH} levels>"
        if isinstance(value, str):
            if self._looks_like_path(name, value) and not (
                self._is_inside_any_root(value)
            ):
                return value
            return None
        if isinstance(value, Mapping):
            pathy = self._name_is_path_like(name)
            for key, item in value.items():
                # Nested keys re-key the pathiness check — unless
                # the enclosing name was already path-like, which
                # stays sticky so `{"paths": {"a": ...}}` keeps
                # validating every leaf beneath it.
                child = (
                    name if pathy or not isinstance(key, str) else key
                )
                leaf = self._scan_value(child, item, depth + 1)
                if leaf is not None:
                    return leaf
            return None
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                leaf = self._scan_value(name, item, depth + 1)
                if leaf is not None:
                    return leaf
        return None

    def _name_is_path_like(self, name: str) -> bool:
        """Whether ``name`` alone marks its values as paths (the
        explicit allowlist, or — in auto-detect — the known set plus
        common path-ish suffixes)."""
        if self._explicit_path_args is not None:
            return name in self._explicit_path_args
        if not self._auto_detect:
            return False
        lowered = name.lower()
        if lowered in DEFAULT_PATH_ARG_NAMES:
            return True
        return any(
            lowered == suffix
            or lowered.endswith((f"_{suffix}", f"-{suffix}", f".{suffix}"))
            for suffix in _PATH_NAME_SUFFIXES
        )

    def _looks_like_path(self, name: str, value: Any) -> bool:
        if self._name_is_path_like(name):
            return True
        if self._explicit_path_args is not None or not self._auto_detect:
            return False
        if isinstance(value, str) and (
            "/" in value or "\\" in value or value.startswith(("~", "file:"))
        ):
            return True
        return False

    def _is_inside_any_root(self, raw_path: str) -> bool:
        if raw_path.startswith("file:"):
            from urllib.parse import unquote, urlparse

            raw_path = unquote(urlparse(raw_path).path)
        try:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = self._base / path
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):  # ValueError: a null byte
            return False
        for root in self._roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False
