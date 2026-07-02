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
allowed root.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any

from ...core.protocols import ToolHost
from ...core.types import ToolDef, ToolEvent, ToolResult

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


class FilesystemSandbox:
    """Restrict a tool host's path-typed arguments to declared roots."""

    def __init__(
        self,
        inner: ToolHost,
        *,
        roots: Iterable[str | Path],
        path_args: Iterable[str] | None = None,
        auto_detect: bool = True,
    ) -> None:
        roots_list = [Path(r).resolve() for r in roots]
        if not roots_list:
            raise ValueError(
                "FilesystemSandbox requires at least one allowed root"
            )
        self._inner = inner
        self._roots: tuple[Path, ...] = tuple(roots_list)
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
            return None
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
        if isinstance(value, str) and ("/" in value or "\\" in value):
            return True
        return False

    def _is_inside_any_root(self, raw_path: str) -> bool:
        try:
            resolved = Path(raw_path).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        for root in self._roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False
