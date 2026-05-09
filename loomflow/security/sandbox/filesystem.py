"""Path-aware sandbox.

Wraps a :class:`ToolHost` and rejects tool calls whose path-typed
arguments resolve outside a configured set of allowed roots. Detection
is configurable:

* Pass ``path_args=("path", "destination", ...)`` to validate exactly
  those argument names.
* Otherwise the sandbox auto-detects: any string argument whose name
  is in :data:`DEFAULT_PATH_ARG_NAMES` *or* whose value contains a
  path separator (``/`` or ``\\``) is treated as a path.

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
            if not self._looks_like_path(name, value):
                continue
            if not isinstance(value, str):
                continue
            if not self._is_inside_any_root(value):
                return name, value
        return None

    def _looks_like_path(self, name: str, value: Any) -> bool:
        if self._explicit_path_args is not None:
            return name in self._explicit_path_args
        if not self._auto_detect:
            return False
        if name.lower() in DEFAULT_PATH_ARG_NAMES:
            return True
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
