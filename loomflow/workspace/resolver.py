"""String / dict resolver for the ``workspace=`` :class:`Agent` /
:class:`Workflow` kwarg.

Recognised string forms:

* ``"temp"`` — a fresh :meth:`LocalDiskWorkspace.temp` workspace.
  Auto-cleaned when the workspace's ``aclose()`` fires.
* ``"temp:<prefix>"`` — temp with a custom dirname prefix; same
  cleanup behaviour. Handy when you want to find your workspace
  by ``ls $TMPDIR | grep <prefix>``.
* ``"memory"`` / ``"inmemory"`` — :class:`InMemoryWorkspace`
  (zero-dep, ephemeral; no filesystem).
* ``"<path>"`` — :meth:`LocalDiskWorkspace.open` at the given
  filesystem path. The directory is created if missing and
  persists past the run (never auto-cleaned).

Recognised dict forms:

* ``{"backend": "disk", "path": "...", "seed": [...]}`` —
  disk workspace at an explicit path, optionally pre-seeded with
  reference docs copied into ``seeds/``.
* ``{"backend": "temp", "prefix": "...", "seed": [...]}`` —
  temp workspace with prefix + seeds.
* ``{"backend": "memory"}`` — in-memory.

``None`` returns ``None`` (no workspace — agents work without a
shared notebook). Workspace instances pass through unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.errors import ConfigError
from .disk import LocalDiskWorkspace
from .inmemory import InMemoryWorkspace
from .protocol import Workspace
from .types import WorkspaceMembership

__all__ = ["resolve_workspace"]


def resolve_workspace(
    spec: Any,
) -> Workspace | WorkspaceMembership | None:
    """Coerce a workspace spec into a :class:`Workspace` or
    :class:`WorkspaceMembership`.

    * ``None`` → ``None`` (no workspace wired)
    * :class:`WorkspaceMembership` → passthrough
    * :class:`Workspace` instance → passthrough
    * ``str`` → routed through :func:`_resolve_string`
    * ``Mapping`` → routed through :func:`_resolve_dict`. When the
      dict contains ``author`` and/or ``teammates`` keys, the
      resolver returns a :class:`WorkspaceMembership` wrapping the
      built workspace + the identity. Otherwise just the workspace.

    :class:`Agent` accepts both shapes — :class:`WorkspaceMembership`
    is the path for "join this notebook as <name>"; raw
    :class:`Workspace` is the simpler "share this notebook with
    no specific identity" path.
    """
    if spec is None:
        return None
    if isinstance(spec, WorkspaceMembership):
        return spec
    if isinstance(spec, str):
        return _resolve_string(spec)
    if isinstance(spec, Mapping):
        return _resolve_dict_with_identity(spec)
    return spec  # type: ignore[no-any-return]


def _resolve_dict_with_identity(
    spec: Mapping[str, Any],
) -> Workspace | WorkspaceMembership:
    """Split off ``author`` / ``teammates`` keys from the dict and
    wrap the resulting workspace in a :class:`WorkspaceMembership`
    when either is present. Otherwise the dict resolves to a bare
    :class:`Workspace`."""
    author = spec.get("author")
    teammates = spec.get("teammates")

    # Strip identity keys so the rest is a clean backend dict.
    backend_dict = {
        k: v for k, v in spec.items()
        if k not in ("author", "teammates")
    }
    # Special case: caller wants to wrap an EXISTING instance with
    # identity metadata: ``{"backend": ws_instance, "author": ...}``.
    backend_value = backend_dict.get("backend")
    if isinstance(backend_value, Workspace):
        workspace: Workspace = backend_value
    elif not backend_dict:
        raise ConfigError(
            "workspace= dict needs a 'backend' (or a 'workspace' "
            "instance under 'backend') in addition to 'author' / "
            "'teammates'."
        )
    else:
        workspace = _resolve_dict(backend_dict)

    if author is None and teammates is None:
        return workspace
    return WorkspaceMembership(
        workspace=workspace,
        name=str(author) if author is not None else None,
        teammates=(
            [str(t) for t in teammates] if teammates else None
        ),
    )


def _resolve_string(spec: str) -> Workspace:
    spec = spec.strip()
    if not spec:
        raise ConfigError(
            "workspace= empty string. Use 'temp', 'memory', or a "
            "filesystem path like '/tmp/my-workspace'."
        )
    if spec in ("memory", "inmemory"):
        return InMemoryWorkspace()
    if spec == "temp":
        return LocalDiskWorkspace.temp()
    if spec.startswith("temp:"):
        prefix = spec[len("temp:"):]
        if not prefix:
            raise ConfigError(
                "workspace= 'temp:' needs a prefix: 'temp:my-research'."
            )
        return LocalDiskWorkspace.temp(prefix=f"{prefix}-")
    # Anything else: treat as a filesystem path.
    return LocalDiskWorkspace.open(spec)


def _resolve_dict(spec: Mapping[str, Any]) -> Workspace:
    backend = spec.get("backend") or spec.get("type") or spec.get("name")
    if not isinstance(backend, str):
        raise ConfigError(
            "workspace= dict must include 'backend' (or 'type' / 'name'). "
            "Recognised values: 'disk', 'temp', 'memory'."
        )
    low = backend.lower()

    if low in ("memory", "inmemory"):
        return InMemoryWorkspace()

    if low == "temp":
        prefix = spec.get("prefix")
        seed = _coerce_seed(spec.get("seed"))
        kwargs: dict[str, Any] = {}
        if isinstance(prefix, str) and prefix:
            kwargs["prefix"] = f"{prefix}-"
        if seed:
            kwargs["seed_paths"] = seed
        cleanup = spec.get("cleanup")
        if cleanup is not None:
            kwargs["cleanup"] = bool(cleanup)
        return LocalDiskWorkspace.temp(**kwargs)

    if low in ("disk", "local", "filesystem", "fs"):
        path = spec.get("path") or spec.get("url")
        if not isinstance(path, str) or not path:
            raise ConfigError(
                "workspace= disk backend requires a string 'path'."
            )
        seed = _coerce_seed(spec.get("seed"))
        return LocalDiskWorkspace.open(path, seed_paths=seed)

    raise ConfigError(
        f"workspace= dict 'backend' = {backend!r} not recognised. "
        "Use 'disk', 'temp', or 'memory'."
    )


def _coerce_seed(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ConfigError(
        "workspace= 'seed' must be a string path or a list of string paths."
    )
