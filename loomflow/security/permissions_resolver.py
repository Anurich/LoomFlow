"""String / dict resolver for the ``permissions=`` :class:`Agent` kwarg.

Brings permissions up to dict-form parity with model / memory /
runtime / telemetry / audit_log so a TOML / YAML / settings file
can wire the whole agent without imports.

Recognised string forms:

* ``"allow_all"`` — :class:`AllowAll` (the no-op default)
* ``"strict"`` — :class:`StandardPermissions` in default mode
* ``"accept_edits"`` — :class:`StandardPermissions` with
  :attr:`Mode.ACCEPT_EDITS`
* ``"bypass"`` — :class:`StandardPermissions` with
  :attr:`Mode.BYPASS` (allows every call including destructive)

Recognised dict forms:

* ``{"backend": "allow_all"}``
* ``{"backend": "standard", "mode": "default" | "acceptEdits" |
  "bypassPermissions", "allowed_tools": [...], "denied_tools": [...]}``

Already-constructed :class:`Permissions` instances pass through
unchanged. ``None`` returns :class:`AllowAll`.

:class:`PerUserPermissions` isn't reachable from this resolver
because the per-user policy map is more naturally constructed in
Python than declared in a flat config. Build it yourself and pass
the instance through ``permissions=``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.errors import ConfigError
from ..core.protocols import Permissions
from .permissions import AllowAll, Mode, StandardPermissions

__all__ = ["resolve_permissions"]


_MODE_ALIASES: dict[str, Mode] = {
    "default": Mode.DEFAULT,
    "acceptedits": Mode.ACCEPT_EDITS,
    "accept_edits": Mode.ACCEPT_EDITS,
    "accept-edits": Mode.ACCEPT_EDITS,
    "bypasspermissions": Mode.BYPASS,
    "bypass_permissions": Mode.BYPASS,
    "bypass-permissions": Mode.BYPASS,
    "bypass": Mode.BYPASS,
}


def resolve_permissions(spec: Any) -> Permissions:
    if spec is None:
        return AllowAll()
    if isinstance(spec, str):
        return _resolve_string(spec)
    if isinstance(spec, Mapping):
        return _resolve_dict(spec)
    return spec  # type: ignore[no-any-return]


def _resolve_string(spec: str) -> Permissions:
    spec = spec.strip()
    if not spec:
        raise ConfigError(
            "permissions= empty string. Use 'allow_all', 'strict', "
            "'accept_edits', or 'bypass'."
        )
    low = spec.lower()
    if low in ("allow_all", "allowall", "allow-all"):
        return AllowAll()
    if low == "strict":
        return StandardPermissions(mode=Mode.DEFAULT)
    if low in ("accept_edits", "acceptedits", "accept-edits"):
        return StandardPermissions(mode=Mode.ACCEPT_EDITS)
    if low in ("bypass", "bypass_permissions", "bypasspermissions"):
        return StandardPermissions(mode=Mode.BYPASS)
    raise ConfigError(
        f"permissions= unrecognised string spec {spec!r}. Use "
        "'allow_all', 'strict', 'accept_edits', or 'bypass'."
    )


def _resolve_dict(spec: Mapping[str, Any]) -> Permissions:
    backend = (
        spec.get("backend")
        or spec.get("type")
        or spec.get("name")
        or "standard"  # most dict-form configs want StandardPermissions
    )
    if not isinstance(backend, str):
        raise ConfigError(
            "permissions= dict 'backend' (or 'type' / 'name') must be a string."
        )
    low = backend.lower()

    if low in ("allow_all", "allowall", "allow-all"):
        return AllowAll()

    if low == "standard":
        mode_raw = spec.get("mode", "default")
        if not isinstance(mode_raw, str):
            raise ConfigError(
                f"permissions= 'mode' must be a string; got {type(mode_raw).__name__}."
            )
        mode = _MODE_ALIASES.get(mode_raw.lower())
        if mode is None:
            raise ConfigError(
                f"permissions= unrecognised mode {mode_raw!r}. Use "
                "'default', 'accept_edits', or 'bypass'."
            )
        allowed = spec.get("allowed_tools")
        denied = spec.get("denied_tools")
        if allowed is not None and not (
            isinstance(allowed, list)
            and all(isinstance(t, str) for t in allowed)
        ):
            raise ConfigError(
                "permissions= 'allowed_tools' must be a list of strings."
            )
        if denied is not None and not (
            isinstance(denied, list)
            and all(isinstance(t, str) for t in denied)
        ):
            raise ConfigError(
                "permissions= 'denied_tools' must be a list of strings."
            )
        return StandardPermissions(
            mode=mode,
            allowed_tools=list(allowed) if allowed is not None else None,
            denied_tools=list(denied) if denied is not None else None,
        )

    raise ConfigError(
        f"permissions= dict 'backend' = {backend!r} not recognised. "
        "Use 'allow_all' or 'standard'."
    )
