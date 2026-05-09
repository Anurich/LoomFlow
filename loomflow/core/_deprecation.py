"""Deprecation warnings + helpers for protocol-evolution fallbacks.

The framework grew a ``user_id`` keyword argument on every
multi-tenant primitive (Memory, Permissions, HookHost, AuditLog)
between 0.9 and 0.10. Existing code that implemented those
protocols without the kwarg is still supported via
``try / except TypeError`` fallbacks scattered across the loop —
the framework calls the new shape first, and on
``TypeError("got an unexpected keyword argument 'user_id'")``
re-calls the old shape so legacy code keeps working.

Those fallbacks are technical debt: they hide the migration the
caller still needs to do, and they paper over real bugs (a
genuine ``TypeError`` from a different mismatch can be
indistinguishable from "the impl is missing the kwarg"). M10.5
adds a small infrastructure so each fallback emits a
:class:`LoomDeprecationWarning` once per process per site,
naming the protocol method and the v1.0 removal target.

Two pieces:

* :class:`LoomDeprecationWarning` — a ``DeprecationWarning``
  subclass so callers can filter framework deprecations
  separately from Python-stdlib ones.
* :func:`warn_legacy_protocol` — call this from inside a
  ``except TypeError`` block to emit a one-time-per-site warning
  pointing at the legacy shape. Idempotent: a single site only
  emits once per process to keep the log clean.
"""

from __future__ import annotations

import warnings
from typing import Final

__all__ = [
    "LoomDeprecationWarning",
    "warn_legacy_protocol",
]


class LoomDeprecationWarning(DeprecationWarning):
    """Marker class for Loom deprecations.

    Inherits from :class:`DeprecationWarning` so the stdlib's
    default-off display still applies; tests opt in via
    :func:`warnings.simplefilter` or
    ``pytest.warns(LoomDeprecationWarning)``.
    """


_REMOVAL_TARGET: Final[str] = "1.0"

# Per-site idempotency: ``(protocol, method)`` pairs we've
# already warned about in this process. We dedupe so the first
# request emits the warning and subsequent requests don't spam.
_SEEN: set[tuple[str, str]] = set()


def warn_legacy_protocol(
    protocol: str,
    method: str,
    *,
    missing_kwarg: str = "user_id",
    removal: str = _REMOVAL_TARGET,
    stacklevel: int = 3,
) -> None:
    """Emit a one-time deprecation warning for a protocol-evolution
    fallback path.

    Call from inside the ``except TypeError`` block that handles a
    legacy impl missing a newer keyword argument::

        try:
            await deps.hooks.pre_tool(call, user_id=run_user_id)
        except TypeError:
            warn_legacy_protocol("HookHost", "pre_tool")
            await deps.hooks.pre_tool(call)

    Idempotent across the process: a given (protocol, method)
    pair only warns once. ``stacklevel`` defaults to 3 so the
    warning points at the caller of the framework method, not the
    framework itself.
    """
    key = (protocol, method)
    if key in _SEEN:
        return
    _SEEN.add(key)
    warnings.warn(
        f"{protocol}.{method}() implementation is missing the "
        f"`{missing_kwarg}=` keyword argument; the framework is "
        f"falling back to the legacy call shape. This shim will be "
        f"removed in loomflow {removal}. Update your "
        f"{protocol} implementation to accept "
        f"`{missing_kwarg}: str | None = None` as a keyword-only "
        f"argument.",
        LoomDeprecationWarning,
        stacklevel=stacklevel,
    )


def _reset_seen_for_tests() -> None:
    """Clear the per-process dedupe set. Test-only."""
    _SEEN.clear()
