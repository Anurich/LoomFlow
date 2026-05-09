"""M10.5 — protocol-evolution fallbacks emit DeprecationWarning.

The framework added ``user_id`` keyword arguments to every
multi-tenant primitive between 0.9 and 0.10. Existing custom
implementations without that kwarg are still supported via
``try / except TypeError`` shims. Those shims now emit a
:class:`LoomDeprecationWarning` so callers can migrate their
custom code before v1.0 removes the fallback entirely.

Tests assert:

* The warning fires once per (protocol, method) per process.
* The warning fires from the legitimate fallback path — i.e.
  using a real legacy implementation, not a synthetic raise.
* Non-fallback paths emit nothing.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from loomflow.core._deprecation import (
    LoomDeprecationWarning,
    _reset_seen_for_tests,
    warn_legacy_protocol,
)
from loomflow.core.types import PermissionDecision, ToolCall
from loomflow.security.permissions import PerUserPermissions

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helper class (a permissions impl missing the user_id kwarg)
# ---------------------------------------------------------------------------


class _LegacyPermissions:
    """A Permissions impl that pre-dates the ``user_id`` kwarg.

    Its ``check`` signature lacks ``user_id``, so calling it with
    that kwarg raises ``TypeError`` — which is exactly the
    fallback signal the framework's shim layer detects."""

    async def check(
        self,
        call: ToolCall,
        *,
        context: dict[str, Any],
    ) -> PermissionDecision:
        return PermissionDecision.allow_()


class _ModernPermissions:
    """The current shape — accepts ``user_id`` and returns allow."""

    async def check(
        self,
        call: ToolCall,
        *,
        context: dict[str, Any],
        user_id: str | None = None,
    ) -> PermissionDecision:
        return PermissionDecision.allow_()


# ---------------------------------------------------------------------------
# Direct unit tests on the helper
# ---------------------------------------------------------------------------


def test_warn_legacy_protocol_fires_once_per_site() -> None:
    _reset_seen_for_tests()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LoomDeprecationWarning)
        warn_legacy_protocol("Memory", "remember")
        warn_legacy_protocol("Memory", "remember")
        warn_legacy_protocol("Memory", "remember")
    matching = [
        w for w in caught
        if issubclass(w.category, LoomDeprecationWarning)
    ]
    assert len(matching) == 1
    assert "Memory.remember" in str(matching[0].message)
    assert "1.0" in str(matching[0].message)


def test_warn_legacy_protocol_distinct_sites_each_fire() -> None:
    """Different (protocol, method) keys are independent — each
    site warns once."""
    _reset_seen_for_tests()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LoomDeprecationWarning)
        warn_legacy_protocol("Memory", "remember")
        warn_legacy_protocol("Memory", "recall")
        warn_legacy_protocol("HookHost", "pre_tool")
    assert len(caught) == 3


# ---------------------------------------------------------------------------
# Integration: real fallback path inside PerUserPermissions
# ---------------------------------------------------------------------------


async def test_per_user_permissions_warns_on_legacy_inner() -> None:
    """``PerUserPermissions`` wraps possibly-legacy inner policies.
    A legacy inner triggers the TypeError fallback, which must
    emit the deprecation warning."""
    _reset_seen_for_tests()
    perms = PerUserPermissions(
        policies={"alice": _LegacyPermissions()},
        default=_ModernPermissions(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LoomDeprecationWarning)
        await perms.check(
            ToolCall(tool="anything"),
            context={},
            user_id="alice",
        )
    matching = [
        w for w in caught
        if issubclass(w.category, LoomDeprecationWarning)
    ]
    assert len(matching) == 1
    assert "Permissions.check" in str(matching[0].message)


async def test_modern_permissions_emits_no_warning() -> None:
    """An impl that already accepts ``user_id`` shouldn't trigger
    the fallback — and so shouldn't emit the warning."""
    _reset_seen_for_tests()
    perms = PerUserPermissions(
        policies={"alice": _ModernPermissions()},
        default=_ModernPermissions(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LoomDeprecationWarning)
        await perms.check(
            ToolCall(tool="anything"),
            context={},
            user_id="alice",
        )
    matching = [
        w for w in caught
        if issubclass(w.category, LoomDeprecationWarning)
    ]
    assert matching == []
