"""Resolve architecture specs (string or instance) to a concrete
:class:`Architecture` instance.

Built-in string aliases live in :data:`KNOWN`. Third-party packages
can also register custom architectures via a setuptools / Poetry
entry point::

    [project.entry-points."loomflow.architecture"]
    my-strategy = "my_pkg.module:MyArchitecture"

Discovery is cached on first lookup; call :func:`clear_arch_cache`
in tests that need to re-discover after installing a fake entry
point. The resolver also accepts any object satisfying the
:class:`Architecture` protocol — most user-facing flows just pass
an instance like ``ReAct(max_turns=20)`` directly.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from ..core.errors import ConfigError
from .base import Architecture
from .plan_and_execute import PlanAndExecute
from .react import ReAct
from .reflexion import Reflexion
from .rewoo import ReWOO
from .self_refine import SelfRefine
from .tree_of_thoughts import TreeOfThoughts

KNOWN: dict[str, type[Architecture]] = {
    "plan-and-execute": PlanAndExecute,
    "react": ReAct,
    "reflexion": Reflexion,
    "rewoo": ReWOO,
    "self-refine": SelfRefine,
    "tree-of-thoughts": TreeOfThoughts,
}

ENTRY_POINT_GROUP = "loomflow.architecture"

_discovered_cache: dict[str, type[Architecture]] | None = None


def _discover_entry_points() -> dict[str, type[Architecture]]:
    """Load and cache every ``loomflow.architecture`` entry point.

    Built-in names in :data:`KNOWN` win over entry-point names with
    the same key so third-party packages can't shadow the framework's
    defaults by accident.
    """
    global _discovered_cache
    if _discovered_cache is not None:
        return _discovered_cache
    discovered: dict[str, type[Architecture]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in KNOWN:
            continue  # built-in wins; don't let third parties shadow
        try:
            obj = ep.load()
        except Exception:  # noqa: BLE001 — broken plugins shouldn't kill resolve
            continue
        if isinstance(obj, type):
            discovered[ep.name] = obj
    _discovered_cache = discovered
    return discovered


def clear_arch_cache() -> None:
    """Forget cached entry-point discoveries.

    Test helper — re-runs discovery on the next ``resolve_architecture``
    call. Production code should never need this.
    """
    global _discovered_cache
    _discovered_cache = None


def resolve_architecture(spec: Architecture | str | None) -> Architecture:
    """Coerce ``spec`` to a concrete :class:`Architecture`.

    * ``None`` → :class:`ReAct` (the default)
    * ``str`` → looked up in :data:`KNOWN`, then in any registered
      ``loomflow.architecture`` entry points
    * Architecture instance → returned as-is

    Unknown strings raise :class:`ConfigError` listing every known
    name (built-in + entry-point) for ergonomics.
    """
    if spec is None:
        return ReAct()
    if isinstance(spec, str):
        cls = KNOWN.get(spec) or _discover_entry_points().get(spec)
        if cls is None:
            discovered = _discover_entry_points()
            known = ", ".join(sorted(set(KNOWN) | set(discovered)))
            raise ConfigError(
                f"unknown architecture: {spec!r}. Known: {known}. "
                "Pass an Architecture instance directly for custom strategies, "
                f"or register one via the {ENTRY_POINT_GROUP!r} entry point group."
            )
        return cls()
    return spec
