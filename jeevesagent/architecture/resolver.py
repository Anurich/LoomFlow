"""Resolve architecture specs (string or instance) to a concrete
:class:`Architecture` instance.

In v0.3, only ``"react"`` is shipped as a known string. The resolver
also accepts any object that satisfies the :class:`Architecture`
protocol — most user-facing flows just pass an instance like
``ReAct(max_turns=20)`` directly.

Future: entry-point discovery so third-party packages can register
custom architectures via ``[project.entry-points."jeevesagent.architecture"]``
and be referenced by string from user code without imports.
"""

from __future__ import annotations

from ..core.errors import ConfigError
from .base import Architecture
from .react import ReAct

KNOWN: dict[str, type[Architecture]] = {
    "react": ReAct,
}


def resolve_architecture(spec: Architecture | str | None) -> Architecture:
    """Coerce ``spec`` to a concrete :class:`Architecture`.

    * ``None`` → :class:`ReAct` (the default)
    * ``str`` → looked up in :data:`KNOWN` (only ``"react"`` in v0.3)
    * Architecture instance → returned as-is

    Unknown strings raise :class:`ConfigError` with a list of known
    names.
    """
    if spec is None:
        return ReAct()
    if isinstance(spec, str):
        cls = KNOWN.get(spec)
        if cls is None:
            known = ", ".join(sorted(KNOWN))
            raise ConfigError(
                f"unknown architecture: {spec!r}. Known: {known}. "
                "Pass an Architecture instance directly for custom strategies."
            )
        return cls()
    return spec
