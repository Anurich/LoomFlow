"""String / dict resolver for the ``telemetry=`` :class:`Agent` kwarg.

Mirrors the existing model / memory / audit_log resolvers so every
production backend can be wired up with the same declarative shape
(strings in TOML, dicts in YAML / settings) without the user having
to import each sink class.

Recognised string forms:

* ``"none"`` / ``"noop"`` — :class:`NoTelemetry`
* ``"memory"`` / ``"inmemory"`` — :class:`InMemoryTelemetry`
* ``"console"`` — :class:`ConsoleTelemetry`
* ``"file:<path>"`` — :class:`FileTelemetry` writing JSONL to
  ``<path>``
* ``"otel"`` — :class:`OTelTelemetry` with default providers

Recognised dict keys (``backend`` discriminator + backend-specific
extras):

* ``{"backend": "none"}`` / ``{"backend": "console"}`` /
  ``{"backend": "memory"}`` — args-free sinks
* ``{"backend": "file", "path": "..."}`` — :class:`FileTelemetry`
* ``{"backend": "otel", "instrumentation_name": "..."}`` —
  :class:`OTelTelemetry`

Already-constructed :class:`Telemetry` instances pass through
unchanged. ``None`` resolves to :class:`NoTelemetry`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.errors import ConfigError
from ..core.protocols import Telemetry
from .tracing import (
    ConsoleTelemetry,
    FileTelemetry,
    InMemoryTelemetry,
    NoTelemetry,
    OTelTelemetry,
)

__all__ = ["resolve_telemetry"]


def resolve_telemetry(spec: Any) -> Telemetry:
    if spec is None:
        return NoTelemetry()
    if isinstance(spec, str):
        return _resolve_string(spec)
    if isinstance(spec, Mapping):
        return _resolve_dict(spec)
    # Duck-typed: anything implementing the protocol passes through.
    return spec  # type: ignore[no-any-return]


def _resolve_string(spec: str) -> Telemetry:
    spec = spec.strip()
    if not spec:
        raise ConfigError(
            "telemetry= empty string. Use 'none', 'console', 'memory', "
            "'file:./spans.jsonl', or 'otel'."
        )
    low = spec.lower()
    if low in ("none", "noop"):
        return NoTelemetry()
    if low in ("memory", "inmemory"):
        return InMemoryTelemetry()
    if low == "console":
        return ConsoleTelemetry()
    if low == "otel":
        return OTelTelemetry()
    if low.startswith("file:"):
        path = spec[len("file:"):]
        if not path:
            raise ConfigError(
                "telemetry= 'file:' needs a path: 'file:./spans.jsonl'."
            )
        return FileTelemetry(path)
    raise ConfigError(
        f"telemetry= unrecognised string spec {spec!r}. Use 'none', "
        "'console', 'memory', 'file:./spans.jsonl', or 'otel'."
    )


def _resolve_dict(spec: Mapping[str, Any]) -> Telemetry:
    backend = spec.get("backend") or spec.get("type") or spec.get("name")
    if not isinstance(backend, str):
        raise ConfigError(
            "telemetry= dict must include 'backend' (or 'type' / 'name'). "
            "Recognised values: 'none', 'console', 'memory', 'file', 'otel'."
        )
    low = backend.lower()
    if low in ("none", "noop"):
        return NoTelemetry()
    if low in ("memory", "inmemory"):
        return InMemoryTelemetry()
    if low == "console":
        return ConsoleTelemetry()
    if low == "file":
        path = spec.get("path")
        if not isinstance(path, str) or not path:
            raise ConfigError(
                "telemetry= file backend requires 'path'."
            )
        return FileTelemetry(path)
    if low == "otel":
        kwargs: dict[str, Any] = {}
        if "instrumentation_name" in spec:
            kwargs["instrumentation_name"] = str(spec["instrumentation_name"])
        # tracer_provider / meter_provider only make sense as
        # already-constructed objects, so we pass them through if the
        # caller supplied them in the dict (handy for tests / DI).
        if "tracer_provider" in spec:
            kwargs["tracer_provider"] = spec["tracer_provider"]
        if "meter_provider" in spec:
            kwargs["meter_provider"] = spec["meter_provider"]
        return OTelTelemetry(**kwargs)
    raise ConfigError(
        f"telemetry= dict 'backend' = {backend!r} not recognised. "
        "Use 'none', 'console', 'memory', 'file', or 'otel'."
    )
