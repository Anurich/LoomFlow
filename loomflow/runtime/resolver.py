"""String / dict resolver for the ``runtime=`` :class:`Agent` kwarg.

Sync resolution only — :class:`PostgresRuntime` requires an async
``connect()`` call so it isn't reachable through this resolver;
construct it yourself with::

    runtime = await PostgresRuntime.connect("postgres://...")
    agent = Agent("...", runtime=runtime)

The string / dict forms cover the two sync-constructable runtimes:

* ``None`` / ``"inproc"`` — :class:`InProcRuntime` (no durability)
* ``"sqlite"`` (ephemeral in-memory db) /
  ``"sqlite:./path.db"`` — :class:`SqliteRuntime`
* ``{"backend": "inproc"}``
* ``{"backend": "sqlite", "path": "./path.db"}``

Already-constructed :class:`Runtime` instances pass through
unchanged so callers can mix dict-form config with a pre-built
PostgresRuntime kept alongside.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.errors import ConfigError
from ..core.protocols import Runtime
from .inproc import InProcRuntime
from .sqlite import SqliteRuntime

__all__ = ["resolve_runtime"]


def resolve_runtime(spec: Any) -> Runtime:
    if spec is None:
        return InProcRuntime()
    if isinstance(spec, str):
        return _resolve_string(spec)
    if isinstance(spec, Mapping):
        return _resolve_dict(spec)
    return spec  # type: ignore[no-any-return]


def _resolve_string(spec: str) -> Runtime:
    spec = spec.strip()
    if not spec:
        raise ConfigError(
            "runtime= empty string. Use 'inproc', 'sqlite', or "
            "'sqlite:./journal.db'."
        )
    if spec == "inproc":
        return InProcRuntime()
    if spec == "sqlite":
        return SqliteRuntime(":memory:")
    if spec.startswith("sqlite:") and not spec.startswith("sqlite://"):
        path = spec[len("sqlite:"):]
        return SqliteRuntime(path or ":memory:")
    if spec.startswith(("postgres://", "postgresql://")):
        raise ConfigError(
            "runtime= postgres URLs require the async constructor: "
            "`await PostgresRuntime.connect(dsn)` then pass the result "
            "as runtime=. This resolver is sync-only."
        )
    raise ConfigError(
        f"runtime= unrecognised string spec {spec!r}. Recognised:\n"
        "  inproc            — in-process, no durability (default)\n"
        "  sqlite            — ephemeral SQLite journal\n"
        "  sqlite:./path.db  — persistent SQLite journal\n"
        "Or pass a Runtime-protocol instance directly."
    )


def _resolve_dict(spec: Mapping[str, Any]) -> Runtime:
    backend = spec.get("backend") or spec.get("type") or spec.get("name")
    if not isinstance(backend, str):
        raise ConfigError(
            "runtime= dict must include 'backend' (or 'type' / 'name'). "
            "Recognised values: 'inproc', 'sqlite'."
        )
    low = backend.lower()
    if low == "inproc":
        return InProcRuntime()
    if low == "sqlite":
        path = spec.get("path") or spec.get("url") or ":memory:"
        return SqliteRuntime(str(path))
    if low in ("postgres", "postgresql"):
        raise ConfigError(
            "runtime= postgres requires the async constructor: "
            "`await PostgresRuntime.connect(dsn)`. Build it yourself "
            "and pass the instance as runtime=."
        )
    raise ConfigError(
        f"runtime= dict 'backend' = {backend!r} not recognised. "
        "Use 'inproc' or 'sqlite'."
    )
