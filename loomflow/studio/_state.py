"""Where Studio's switches live, and how a run finds its recorder.

Precedence, highest first: a :func:`~loomflow.studio.session` block,
then :func:`~loomflow.studio.enable`, then the ``LOOMFLOW_STUDIO``
environment variable, then off. The environment is read at each run's
start (not cached), so a variable set after import still takes effect.

Everything a running trace needs is in context variables, so a
workflow's nodes, an agent's sub-agents and parallel tool calls all
see the trace they belong to, while concurrent runs stay apart.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._recorder import Recorder, TraceState

ENV_VAR = "LOOMFLOW_STUDIO"
PROJECT_ENV_VAR = "LOOMFLOW_STUDIO_PROJECT"
KEEP_DAYS_ENV_VAR = "LOOMFLOW_STUDIO_KEEP_DAYS"
DEFAULT_PATH = Path(".loomflow") / "studio.db"
_OFF = frozenset({"", "0", "off", "false", "no"})
_ON = frozenset({"1", "on", "true", "yes"})


@dataclass(frozen=True)
class Settings:
    """How a recorder records (see :func:`~loomflow.studio.enable`)."""

    path: Path = DEFAULT_PATH
    project: str | None = None
    sample_rate: float = 1.0
    capture_content: bool = True
    redact: Callable[[str], str] | None = None
    # Delete runs older than this many days (eval runs are kept).
    keep_days: float | None = None


@dataclass(frozen=True)
class Scope:
    """Overrides for the runs started inside a ``session(...)`` block."""

    enabled: bool | None = None
    project: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    # Set by the eval harness: the experiment a case run belongs to.
    experiment_id: str | None = None
    case_key: str | None = None
    # Record to this file whatever else is configured (evals the web
    # app starts record to the file it shows).
    settings: Settings | None = None


# The trace being recorded, and the span new spans hang under.
trace_var: ContextVar[TraceState | None] = ContextVar("loomflow_studio_trace", default=None)
parent_var: ContextVar[str | None] = ContextVar("loomflow_studio_parent", default=None)
# Set inside an unsampled trace, so its nested runs don't start their own.
suppress_var: ContextVar[bool] = ContextVar("loomflow_studio_suppress", default=False)
scope_var: ContextVar[Scope | None] = ContextVar("loomflow_studio_scope", default=None)
# The id the next experiment started here takes (the web app picks it
# first, to show the experiment's page while it runs).
next_experiment_var: ContextVar[str | None] = ContextVar("loomflow_studio_next_experiment", default=None)

_global: Settings | None = None
_recorders: dict[str, Recorder] = {}
_lock = threading.Lock()


def set_global(settings: Settings | None) -> None:
    global _global
    _global = settings


def get_global() -> Settings | None:
    return _global


def settings_from_env() -> Settings | None:
    """The settings ``LOOMFLOW_STUDIO`` asks for, or ``None`` (off)."""
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw.strip().lower() in _OFF:
        return None
    value = raw.strip()
    path = DEFAULT_PATH if value.lower() in _ON else Path(value).expanduser()
    return Settings(
        path=path,
        project=os.environ.get(PROJECT_ENV_VAR) or None,
        keep_days=keep_days_from_env(),
    )


def keep_days_from_env() -> float | None:
    """``LOOMFLOW_STUDIO_KEEP_DAYS`` as a number of days, or ``None``."""
    raw = (os.environ.get(KEEP_DAYS_ENV_VAR) or "").strip()
    if not raw:
        return None
    try:
        days = float(raw)
    except ValueError:
        raise ValueError(f"{KEEP_DAYS_ENV_VAR} must be a number of days, not {raw!r}") from None
    if days <= 0:
        raise ValueError(f"{KEEP_DAYS_ENV_VAR} must be more than 0, not {raw!r}")
    return days


def current_settings() -> Settings | None:
    """The settings a run starting now records with, or ``None``."""
    scope = scope_var.get()
    if scope is not None and scope.enabled is False:
        return None
    if scope is not None and scope.settings is not None:
        return scope.settings
    settings = _global or settings_from_env()
    if settings is None and scope is not None and scope.enabled:
        settings = Settings()
    return settings


def recorder_for(settings: Settings) -> Recorder:
    """The process's recorder for ``settings.path`` (one per file), set
    up with ``settings``."""
    from ._recorder import Recorder

    key = str(settings.path.expanduser().resolve())
    with _lock:
        recorder = _recorders.get(key)
        if recorder is None:
            recorder = _recorders[key] = Recorder(settings)
        else:
            recorder.configure(settings)
        return recorder


def active_recorder() -> Recorder | None:
    settings = current_settings()
    return None if settings is None else recorder_for(settings)


def all_recorders() -> list[Recorder]:
    with _lock:
        return list(_recorders.values())


def default_project() -> str:
    """The current folder's name: the project a run records under when
    nothing names one."""
    return Path.cwd().name or "default"
