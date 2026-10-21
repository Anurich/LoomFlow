"""Loomflow Studio: record what your agents do, and look at it.

Turn recording on once, and every :class:`~loomflow.Agent` and
:class:`~loomflow.Workflow` run in the process is recorded to a local
SQLite file, with no change to the agents themselves::

    import loomflow.studio
    loomflow.studio.enable()        # or: export LOOMFLOW_STUDIO=1

    result = await agent.run("My card was charged twice")
    loomflow.studio.score(result, "resolved", True)

Then browse the runs::

    loom studio                     # http://127.0.0.1:7600

A recorded run is a **trace**: the agent run, its turns, model calls
(with the messages sent and the reply), tool calls (arguments, result,
the permission decision), guardrail hits, architecture steps and
errors, nested sub-agent and workflow runs included. Each agent and
workflow is catalogued by its definition, and a new **version** is
recorded whenever its instructions, model, tools or settings change.
:class:`~loomflow.eval.EvalHarness` runs are recorded as
**experiments** that can be compared case by case.

Everything stays in the file (``.loomflow/studio.db`` by default);
nothing is sent anywhere. Pass ``redact=`` to scrub text before it is
written, or ``capture_content=False`` to keep timing, tokens and
structure only.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import _state
from ._state import DEFAULT_PATH, ENV_VAR, PROJECT_ENV_VAR, Settings

__all__ = [
    "DEFAULT_PATH",
    "ENV_VAR",
    "PROJECT_ENV_VAR",
    "disable",
    "enable",
    "flush",
    "is_enabled",
    "score",
    "session",
]


def _redactor(redact: Callable[[str], str] | str | None) -> Callable[[str], str] | None:
    if redact is None or callable(redact):
        return redact
    if redact == "pii":
        from ..guardrails.builtin import PIIGuard

        def _pii(text: str) -> str:
            return PIIGuard._redact(text)[0]

        return _pii
    raise ValueError(f"redact must be a function, 'pii' or None, not {redact!r}")


def enable(
    path: str | Path | None = None,
    *,
    project: str | None = None,
    sample_rate: float = 1.0,
    capture_content: bool = True,
    redact: Callable[[str], str] | str | None = None,
) -> Path:
    """Record every run in this process to the Studio file at ``path``.

    ``path`` defaults to ``.loomflow/studio.db`` in the current folder,
    the file ``loom studio`` opens by default. ``project`` groups the
    runs (default: the current folder's name). ``sample_rate`` keeps
    that fraction of traces (a trace is kept or dropped whole).
    ``capture_content=False`` keeps no prompts, replies or tool
    payloads; ``redact`` (a ``str -> str`` function, or ``"pii"`` for
    loomflow's PII redaction) scrubs every piece of text first.

    Overrides ``LOOMFLOW_STUDIO``; a :func:`session` block overrides
    this. Returns the file's path.
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be between 0 and 1, not {sample_rate!r}")
    settings = Settings(
        path=Path(path).expanduser() if path is not None else DEFAULT_PATH,
        project=project,
        sample_rate=sample_rate,
        capture_content=capture_content,
        redact=_redactor(redact),
    )
    _state.set_global(settings)
    # Create the file and schema now, so a bad path fails here.
    _state.recorder_for(settings)
    return settings.path


def disable() -> None:
    """Stop recording runs started from now on (``LOOMFLOW_STUDIO``
    still applies; set it to ``off`` to be sure). Records already
    queued are still written."""
    _state.set_global(None)


def is_enabled() -> bool:
    """Whether a run started here, now, would be recorded."""
    return _state.current_settings() is not None


@contextlib.contextmanager
def session(
    *,
    project: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    enabled: bool | None = None,
) -> Iterator[None]:
    """Settings for the runs started inside this block: a ``project``,
    ``tags`` and ``metadata`` to record with them, or ``enabled=False``
    to record none of them (``True`` records them even when Studio is
    otherwise off). Blocks nest; the innermost wins."""
    outer = _state.scope_var.get()
    scope = _state.Scope(
        enabled=enabled if enabled is not None else (outer.enabled if outer else None),
        project=project or (outer.project if outer else None),
        tags=tuple(outer.tags if outer else ()) + tuple(tags),
        metadata={**(outer.metadata if outer else {}), **(metadata or {})},
        experiment_id=outer.experiment_id if outer else None,
        case_key=outer.case_key if outer else None,
    )
    token = _state.scope_var.set(scope)
    try:
        yield
    finally:
        _state.scope_var.reset(token)


def score(
    run: Any,
    key: str,
    value: bool | int | float | str,
    *,
    comment: str | None = None,
    source: str = "api",
) -> None:
    """Attach a score to a recorded run.

    ``run`` is a :class:`~loomflow.RunResult` or its ``run_id``.
    ``value`` is a bool (stored as 1/0, shown as yes/no), a number, or
    a label. ``source`` says where it came from (``"api"``, ``"user"``,
    ``"judge"``, ...). Raises ``RuntimeError`` when Studio isn't on.
    """
    from ._hooks import _score_row

    run_id = run if isinstance(run, str) else getattr(run, "run_id", None)
    if not run_id:
        raise ValueError("score() needs a RunResult or a run_id")
    recorder = _state.active_recorder()
    if recorder is None:
        recorders = _state.all_recorders()
        if not recorders:
            raise RuntimeError(
                "Studio isn't recording: call loomflow.studio.enable() or set LOOMFLOW_STUDIO"
            )
        recorder = recorders[-1]
    recorder.submit(("score", _score_row(run_id, key, value, comment, source)))


def flush(timeout: float = 5.0) -> bool:  # noqa: ASYNC109 — blocking by design
    """Wait until every record queued so far is written to disk.
    Returns ``False`` if ``timeout`` seconds pass first."""
    return all(r.flush(timeout=timeout) for r in _state.all_recorders())
