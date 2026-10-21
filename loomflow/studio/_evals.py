"""Running evals from the web app.

``loom studio --agent app.agents:billing`` (or ``create_app(agents=)``)
names the agents the page may run. The web app process imports them, so
an eval runs with the real agent: its tools, guardrails and keys. An
eval started from the page runs on the app's event loop, records to the
file the page shows, and appears as an experiment while it runs. One
runs at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from . import _state
from ._catalog import agent_identity

_LOG = logging.getLogger("loomflow.studio")

METRICS = {
    "contains": "Contains the expected answer (any case)",
    "exact": "Matches the expected answer exactly",
    "judge": "LLM judge",
}


@dataclass(frozen=True)
class Target:
    """An agent the web app may run evals on."""

    key: str
    name: str
    spec: str | None
    agent: Any


def targets(agents: Iterable[Any] | Mapping[str, Any]) -> dict[str, Target]:
    """``create_app(agents=)`` as targets keyed by catalog key: Agent
    objects, or ``module:attribute`` names to import (a mapping's keys
    are only labels for the error messages)."""
    from .._targets import load_target
    from ..agent.api import Agent

    items = list(agents.values()) if isinstance(agents, Mapping) else list(agents)
    out: dict[str, Target] = {}
    for item in items:
        spec = item if isinstance(item, str) else None
        agent = load_target(item) if spec is not None else item
        if not isinstance(agent, Agent):
            raise ValueError(
                f"{spec or agent!r} is a {type(agent).__name__}, not an Agent: Studio runs evals on agents"
                " (workflows aren't supported yet)"
            )
        key, name = agent_identity(agent)
        if key in out:
            raise ValueError(f"two agents are both named {name!r}; give them different name= values")
        out[key] = Target(key=key, name=name, spec=spec, agent=agent)
    return out


def build_metrics(names: Iterable[str], judge_model: str | None) -> list[Any]:
    """The metrics the form picked. Raises ``ValueError`` with a message
    for the page when a choice can't be used."""
    from ..eval import Contains, ExactMatch

    chosen = [n for n in names if n in METRICS]
    if not chosen:
        raise ValueError("Pick at least one way to score the cases.")
    metrics: list[Any] = []
    if "contains" in chosen:
        metrics.append(Contains(case_sensitive=False))
    if "exact" in chosen:
        metrics.append(ExactMatch())
    if "judge" in chosen:
        if not judge_model:
            raise ValueError("Name a model for the LLM judge, for example gpt-5-mini.")
        from ..agent.api import _resolve_model
        from ..eval.judge import LLMJudge

        try:
            metrics.append(LLMJudge(_resolve_model(judge_model)))
        except Exception as exc:  # noqa: BLE001 — a bad model name is the user's to fix
            raise ValueError(f"The judge model {judge_model!r} can't be used: {exc}") from None
    return metrics


@dataclass
class Run:
    """An eval the web app started."""

    id: str
    name: str
    target: Target
    dataset: str
    total: int
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    # Why it was cancelled, for its page.
    stop_reason: str = "Cancelled"
    project: str = ""
    started: bool = False

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class Runner:
    """The evals one web app has started, at most one running."""

    def __init__(self, db_path: Path, targets: Mapping[str, Target]) -> None:
        self.db_path = db_path
        self.targets = dict(targets)
        self.runs: dict[str, Run] = {}

    def busy(self) -> Run | None:
        return next((r for r in self.runs.values() if r.running), None)

    def start(
        self,
        *,
        target: Target,
        dataset: str,
        project: str,
        total: int,
        metrics: list[Any],
        name: str,
        concurrency: int,
        max_cost_usd: float | None,
    ) -> Run:
        """Start an eval on the running event loop; returns at once."""
        run = Run(
            id="exp_" + uuid.uuid4().hex[:20], name=name, target=target, dataset=dataset, total=total, project=project,
        )
        self.runs[run.id] = run
        run.task = asyncio.get_running_loop().create_task(
            self._run(run, project=project, metrics=metrics, concurrency=concurrency, max_cost_usd=max_cost_usd),
            name=f"loomflow-studio-eval-{run.id}",
        )
        run.task.add_done_callback(lambda task: self._never_started(run, task))
        return run

    def _never_started(self, run: Run, task: asyncio.Task[None]) -> None:
        """A task cancelled before its first step never runs its body,
        so record that it stopped here."""
        if task.cancelled() and not run.started:
            self._record_stop(run, run.stop_reason, created=False)

    def _record_stop(self, run: Run, reason: str, *, created: bool) -> None:
        recorder = _state.recorder_for(_state.Settings(path=self.db_path, project=run.project))
        if not created:
            # Stopped before the harness recorded the experiment: its
            # page still needs a row to say so.
            recorder.submit(("experiment_start", {
                "id": run.id, "project": run.project, "name": run.name, "dataset": run.dataset, "agent": None,
                "started_at": time.time(), "metadata": {"cases": run.total},
            }))
        recorder.submit(("experiment_finish", run.id, time.time(), {"stopped": reason}))

    async def _run(
        self,
        run: Run,
        *,
        project: str,
        metrics: list[Any],
        concurrency: int,
        max_cost_usd: float | None,
    ) -> None:
        from ..eval import EvalHarness
        from . import dataset as load_dataset

        run.started = True
        settings = _state.Settings(path=self.db_path, project=project)
        recorder = _state.recorder_for(settings)
        # Only this task's context: the eval and its runs record to the
        # file the page shows, under the id the page already links to.
        _state.scope_var.set(_state.Scope(enabled=True, project=project, settings=settings))
        _state.next_experiment_var.set(run.id)

        def stopped(reason: str) -> None:
            # The harness takes the preset id when it records the start.
            self._record_stop(run, reason, created=_state.next_experiment_var.get() != run.id)

        try:
            cases = load_dataset(run.dataset, project=project, path=self.db_path)
            harness = EvalHarness(run.target.agent, metrics, concurrency=concurrency)
            await harness.run(cases, experiment=run.name, max_cost_usd=max_cost_usd)
        except asyncio.CancelledError:
            stopped(run.stop_reason)
            raise
        except Exception as exc:  # noqa: BLE001 — shown on the experiment's page
            _LOG.exception("Studio eval %s failed", run.id)
            stopped(f"Failed: {type(exc).__name__}: {exc}")
        # The run counts as running until its results are on disk, so
        # its page never sees it ended but unsaved.
        await anyio.to_thread.run_sync(recorder.flush)

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if run is None or not run.running or run.task is None:
            return False
        run.task.cancel()
        return True

    async def aclose(self) -> None:
        """Cancel the running eval (the server is stopping)."""
        running = [r for r in self.runs.values() if r.task is not None and not r.task.done()]
        tasks = [r.task for r in running if r.task is not None]
        for run in running:
            run.stop_reason = "Studio was stopped"
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task
