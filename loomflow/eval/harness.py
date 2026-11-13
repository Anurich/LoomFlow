"""Offline + online eval harness.

Offline: :class:`EvalHarness.run` iterates a :class:`Dataset`, runs the
agent once per case under an identity of its own (a fresh ``session_id``
and a ``user_id`` unique to the case and to this harness run, both
prefixed ``__eval__``), so no case can recall another case's answer —
or an earlier suite run's — through the agent's memory, and eval
traffic stays out of real users' partitions. What the case wrote to the
agent's memory is erased when the case ends. The harness captures the
full event trace via ``Agent.run``'s ``emit=`` seam, scores each case
with the configured metrics, and returns an :class:`EvalReport` with
per-case scores, per-metric aggregates, ``to_json()``, and a
CI-friendly ``assert_thresholds`` gate. Cases run concurrently under an
anyio ``CapacityLimiter``.

A case fails — ``report.passed`` is False and ``assert_thresholds``
raises — when its run raised, timed out (``case_timeout``) or came back
interrupted (``allow_interrupted=False``, the default), or when a
metric raised or returned something that is not a score in ``[0, 1]``
(NaN, infinity, out of range).

Online: :meth:`EvalHarness.online` returns an :class:`OnlineScorer` the
application calls *after its own runs* — nothing is wrapped or
monkeypatched. ``maybe_score`` samples via a seeded ``random.Random``
and scores only ground-truth-free metrics (each metric's ``applies``
gate skips the rest); ``submit`` is the fire-and-forget variant that
schedules scoring on an internal task group (``async with scorer:``)
so the response path never blocks on judge calls. ``rollup()`` returns
per-day aggregates.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import anyio
import anyio.abc
from pydantic import BaseModel, Field

from ..core.ids import new_id
from ..core.types import Event, RunResult
from ..studio import _hooks as _studio
from .dataset import Case, Dataset
from .judge import _metric_usage_var, warn_if_same_model
from .metrics import Metric

__all__ = ["CaseResult", "EvalHarness", "EvalReport", "OnlineScorer"]

EVAL_USER_ID = "__eval__"
"""Prefix of the ``user_id`` every offline eval case runs under. Each
case gets its own (``__eval__:<harness run>:<case number>``), so eval
episodes are invisible to real users' recall and to each other."""


@runtime_checkable
class _RunnableAgent(Protocol):
    """Structural slice of :class:`loomflow.Agent` the harness needs.

    Kept as a Protocol so tests (and non-Agent runnables like
    architecture wrappers) can plug in anything with a compatible
    ``run``. ``emit`` is the trace-capture seam: an awaitable callback
    invoked once per :class:`Event` during the run.
    """

    async def run(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        emit: Callable[[Event], Awaitable[None]] | None = None,
    ) -> RunResult: ...


class CaseResult(BaseModel):
    """Outcome of one case: the agent's output plus per-metric scores.

    ``error`` is set when the case failed as a whole: ``agent.run``
    raised or the case timed out (``scores`` is then empty), or the run
    came back interrupted and the harness wasn't told to allow that
    (the partial answer is still scored). ``interrupted`` /
    ``interruption_reason`` record an interrupted run either way.
    ``metric_errors`` maps a metric that raised on this case — or
    returned something that isn't a score in ``[0, 1]`` — to its error.
    The report's ``passed`` flag and ``assert_thresholds`` treat
    ``error`` and ``metric_errors`` as failures. ``cost_usd`` is what
    the case cost: the agent's run plus the metrics' own model calls
    (``metric_cost_usd``, e.g. an LLM judge).
    """

    case_id: str
    input: str
    output: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    metric_errors: dict[str, str] = Field(default_factory=dict)
    interrupted: bool = False
    interruption_reason: str | None = None
    cost_usd: float = 0.0
    metric_cost_usd: float = 0.0


class EvalReport(BaseModel):
    """Per-case scores + per-metric aggregates for one eval run.

    ``skipped`` lists the ids of the cases that didn't run because the
    run's ``max_cost_usd`` was reached first."""

    results: list[CaseResult] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    skipped: list[str] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """What the cases cost together: the agent's runs and the
        metrics' own model calls (LLM judges)."""
        return sum(r.cost_usd for r in self.results)

    @property
    def passed(self) -> bool:
        """True when every case ran to completion (no exception,
        timeout or disallowed interruption) and every metric gave it a
        valid score."""
        return not self.skipped and all(r.error is None and not r.metric_errors for r in self.results)

    def scores_for(self, metric: str) -> list[float]:
        return [r.scores[metric] for r in self.results if metric in r.scores]

    def mean(self, metric: str) -> float | None:
        """Mean score for ``metric`` over the cases it applied to, or None."""
        scores = self.scores_for(metric)
        return sum(scores) / len(scores) if scores else None

    def min(self, metric: str) -> float | None:
        """Minimum score for ``metric`` over the cases it applied to, or None."""
        scores = self.scores_for(metric)
        return min(scores) if scores else None

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-metric ``{"mean": ..., "min": ..., "count": ...}`` aggregates."""
        names = sorted({name for r in self.results for name in r.scores})
        out: dict[str, dict[str, float]] = {}
        for name in names:
            scores = self.scores_for(name)
            out[name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "count": float(len(scores)),
            }
        return out

    def to_json(self, *, indent: int | None = 2) -> str:
        """The full report — cases, aggregates, pass flag — as JSON."""
        return json.dumps(
            {
                "passed": self.passed,
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat(),
                "metrics": self.summary(),
                "cost_usd": self.cost_usd,
                "skipped": self.skipped,
                "cases": [r.model_dump(mode="json") for r in self.results],
            },
            indent=indent,
        )

    def assert_thresholds(self, thresholds: Mapping[str, float]) -> None:
        """CI gate: raise :class:`AssertionError` listing every failure.

        A threshold fails when the metric's mean is below it, when the
        metric produced no scores at all (a silent skip must not pass a
        gate), or when it raised (or returned a non-score such as NaN)
        on any case — a mean over the cases it managed to score says
        nothing about the rest. Errored, timed-out and interrupted
        cases are failures regardless of thresholds.
        """
        failures: list[str] = []
        for r in self.results:
            if r.error is not None:
                failures.append(f"case {r.case_id} errored: {r.error}")
        for name, threshold in thresholds.items():
            errored = [r.case_id for r in self.results if name in r.metric_errors]
            if errored:
                failures.append(
                    f"{name}: raised on {len(errored)}/{len(self.results)} case(s) "
                    f"({', '.join(errored[:5])}{', ...' if len(errored) > 5 else ''})"
                )
            mean = self.mean(name)
            if mean is None:
                failures.append(f"{name}: no scores recorded (threshold {threshold:.2f})")
            elif not mean >= threshold:  # also fails a NaN mean
                failures.append(f"{name}: mean {mean:.4f} < threshold {threshold:.2f}")
        if failures:
            raise AssertionError(
                "Eval thresholds not met:\n" + "\n".join(f"  - {f}" for f in failures)
            )


def _metric_cost(usage: Sequence[Any]) -> float:
    return float(sum(float(getattr(u, "cost_usd", 0.0) or 0.0) for u in usage))


def _metric_applies(metric: Metric, case: Case) -> bool:
    applies = getattr(metric, "applies", None)
    if callable(applies):
        return bool(applies(case))
    return True


def _check_unique_names(metrics: Sequence[Metric]) -> None:
    """Two metrics with one name would overwrite each other's scores."""
    seen: set[str] = set()
    for metric in metrics:
        name = metric.name
        if name in seen:
            raise ValueError(
                f"two metrics are both named {name!r}: their scores would overwrite "
                "each other. Give each its own name, e.g. "
                "LLMJudge(model, rubric=..., name='accuracy')."
            )
        seen.add(name)


def _valid_score(value: Any) -> float:
    """``value`` as a score in ``[0, 1]``, or ``ValueError``: a NaN
    would pass any ``mean < threshold`` gate, and an out-of-range value
    would skew the mean."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric returned {value!r} ({type(value).__name__}), not a number")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"metric returned {score!r}, not a finite score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"metric returned {score!r}, outside the range [0, 1]")
    return score


async def _score_case(
    metrics: Sequence[Metric],
    case: Case,
    result: RunResult,
    events: list[Event],
) -> tuple[dict[str, float], dict[str, str]]:
    """``(scores, metric_errors)`` for one case."""
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    for metric in metrics:
        if not _metric_applies(metric, case):
            continue
        try:
            scores[metric.name] = _valid_score(await metric.score(case, result, events))
        except Exception as exc:  # noqa: BLE001 - one bad metric must not sink the run
            errors[metric.name] = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"metric {metric.name!r} raised on case {case.id}: "
                f"{type(exc).__name__}: {exc}; the case counts as failed",
                UserWarning,
                stacklevel=2,
            )
    return scores, errors


class EvalHarness:
    """Run an agent over a dataset and score the traces.

    ``metrics`` is any sequence of :class:`Metric` implementations
    (built-ins live in :mod:`loomflow.eval.metrics`, the LLM judge in
    :mod:`loomflow.eval.judge`). ``concurrency`` bounds how many cases
    run at once. At construction the harness warns if any metric's
    ``model`` is the agent's own :class:`Model` instance — the
    same-model-judge anti-pattern — and raises ``ValueError`` when two
    metrics share a ``name`` (their scores would overwrite each other).
    """

    def __init__(
        self,
        agent: _RunnableAgent,
        metrics: Sequence[Metric],
        *,
        concurrency: int = 4,
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self._agent = agent
        self._metrics = list(metrics)
        _check_unique_names(self._metrics)
        self._concurrency = concurrency
        agent_model = getattr(agent, "model", None)
        for metric in self._metrics:
            warn_if_same_model(metric, agent_model)

    async def run(
        self,
        dataset: Dataset | Sequence[Case],
        *,
        experiment: str | None = None,
        max_cost_usd: float | None = None,
        case_timeout: float | None = None,
        allow_interrupted: bool = False,
    ) -> EvalReport:
        """Offline eval: run every case, score, aggregate.

        ``max_cost_usd`` stops starting new cases once the cases so far
        have cost that much — the agent's runs plus the metrics' own
        model calls, such as an LLM judge (cases already running
        finish, so it can go over by up to ``concurrency - 1`` cases);
        the rest are listed in the report's ``skipped``.

        ``case_timeout`` is the seconds one case may take, run and
        scoring together. A case that runs out is cancelled and
        recorded as failed (``error`` says it timed out) and the suite
        moves on; what its cancelled agent run had spent is not known,
        so only its metrics' spend is counted.

        A run that comes back interrupted (``max_turns``, a budget, a
        guardrail, an approval it is waiting for) did not finish its
        task: the case fails with ``error="interrupted: <reason>"``
        (its partial answer is still scored). Pass
        ``allow_interrupted=True`` when stopping early is the expected
        outcome; the interruption is recorded on the case either way.

        When Studio is recording, the run is saved as an experiment
        (named ``experiment``, default the agent's name), each case's
        run as a trace with its scores, so experiments on the same
        dataset can be compared.
        """
        if max_cost_usd is not None and not max_cost_usd > 0:
            raise ValueError(f"max_cost_usd must be more than 0, got {max_cost_usd!r}")
        if case_timeout is not None and not case_timeout > 0:
            raise ValueError(f"case_timeout must be more than 0, got {case_timeout!r}")
        cases = list(dataset)
        started_at = datetime.now(UTC)
        results: list[CaseResult | None] = [None] * len(cases)
        skipped = [False] * len(cases)
        spent = 0.0
        limiter = anyio.CapacityLimiter(self._concurrency)
        # Unique to this call: cases of an earlier (or concurrent) run
        # of the same suite on the same agent never share a partition.
        run_token = new_id("run")
        studio = await _studio.experiment_start(
            self._agent, dataset, cases, [m.name for m in self._metrics], name=experiment
        )

        async def _one(index: int, case: Case) -> None:
            nonlocal spent
            async with limiter:
                if max_cost_usd is not None and spent >= max_cost_usd:
                    skipped[index] = True
                    return
                result = await self._run_case(
                    case,
                    studio=studio,
                    position=index,
                    user_id=f"{EVAL_USER_ID}:{run_token}:{index}",
                    case_timeout=case_timeout,
                    allow_interrupted=allow_interrupted,
                )
                results[index] = result
                spent += result.cost_usd

        async with anyio.create_task_group() as tg:
            for index, case in enumerate(cases):
                tg.start_soon(_one, index, case)

        finished_at = datetime.now(UTC)
        report = EvalReport(
            results=[r for r in results if r is not None],
            started_at=started_at,
            finished_at=finished_at,
            skipped=[case.id for case, skip in zip(cases, skipped, strict=True) if skip],
        )
        summary: dict[str, Any] = {"passed": report.passed, "metrics": report.summary()}
        if report.skipped:
            summary["skipped"] = len(report.skipped)
            summary["stopped"] = f"Spending cap of ${max_cost_usd:g} reached"
        _studio.experiment_finish(studio, summary)
        return report

    async def _run_case(
        self,
        case: Case,
        *,
        studio: _studio.ExperimentHandle | None = None,
        position: int = 0,
        user_id: str | None = None,
        case_timeout: float | None = None,
        allow_interrupted: bool = False,
    ) -> CaseResult:
        session_id = new_id("eval")
        user_id = user_id if user_id is not None else f"{EVAL_USER_ID}:{new_id('run')}:{position}"
        key = _studio.case_key(case)
        outcome: CaseResult | None = None
        # The metrics' own model calls (LLM judges) report their usage
        # here; this task's context, so concurrent cases don't mix.
        metric_usage: list[Any] = []
        usage_token = _metric_usage_var.set(metric_usage)
        try:
            with anyio.move_on_after(case_timeout):
                outcome = await self._run_and_score(
                    case,
                    studio=studio,
                    position=position,
                    key=key,
                    session_id=session_id,
                    user_id=user_id,
                    allow_interrupted=allow_interrupted,
                    metric_usage=metric_usage,
                )
        finally:
            _metric_usage_var.reset(usage_token)
            await self._forget(user_id)
        if outcome is not None:
            return outcome
        error = f"TimeoutError: case took longer than case_timeout={case_timeout:g}s"
        _studio.experiment_record_case(
            studio, position=position, key=key, case=case, run_id=None,
            output=None, scores={}, error=error,
        )
        return CaseResult(
            case_id=case.id,
            input=case.input,
            session_id=session_id,
            user_id=user_id,
            error=error,
            cost_usd=_metric_cost(metric_usage),
            metric_cost_usd=_metric_cost(metric_usage),
        )

    async def _run_and_score(
        self,
        case: Case,
        *,
        studio: _studio.ExperimentHandle | None,
        position: int,
        key: str,
        session_id: str,
        user_id: str,
        allow_interrupted: bool,
        metric_usage: list[Any],
    ) -> CaseResult:
        events: list[Event] = []

        async def _capture(event: Event) -> None:
            events.append(event)

        try:
            with _studio.experiment_case(studio, key):
                result = await self._agent.run(
                    case.input,
                    session_id=session_id,
                    user_id=user_id,
                    emit=_capture,
                )
        except Exception as exc:  # noqa: BLE001 - a crashing case must not abort the suite
            error = f"{type(exc).__name__}: {exc}"
            _studio.experiment_record_case(
                studio, position=position, key=key, case=case, run_id=None,
                output=None, scores={}, error=error,
            )
            return CaseResult(
                case_id=case.id,
                input=case.input,
                session_id=session_id,
                user_id=user_id,
                error=error,
            )
        scores, metric_errors = await _score_case(self._metrics, case, result, events)
        metric_cost = _metric_cost(metric_usage)
        cost = float(getattr(result, "cost_usd", 0.0) or 0.0) + metric_cost
        interrupted = bool(getattr(result, "interrupted", False))
        reason = getattr(result, "interruption_reason", None) if interrupted else None
        stopped: str | None = None
        if interrupted and not allow_interrupted:
            stopped = f"interrupted: {reason or 'unknown reason'}"
        _studio.experiment_record_case(
            studio, position=position, key=key, case=case, run_id=result.run_id,
            output=result.output, scores=scores,
            error="; ".join(
                [*([stopped] if stopped else []), *(f"{k}: {v}" for k, v in metric_errors.items())]
            )
            or None,
        )
        return CaseResult(
            case_id=case.id,
            input=case.input,
            output=result.output,
            session_id=session_id,
            user_id=user_id,
            scores=scores,
            error=stopped,
            metric_errors=metric_errors,
            interrupted=interrupted,
            interruption_reason=reason,
            cost_usd=cost,
            metric_cost_usd=metric_cost,
        )

    async def _forget(self, user_id: str) -> None:
        """Erase what one case wrote to the agent's memory. Best effort:
        the case's identity is never reused, so anything left behind (a
        memory without ``forget``, a store that is down) is unreachable
        clutter, not contamination."""
        forget = getattr(getattr(self._agent, "memory", None), "forget", None)
        if not callable(forget):
            return
        with anyio.CancelScope(shield=True), anyio.move_on_after(5):
            try:
                await forget(user_id=user_id)
            except Exception:  # noqa: BLE001, S110 - cleanup must not fail a scored case
                pass

    def online(self, *, sample_rate: float = 0.1, seed: int | None = None) -> OnlineScorer:
        """An :class:`OnlineScorer` sharing this harness's metric suite."""
        return OnlineScorer(self._metrics, sample_rate=sample_rate, seed=seed)


class OnlineScorer:
    """Score a sampled fraction of live runs; no ground truth needed.

    The application calls :meth:`maybe_score` (awaited, returns the
    scores or ``None`` when the run wasn't sampled) or :meth:`submit`
    (fire-and-forget; requires ``async with scorer:`` so the internal
    task group exists) after its own ``agent.run`` / ``agent.stream``
    completes, passing the prompt, the :class:`RunResult`, and the
    captured events. Metrics gated on ground truth (``applies`` returns
    False for the unlabelled case) are skipped automatically.

    Sampling uses ``random.Random(seed)`` — deterministic for a fixed
    seed and call sequence. Aggregates accumulate per UTC day (keyed by
    the run's ``finished_at``); read them with :meth:`rollup`.
    """

    def __init__(
        self,
        metrics: Sequence[Metric],
        *,
        sample_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
        self._metrics = list(metrics)
        _check_unique_names(self._metrics)
        self._sample_rate = sample_rate
        self._rng = random.Random(seed)
        self._seen = 0
        self._sampled = 0
        # day (ISO date) -> metric name -> list of scores
        self._by_day: dict[str, dict[str, list[float]]] = {}
        self._tg: anyio.abc.TaskGroup | None = None

    async def __aenter__(self) -> OnlineScorer:
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        tg, self._tg = self._tg, None
        if tg is None:  # pragma: no cover - unbalanced exit
            return None
        return await tg.__aexit__(exc_type, exc, tb)

    async def maybe_score(
        self, prompt: str, result: RunResult, events: list[Event]
    ) -> dict[str, float] | None:
        """Sample-and-score. Returns the scores, or None when skipped."""
        self._seen += 1
        if self._rng.random() >= self._sample_rate:
            return None
        self._sampled += 1
        case = Case(input=prompt)  # unlabelled: ground-truth metrics skip via applies()
        scores, _errors = await _score_case(self._metrics, case, result, events)
        # Studio (when recording): the scores join the run's trace.
        _studio.record_scores(result, scores, source="online")
        day = result.finished_at.astimezone(UTC).date().isoformat()
        bucket = self._by_day.setdefault(day, {})
        for name, score in scores.items():
            bucket.setdefault(name, []).append(score)
        return scores

    def submit(self, prompt: str, result: RunResult, events: list[Event]) -> None:
        """Fire-and-forget :meth:`maybe_score` on the internal task group.

        Non-blocking contract: returns immediately; scoring (including
        any LLM-judge calls) runs in the background and completes
        before ``async with scorer:`` exits. Raises RuntimeError when
        called outside the context manager.
        """
        if self._tg is None:
            raise RuntimeError(
                "OnlineScorer.submit() requires the scorer to be entered: "
                "use 'async with scorer:' around the serving loop, or call "
                "'await scorer.maybe_score(...)' directly."
            )
        self._tg.start_soon(self._submit_one, prompt, result, events)

    async def _submit_one(self, prompt: str, result: RunResult, events: list[Event]) -> None:
        try:
            await self.maybe_score(prompt, result, events)
        except Exception as exc:  # noqa: BLE001 - background scoring must never crash the app
            warnings.warn(
                f"online scoring failed: {type(exc).__name__}: {exc}",
                UserWarning,
                stacklevel=2,
            )

    def rollup(self) -> dict[str, Any]:
        """Daily-style aggregates: counts + per-day per-metric mean/min/count."""
        days: dict[str, dict[str, dict[str, float]]] = {}
        for day, metrics in sorted(self._by_day.items()):
            days[day] = {
                name: {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "count": float(len(scores)),
                }
                for name, scores in sorted(metrics.items())
            }
        return {
            "seen": self._seen,
            "sampled": self._sampled,
            "sample_rate": self._sample_rate,
            "days": days,
        }
