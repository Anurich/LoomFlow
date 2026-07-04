"""Offline + online eval harness.

Offline: :class:`EvalHarness.run` iterates a :class:`Dataset`, runs the
agent once per case (fresh ``session_id`` per case, ``user_id="__eval__"``
so eval traffic stays in its own memory partition), captures the full
event trace via ``Agent.run``'s ``emit=`` seam, scores each case with
the configured metrics, and returns an :class:`EvalReport` with per-case
scores, per-metric aggregates, ``to_json()``, and a CI-friendly
``assert_thresholds`` gate. Cases run concurrently under an anyio
``CapacityLimiter``.

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
from .dataset import Case, Dataset
from .judge import warn_if_same_model
from .metrics import Metric

__all__ = ["CaseResult", "EvalHarness", "EvalReport", "OnlineScorer"]

EVAL_USER_ID = "__eval__"
"""``user_id`` used for all offline eval runs — keeps eval episodes in
their own memory partition, invisible to real users' recall."""


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

    ``error`` is set (and ``scores`` left empty) when ``agent.run``
    raised; the report's ``passed`` flag and ``assert_thresholds``
    both treat errored cases as failures.
    """

    case_id: str
    input: str
    output: str | None = None
    session_id: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class EvalReport(BaseModel):
    """Per-case scores + per-metric aggregates for one eval run."""

    results: list[CaseResult] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

    @property
    def passed(self) -> bool:
        """True when every case ran to completion without raising."""
        return all(r.error is None for r in self.results)

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
                "cases": [r.model_dump(mode="json") for r in self.results],
            },
            indent=indent,
        )

    def assert_thresholds(self, thresholds: Mapping[str, float]) -> None:
        """CI gate: raise :class:`AssertionError` listing every failure.

        A threshold fails when the metric's mean is below it, or when
        the metric produced no scores at all (a silent skip must not
        pass a gate). Errored cases are failures regardless of
        thresholds.
        """
        failures: list[str] = []
        for r in self.results:
            if r.error is not None:
                failures.append(f"case {r.case_id} errored: {r.error}")
        for name, threshold in thresholds.items():
            mean = self.mean(name)
            if mean is None:
                failures.append(f"{name}: no scores recorded (threshold {threshold:.2f})")
            elif mean < threshold:
                failures.append(f"{name}: mean {mean:.4f} < threshold {threshold:.2f}")
        if failures:
            raise AssertionError(
                "Eval thresholds not met:\n" + "\n".join(f"  - {f}" for f in failures)
            )


def _metric_applies(metric: Metric, case: Case) -> bool:
    applies = getattr(metric, "applies", None)
    if callable(applies):
        return bool(applies(case))
    return True


async def _score_case(
    metrics: Sequence[Metric],
    case: Case,
    result: RunResult,
    events: list[Event],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric in metrics:
        if not _metric_applies(metric, case):
            continue
        try:
            scores[metric.name] = await metric.score(case, result, events)
        except Exception as exc:  # noqa: BLE001 - one bad metric must not sink the run
            warnings.warn(
                f"metric {metric.name!r} raised on case {case.id}: "
                f"{type(exc).__name__}: {exc}; skipping its score",
                UserWarning,
                stacklevel=2,
            )
    return scores


class EvalHarness:
    """Run an agent over a dataset and score the traces.

    ``metrics`` is any sequence of :class:`Metric` implementations
    (built-ins live in :mod:`loomflow.eval.metrics`, the LLM judge in
    :mod:`loomflow.eval.judge`). ``concurrency`` bounds how many cases
    run at once. At construction the harness warns if any metric's
    ``model`` is the agent's own :class:`Model` instance — the
    same-model-judge anti-pattern.
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
        self._concurrency = concurrency
        agent_model = getattr(agent, "model", None)
        for metric in self._metrics:
            warn_if_same_model(metric, agent_model)

    async def run(self, dataset: Dataset | Sequence[Case]) -> EvalReport:
        """Offline eval: run every case, score, aggregate."""
        cases = list(dataset)
        started_at = datetime.now(UTC)
        results: list[CaseResult | None] = [None] * len(cases)
        limiter = anyio.CapacityLimiter(self._concurrency)

        async def _one(index: int, case: Case) -> None:
            async with limiter:
                results[index] = await self._run_case(case)

        async with anyio.create_task_group() as tg:
            for index, case in enumerate(cases):
                tg.start_soon(_one, index, case)

        finished_at = datetime.now(UTC)
        return EvalReport(
            results=[r for r in results if r is not None],
            started_at=started_at,
            finished_at=finished_at,
        )

    async def _run_case(self, case: Case) -> CaseResult:
        events: list[Event] = []

        async def _capture(event: Event) -> None:
            events.append(event)

        session_id = new_id("eval")
        try:
            result = await self._agent.run(
                case.input,
                session_id=session_id,
                user_id=EVAL_USER_ID,
                emit=_capture,
            )
        except Exception as exc:  # noqa: BLE001 - a crashing case must not abort the suite
            return CaseResult(
                case_id=case.id,
                input=case.input,
                session_id=session_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        scores = await _score_case(self._metrics, case, result, events)
        return CaseResult(
            case_id=case.id,
            input=case.input,
            output=result.output,
            session_id=session_id,
            scores=scores,
        )

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
        scores = await _score_case(self._metrics, case, result, events)
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
