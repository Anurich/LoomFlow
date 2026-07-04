"""Built-in eval metrics and the :class:`Metric` protocol.

A metric scores one (case, run) pair in ``[0.0, 1.0]`` from three
inputs: the :class:`~loomflow.eval.Case` (ground truth, when present),
the final :class:`~loomflow.RunResult`, and the full list of
:class:`~loomflow.core.types.Event`\\ s captured during the run (the
trace). Trace-derived metrics read ``TOOL_CALL`` / ``TOOL_RESULT``
events, whose payloads carry ``ToolCall.model_dump()`` under ``"call"``
and ``ToolResult.model_dump()`` under ``"result"`` respectively.

Metrics MAY additionally expose ``applies(case) -> bool``; the harness
skips a metric for cases where it returns ``False`` (e.g. ground-truth
metrics on unlabelled cases — the online-scoring path). Metrics without
``applies`` are scored on every case.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from ..core.types import Event, EventKind, RunResult
from .dataset import Case

__all__ = [
    "Contains",
    "ExactMatch",
    "Metric",
    "MultiStepCoherence",
    "ToolExecutionSuccess",
    "ToolSelectionAccuracy",
    "tool_calls_from_events",
    "tool_results_from_events",
]


@runtime_checkable
class Metric(Protocol):
    """Scores one case's run in ``[0.0, 1.0]``. Higher is better."""

    name: str

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float: ...


def tool_calls_from_events(events: list[Event]) -> list[dict[str, Any]]:
    """Extract ``ToolCall.model_dump()`` dicts from TOOL_CALL events, in order."""
    return [
        e.payload["call"] for e in events if e.kind is EventKind.TOOL_CALL and "call" in e.payload
    ]


def tool_results_from_events(events: list[Event]) -> list[dict[str, Any]]:
    """Extract ``ToolResult.model_dump()`` dicts from TOOL_RESULT events, in order."""
    return [
        e.payload["result"]
        for e in events
        if e.kind is EventKind.TOOL_RESULT and "result" in e.payload
    ]


class ToolSelectionAccuracy:
    """Fraction of ``case.expected_tools`` the agent actually called.

    Order-insensitive set semantics: duplicates in ``expected_tools``
    count once, and extra unexpected calls are NOT penalised (this
    metric measures recall of the expected set, not precision). An
    empty ``expected_tools`` list scores 1.0 vacuously ("the agent
    was expected to call nothing in particular"). Cases with
    ``expected_tools=None`` are skipped via :meth:`applies`.
    """

    name = "tool_selection_accuracy"

    def applies(self, case: Case) -> bool:
        return case.expected_tools is not None

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float:
        expected = set(case.expected_tools or [])
        if not expected:
            return 1.0
        called = {c["tool"] for c in tool_calls_from_events(events)}
        return len(expected & called) / len(expected)


class ToolExecutionSuccess:
    """Fraction of tool calls whose results were ok (not error / denied).

    Reads TOOL_RESULT events; a result counts as successful when
    ``ok`` is true and ``denied`` is false (denied results already
    carry ``ok=False``, but we check both defensively). A run with no
    tool results scores 1.0 vacuously — nothing failed.
    """

    name = "tool_execution_success"

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float:
        results = tool_results_from_events(events)
        if not results:
            return 1.0
        ok = sum(1 for r in results if r.get("ok") and not r.get("denied"))
        return ok / len(results)


def _call_signature(call: dict[str, Any]) -> str:
    """Stable identity for "the same tool call": name + canonical args JSON."""
    args = call.get("args") or {}
    try:
        canonical = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = repr(args)
    return f"{call.get('tool', '?')}({canonical})"


class MultiStepCoherence:
    """Heuristic coherence score for multi-step tool traces.

    This is a *crude, honest heuristic* — it does not understand the
    task; it only flags two mechanical incoherence smells in the
    trace:

    1. **Repeated identical calls** — the same tool invoked again
       with byte-identical args. Each repeat after the first
       occurrence costs ``duplicate_penalty`` (default 0.25). A
       coherent agent rarely needs to ask the exact same question
       twice.
    2. **Error-retry loops** — a repeat of a call whose *earlier*
       identical invocation returned an error. Retrying a failed call
       verbatim (same args, no adaptation) costs an additional
       ``retry_penalty`` (default 0.25) on top of the duplicate
       penalty, because it signals the agent is not reading its
       errors.

    Score is ``1.0 - penalties``, clamped to ``[0.0, 1.0]``. Runs with
    fewer than two tool calls score 1.0 — there is nothing to be
    incoherent about. Legitimate patterns (polling a status endpoint,
    idempotent re-reads after a state change) WILL be penalised;
    tune the penalties or drop the metric for such agents.
    """

    name = "multi_step_coherence"

    def __init__(self, *, duplicate_penalty: float = 0.25, retry_penalty: float = 0.25) -> None:
        self.duplicate_penalty = duplicate_penalty
        self.retry_penalty = retry_penalty

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float:
        calls = tool_calls_from_events(events)
        if len(calls) < 2:
            return 1.0
        # call_id -> ok, from the results channel.
        ok_by_id: dict[str, bool] = {}
        for r in tool_results_from_events(events):
            call_id = r.get("call_id")
            if isinstance(call_id, str):
                ok_by_id[call_id] = bool(r.get("ok")) and not r.get("denied")
        penalty = 0.0
        seen: dict[str, list[str]] = {}  # signature -> call ids in order
        for call in calls:
            sig = _call_signature(call)
            earlier = seen.setdefault(sig, [])
            if earlier:
                penalty += self.duplicate_penalty
                # Error-retry: any earlier identical call known to have failed.
                if any(ok_by_id.get(cid) is False for cid in earlier):
                    penalty += self.retry_penalty
            call_id = call.get("id")
            earlier.append(call_id if isinstance(call_id, str) else "")
        return max(0.0, min(1.0, 1.0 - penalty))


class ExactMatch:
    """1.0 when the agent output equals ``case.expected`` exactly.

    Comparison is on ``str.strip()``-ed text (leading/trailing
    whitespace never counts against the agent). Cases without an
    ``expected`` string are skipped via :meth:`applies`.
    """

    name = "exact_match"

    def applies(self, case: Case) -> bool:
        return case.expected is not None

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float:
        expected = case.expected or ""
        return 1.0 if result.output.strip() == expected.strip() else 0.0


class Contains:
    """1.0 when ``case.expected`` appears as a substring of the output.

    ``case_sensitive=False`` folds both sides with ``str.casefold()``.
    Cases without an ``expected`` string are skipped via :meth:`applies`.
    """

    name = "contains"

    def __init__(self, *, case_sensitive: bool = True) -> None:
        self.case_sensitive = case_sensitive

    def applies(self, case: Case) -> bool:
        return case.expected is not None

    async def score(self, case: Case, result: RunResult, events: list[Event]) -> float:
        expected = case.expected or ""
        output = result.output
        if not self.case_sensitive:
            expected, output = expected.casefold(), output.casefold()
        return 1.0 if expected in output else 0.0
