"""loomflow.eval — offline + online agent evaluation (G9).

Tier-2 submodule: import from ``loomflow.eval`` explicitly (nothing is
re-exported from the top-level ``loomflow`` package)::

    from loomflow.eval import (
        Case, Dataset, EvalHarness, ToolSelectionAccuracy, LLMJudge,
    )

    h = EvalHarness(agent, metrics=[ToolSelectionAccuracy(), LLMJudge(model=judge_model)])
    report = await h.run(dataset)                  # offline: fixed cases
    report.assert_thresholds({"tool_selection_accuracy": 0.9})

    scorer = h.online(sample_rate=0.1, seed=7)     # online: sample live traffic
    async with scorer:
        ...  # after each live run: scorer.submit(prompt, result, events)
    print(scorer.rollup())

Everything runs offline against in-process fakes
(:class:`~loomflow.ScriptedModel` / :class:`~loomflow.EchoModel`);
no network is required to eval or to test the eval.
"""

from .dataset import Case, Dataset
from .harness import EVAL_USER_ID, CaseResult, EvalHarness, EvalReport, OnlineScorer
from .judge import LLMJudge
from .metrics import (
    Contains,
    ExactMatch,
    Metric,
    MultiStepCoherence,
    ToolExecutionSuccess,
    ToolSelectionAccuracy,
    tool_calls_from_events,
    tool_results_from_events,
)

__all__ = [
    "EVAL_USER_ID",
    "Case",
    "CaseResult",
    "Contains",
    "Dataset",
    "EvalHarness",
    "EvalReport",
    "ExactMatch",
    "LLMJudge",
    "Metric",
    "MultiStepCoherence",
    "OnlineScorer",
    "ToolExecutionSuccess",
    "ToolSelectionAccuracy",
    "tool_calls_from_events",
    "tool_results_from_events",
]
