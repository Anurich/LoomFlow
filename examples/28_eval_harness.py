"""28_eval_harness.py — the eval harness: datasets, metrics, judge,
threshold gates.

``loomflow.eval`` turns "does my agent still work?" into a CI gate::

    dataset = Dataset([Case(input=..., expected=..., expected_tools=[...])])
    harness = EvalHarness(agent, metrics=[ExactMatch(), ToolSelectionAccuracy(),
                                          LLMJudge(judge_model)])
    report  = await harness.run(dataset)
    report.assert_thresholds({"exact_match": 0.9})   # raises on regression

The pieces:

* :class:`Case` / :class:`Dataset` — ground truth with a JSONL
  round-trip (``to_jsonl`` / ``from_jsonl``) so your golden set lives
  in the repo.
* Deterministic metrics — :class:`ExactMatch`, :class:`Contains`,
  :class:`ToolSelectionAccuracy` (did it call the right tools?),
  ``ToolExecutionSuccess``, ``MultiStepCoherence``. Each metric only
  scores cases it *applies* to (no expected → no exact-match score).
* :class:`LLMJudge` — LLM-as-judge that demands an explicit
  ``score: <0-1>`` line (prose numbers rejected), retries once, falls
  back to a NEUTRAL 0.5 on parse failure, and warns if you grade a
  model with itself.
* :meth:`EvalReport.assert_thresholds` — mean-per-metric gate that
  lists every failure, which is the line you put in CI.
* ``harness.online(sample_rate=...)`` — the same metric suite scoring
  a sample of LIVE traffic (not shown here).

Runs OFFLINE with :class:`ScriptedModel` for both the agent under test
and the judge (no API key).

Run with::

    python examples/28_eval_harness.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anyio

from loomflow import Agent, ScriptedModel, ScriptedTurn, ToolCall, tool
from loomflow.eval import (
    Case,
    Dataset,
    EvalHarness,
    ExactMatch,
    LLMJudge,
    ToolSelectionAccuracy,
)


@tool
async def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"{city}: 21C, sunny"


async def main() -> None:
    # ---- 1. A golden dataset, round-tripped through JSONL ------------
    print("=" * 64)
    print("Part 1 — Dataset + JSONL round-trip")
    print("=" * 64)

    dataset = Dataset(
        [
            Case(input="What is 2+2? Digits only.", expected="4",
                 metadata={"topic": "math"}),
            Case(input="Capital of France? One word.", expected="Paris"),
            Case(input="Fetch the weather in Oslo.",
                 expected_tools=["get_weather"]),
            Case(input="Reply exactly: SHIP IT", expected="SHIP IT"),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "golden.jsonl"
        dataset.to_jsonl(path)
        loaded = Dataset.from_jsonl(path)
        print(f"  wrote {len(dataset)} cases → {path.name}; "
              f"reloaded {len(loaded)} — round-trip OK")

    # ---- 2. The agent under test (scripted, deterministic) -----------
    # concurrency=1 keeps the scripted turns aligned with case order.
    # Case 4 deliberately answers in lowercase → ExactMatch misses.
    agent = Agent(
        "You answer briefly and use tools when asked.",
        model=ScriptedModel(
            [
                ScriptedTurn(text="4"),
                ScriptedTurn(text="Paris"),
                ScriptedTurn(tool_calls=[ToolCall(tool="get_weather",
                                                  args={"city": "Oslo"})]),
                ScriptedTurn(text="Oslo is 21C and sunny."),
                ScriptedTurn(text="ship it"),
            ]
        ),
        tools=[get_weather],
    )

    # A SEPARATE scripted judge (same-model judging warns — self-
    # preference bias). One "score:" line per case, in case order.
    judge = LLMJudge(
        ScriptedModel(
            [
                ScriptedTurn(text="Correct and minimal.\nscore: 1.0"),
                ScriptedTurn(text="Correct.\nscore: 1.0"),
                ScriptedTurn(text="Right tool, clear summary.\nscore: 0.9"),
                ScriptedTurn(text="Wrong casing; instruction ignored.\nscore: 0.3"),
            ]
        )
    )

    harness = EvalHarness(
        agent,
        metrics=[ExactMatch(), ToolSelectionAccuracy(), judge],
        concurrency=1,
    )

    print()
    print("=" * 64)
    print("Part 2 — run the harness")
    print("=" * 64)
    report = await harness.run(loaded)

    for r in report.results:
        print(f"  {r.input[:34]:<36} → {r.output[:28]!r:<32} {r.scores}")
    print("  summary:")
    for metric, stats in report.summary().items():
        print(f"    {metric:<24} mean={stats['mean']:.2f}  "
              f"min={stats['min']:.2f}  n={stats['count']:.0f}")

    # ---- 3. Threshold gates: the CI line ------------------------------
    print()
    print("=" * 64)
    print("Part 3 — assert_thresholds")
    print("=" * 64)

    report.assert_thresholds(
        {"exact_match": 0.6, "tool_selection_accuracy": 1.0, "llm_judge": 0.7}
    )
    print("  gate {exact_match: 0.6, tool_selection_accuracy: 1.0,")
    print("        llm_judge: 0.7} → PASSED")

    try:
        report.assert_thresholds({"exact_match": 0.9})
    except AssertionError as exc:
        print("  gate {exact_match: 0.9} → FAILED as expected:")
        print(f"    {exc}")
    print("  → Wire report.assert_thresholds(...) into CI and an agent")
    print("    regression fails the build with the exact metric that slipped.")


if __name__ == "__main__":
    anyio.run(main)
