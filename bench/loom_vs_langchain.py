"""Benchmark: Loom vs LangChain create_agent — wall-clock latency.

Both frameworks use the **same OpenAI model** (``gpt-4.1-mini``) via the
**same Python OpenAI SDK** under the hood, so any difference in
end-to-end time is framework overhead — not LLM inference cost.

We run each framework against three scenarios:

1. **Pure chat** — no tool call. Measures construction + prompt
   round-trip overhead.
2. **One tool call** — agent must call a tool, then synthesize an
   answer. Measures tool-dispatch overhead.
3. **Two tool calls in one turn** — measures parallel-dispatch
   handling.

Each scenario runs ``ITERATIONS`` times. Results report median
(stable under network jitter), min/max (outliers visible), and
tokens consumed (if the framework reports them).

Run::

    pip install langchain langchain-openai langchain-core
    OPENAI_API_KEY=sk-... python bench/loom_vs_langchain.py

Optional flags::

    --iterations N    # runs per scenario per framework (default 3)
    --warmup          # do one warm-up call per framework before timing
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Load OPENAI_API_KEY from .env if present.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print("\n  ✗ OPENAI_API_KEY required.\n")
    sys.exit(1)

MODEL_ID = "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Scenarios — same prompt, same expected behaviour, both frameworks
# ---------------------------------------------------------------------------

PROMPT_PURE_CHAT = "What is 2 + 2? Answer with just the number."
PROMPT_ONE_TOOL = (
    "What's the weather in Tokyo? Use the get_weather tool, then "
    "summarize the result in one short sentence."
)
PROMPT_TWO_TOOLS = (
    "What's the weather in BOTH Tokyo and Paris? Call get_weather "
    "twice (once for each city), then summarize."
)


# ---------------------------------------------------------------------------
# Loom setup
# ---------------------------------------------------------------------------


from loomflow import Agent as Loom  # noqa: E402
from loomflow import tool as loom_tool  # noqa: E402


@loom_tool(name="get_weather")
async def _loom_get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It's sunny and 72°F in {city}."


def _build_loom_agent() -> Loom:
    return Loom(
        "You are a helpful assistant. Be concise.",
        model=MODEL_ID,
        tools=[_loom_get_weather],
    )


async def _run_loom(agent: Loom, prompt: str) -> tuple[str, int, int]:
    """Run Loom end to end. Returns (output, tokens_in, tokens_out)."""
    result = await agent.run(prompt)
    return result.output, result.tokens_in, result.tokens_out


# ---------------------------------------------------------------------------
# LangChain create_agent setup
# ---------------------------------------------------------------------------


from langchain.agents import create_agent  # noqa: E402
from langchain_core.tools import tool as lc_tool  # noqa: E402


@lc_tool("get_weather")
def _lc_get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It's sunny and 72°F in {city}."


def _build_lc_agent() -> object:
    return create_agent(
        model=f"openai:{MODEL_ID}",
        tools=[_lc_get_weather],
        system_prompt="You are a helpful assistant. Be concise.",
    )


async def _run_lc(agent: object, prompt: str) -> tuple[str, int, int]:
    """Run LangChain agent end to end. Returns (output, tokens_in, tokens_out).

    Uses ``ainvoke`` so LangChain goes through the async OpenAI
    client (``AsyncOpenAI``), matching Loom's transport.
    Comparing ``invoke`` (sync httpx pool) against ``agent.run()``
    (async httpx pool) muddles framework overhead with
    transport-level differences in connection management.
    """
    result = await agent.ainvoke(  # type: ignore[attr-defined]
        {"messages": [{"role": "user", "content": prompt}]}
    )
    messages = result.get("messages", [])
    final = messages[-1] if messages else None
    output = getattr(final, "content", "") if final else ""

    # Sum usage across every AIMessage in the run (langchain only
    # populates usage_metadata on AI messages).
    tokens_in = 0
    tokens_out = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        tokens_in += int(usage.get("input_tokens", 0) or 0)
        tokens_out += int(usage.get("output_tokens", 0) or 0)
    return str(output), tokens_in, tokens_out


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    seconds: float
    tokens_in: int = 0
    tokens_out: int = 0
    output_len: int = 0


@dataclass
class ScenarioStats:
    name: str
    framework: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def median_s(self) -> float:
        return statistics.median(r.seconds for r in self.runs)

    @property
    def min_s(self) -> float:
        return min(r.seconds for r in self.runs)

    @property
    def max_s(self) -> float:
        return max(r.seconds for r in self.runs)

    @property
    def median_tokens_in(self) -> float:
        return statistics.median(r.tokens_in for r in self.runs)

    @property
    def median_tokens_out(self) -> float:
        return statistics.median(r.tokens_out for r in self.runs)


async def _time_loom(prompt: str) -> RunResult:
    agent = _build_loom_agent()
    t0 = time.perf_counter()
    output, t_in, t_out = await _run_loom(agent, prompt)
    elapsed = time.perf_counter() - t0
    return RunResult(
        seconds=elapsed,
        tokens_in=t_in,
        tokens_out=t_out,
        output_len=len(output),
    )


async def _time_lc(prompt: str) -> RunResult:
    agent = _build_lc_agent()
    t0 = time.perf_counter()
    output, t_in, t_out = await _run_lc(agent, prompt)
    elapsed = time.perf_counter() - t0
    return RunResult(
        seconds=elapsed,
        tokens_in=t_in,
        tokens_out=t_out,
        output_len=len(output),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bench Loom vs LangChain create_agent."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Runs per scenario per framework (default 3).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Do one warm-up call per framework before timing.",
    )
    args = parser.parse_args()

    scenarios = [
        ("pure_chat", PROMPT_PURE_CHAT),
        ("one_tool_call", PROMPT_ONE_TOOL),
        ("two_tool_calls", PROMPT_TWO_TOOLS),
    ]

    print()
    print("=" * 78)
    print(
        f"  Benchmarking Loom vs LangChain create_agent on {MODEL_ID}"
    )
    print(
        f"  iterations={args.iterations} warmup={args.warmup}"
    )
    print("=" * 78)

    if args.warmup:
        print("\nWarmup (1 call each, not timed)...")
        await _time_loom("Say hi.")
        await _time_lc("Say hi.")

    all_stats: list[ScenarioStats] = []

    for scenario_name, prompt in scenarios:
        print()
        print(f"── Scenario: {scenario_name} ──")
        print(f"   Prompt: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

        for framework_name, runner in [
            ("Loom", _time_loom),
            ("LangChain", _time_lc),
        ]:
            stats = ScenarioStats(name=scenario_name, framework=framework_name)
            for i in range(args.iterations):
                res = await runner(prompt)
                stats.runs.append(res)
                print(
                    f"   {framework_name:9s} run {i + 1}: "
                    f"{res.seconds * 1000:7.0f} ms  "
                    f"in={res.tokens_in:4d} out={res.tokens_out:4d}"
                )
            all_stats.append(stats)

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("  SUMMARY (medians)")
    print("=" * 78)
    print(
        f"{'Scenario':20s} {'Framework':12s} "
        f"{'median':>10s} {'min':>9s} {'max':>9s} "
        f"{'tok_in':>8s} {'tok_out':>9s}"
    )
    print("─" * 78)

    # Group by scenario for an easy-read comparison.
    by_scenario: dict[str, list[ScenarioStats]] = {}
    for s in all_stats:
        by_scenario.setdefault(s.name, []).append(s)

    for scenario_name, group in by_scenario.items():
        for s in group:
            print(
                f"{scenario_name:20s} {s.framework:12s} "
                f"{s.median_s * 1000:>8.0f} ms  "
                f"{s.min_s * 1000:>7.0f}ms "
                f"{s.max_s * 1000:>7.0f}ms "
                f"{s.median_tokens_in:>8.0f} {s.median_tokens_out:>9.0f}"
            )
        # Delta line
        if len(group) == 2:
            j, lc = (s for s in group if s.framework == "Loom"), (
                s for s in group if s.framework == "LangChain"
            )
            loom = next(j)
            langchain = next(lc)
            delta_ms = (langchain.median_s - loom.median_s) * 1000
            faster = "Loom" if delta_ms > 0 else "LangChain"
            pct = abs(delta_ms) / max(
                langchain.median_s * 1000, loom.median_s * 1000
            ) * 100
            print(
                f"{'':20s} {'Δ':12s} "
                f"{delta_ms:>+8.0f} ms  "
                f"({faster} faster by {pct:.1f}%)"
            )
        print()

    print("─" * 78)
    print(
        "Note: end-to-end wall-clock includes the OpenAI round-trip "
        "(typically 70-80% of the time). Differences here reflect "
        "framework overhead + how each framework structures the "
        "request, not LLM speed."
    )


if __name__ == "__main__":
    asyncio.run(main())
