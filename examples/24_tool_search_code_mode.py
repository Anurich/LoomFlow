"""24_tool_search_code_mode.py — token-lean tool catalogues: tool
search + code mode.

Two answers to the same scaling problem. An agent wired to a couple of
MCP servers easily carries 30+ tools; shipping every full JSON schema
on every model call costs tens of thousands of tokens *and lowers
accuracy* (the model has to wade through 29 irrelevant schemas to find
the one it needs).

**Part 1 — tool search / deferred loading.** Progressive disclosure
for tool definitions. When the tool block is heavier than a threshold,
every def is reduced to a callable *stub* (name + one-line description
+ permissive schema) and a local ``search_tools(query)`` tool is
injected so the model can browse the catalogue. A stubbed tool is
still directly callable — dispatch validates against the REAL schema —
and once called, its full definition ships on every subsequent turn
(hydration). ``keep_tools`` pins hot tools at full fidelity::

    Agent(..., tuning=Tuning(
        tool_search=True,
        tool_search_threshold_tokens=500,   # default 10_000
        keep_tools=["fat_tool_0"],          # never stubbed
    ))

**Part 2 — code mode.** Instead of N schemas, register exactly two
tools via ``make_code_mode_tools(...)``: ``search_api(query)`` returns
typed Python signature stubs, and ``run_code(code)`` executes the
model's Python with every tool bound as a real ``async`` callable.
Intermediate tool results stay OUT of the model's context — only what
the code assigns to ``result`` (or prints) comes back. That kills the
50k-token intermediate-result problem: the model writes
``rows = await fetch(); result = sum(...)`` and sees a number, not the
rows.

Runs OFFLINE with :class:`ScriptedModel` (no API key).

Run with::

    python examples/24_tool_search_code_mode.py
"""

from __future__ import annotations

import json
from typing import Any

import anyio

from loomflow import Agent, ScriptedModel, ScriptedTurn, Tool, ToolCall, Tuning, tool
from loomflow.tools.code_mode import make_code_mode_tools
from loomflow.tools.search import SEARCH_TOOL_NAME, estimate_tool_def_tokens

# ---------------------------------------------------------------------------
# Part 1 fixtures — 30 tools with deliberately fat schemas.
# ---------------------------------------------------------------------------


def _fat_tool(i: int) -> Tool:
    """A tool with a heavy 8-parameter schema and a long description."""
    props = {
        f"param_{j}": {
            "type": "string",
            "description": (
                f"Detailed parameter {j} for tool {i}. "
                + "It controls a very specific aspect of the operation. " * 4
            ),
        }
        for j in range(8)
    }
    return Tool(
        name=f"fat_tool_{i}",
        description=(
            f"Fat tool number {i} that does specialised work. "
            + "It supports many modes and has extensive documentation. " * 6
        ),
        fn=lambda **kwargs: f"ran fat_tool_{i}",
        input_schema={"type": "object", "properties": props, "required": []},
    )


class RecordingModel:
    """Wraps a ScriptedModel and records the tool defs of every call —
    so we can SHOW what the model actually paid for."""

    name = "recording"

    def __init__(self, inner: ScriptedModel) -> None:
        self._inner = inner
        self.tools_per_call: list[list[Any]] = []

    async def complete(self, messages: Any, *, tools: Any = None, **kw: Any) -> Any:
        self.tools_per_call.append(list(tools or []))
        return await self._inner.complete(messages, tools=tools, **kw)


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def part_1_tool_search() -> None:
    banner("Part 1 — Tuning(tool_search=True): stubs, search, hydration")

    fat = [_fat_tool(i) for i in range(30)]
    full_estimate = estimate_tool_def_tokens([t.to_def() for t in fat])

    # The scripted model calls a STUBBED tool directly on turn 1 —
    # proving stubs stay callable — then finishes on turn 2.
    model = RecordingModel(
        ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[ToolCall(tool="fat_tool_3", args={"param_0": "x"})]
                ),
                ScriptedTurn(text="done"),
            ]
        )
    )
    agent = Agent(
        "You are an operator with a large tool catalogue.",
        model=model,  # type: ignore[arg-type]
        tools=list(fat),
        tuning=Tuning(
            tool_search=True,
            tool_search_threshold_tokens=500,  # our block is way heavier
            keep_tools=["fat_tool_0"],  # pinned: always full schema
        ),
    )
    result = await agent.run("Run tool 3 for me.")
    print(f"  run output:                 {result.output!r}")

    first_estimate = estimate_tool_def_tokens(model.tools_per_call[0])
    print(f"  full tool block:           ~{full_estimate:,} tokens (30 fat tools)")
    print(f"  first-request tool block:  ~{first_estimate:,} tokens (stubbed)")
    print(f"  saved:                      {1 - first_estimate / full_estimate:.0%}")

    first = {d.name: d for d in model.tools_per_call[0]}
    second = {d.name: d for d in model.tools_per_call[1]}
    print(f"  search tool injected:       {SEARCH_TOOL_NAME in first}")
    print(f"  fat_tool_0 (keep_tools):    full schema on turn 1 = "
          f"{'properties' in first['fat_tool_0'].input_schema}")
    print(f"  fat_tool_3 stub, turn 1:    {first['fat_tool_3'].input_schema}")
    print(f"  fat_tool_3 hydrated, turn2: full schema = "
          f"{'properties' in second['fat_tool_3'].input_schema}")
    print(f"  fat_tool_5 untouched:       still a stub = "
          f"{'properties' not in second['fat_tool_5'].input_schema}")
    print("  → The stubbed tool still EXECUTED (dispatch validates against")
    print("    the real schema); only the tokens for unused schemas were saved.")


# ---------------------------------------------------------------------------
# Part 2 — code mode: two tools instead of N, results filtered in code.
# ---------------------------------------------------------------------------


@tool
async def fetch_sales() -> str:
    """Fetch the raw quarterly sales ledger as a JSON array."""
    rows = [{"region": f"r{i % 7}", "amount": (i * 37) % 500} for i in range(2000)]
    return json.dumps(rows)


@tool
async def get_fx_rate(currency: str) -> float:
    """Look up the current FX rate for a currency against USD."""
    return {"EUR": 1.08, "GBP": 1.27}.get(currency, 1.0)


async def part_2_code_mode() -> None:
    banner("Part 2 — code mode: search_api + run_code (two-tool pattern)")

    search_api, run_code = make_code_mode_tools([fetch_sales, get_fx_rate])

    # 1. The model discovers the API by keyword — signatures, no schemas.
    stubs = await search_api.execute({"query": "sales ledger"})
    print("  search_api('sales ledger') →")
    for line in stubs.splitlines()[:6]:
        print(f"    {line}")

    # 2. The model computes over the big payload IN CODE. The raw
    #    ledger never enters its context — only the aggregate does.
    raw = await fetch_sales.execute({})
    code = (
        "import json\n"
        "rows = json.loads(await fetch_sales())\n"
        "rate = await get_fx_rate(currency='EUR')\n"
        "result = round(sum(r['amount'] for r in rows) * rate, 2)\n"
    )
    out = await run_code.execute({"code": code})
    print(f"\n  raw fetch_sales() payload:  {len(raw):,} chars (never sent to model)")
    print(f"  run_code result:            {out!r} ({len(out)} chars)")
    print("  → The model saw one small number instead of a 2000-row ledger.")
    print("    (Pass executor=SubprocessExecutor() for out-of-process,")
    print("    tool-free data crunching with a hard timeout.)")


async def main() -> None:
    await part_1_tool_search()
    await part_2_code_mode()
    print()


if __name__ == "__main__":
    anyio.run(main)
