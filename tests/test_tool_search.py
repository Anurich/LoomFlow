"""G1 — Tool Search / deferred tool loading tests.

Covers :mod:`loomflow.tools.search` (estimator, stub shapes, ranking,
the ``search_tools`` Tool) plus the end-to-end ReAct wiring: with 30
fat-schema tools and ``Tuning(tool_search=True)`` the first-request
tool block shrinks >70%, a stubbed tool still executes and hydrates
its full schema on the next turn, ``keep_tools`` stay full, and the
feature disabled is byte-identical to the pre-G1 def list.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, Tuning
from loomflow.core.types import Message, ToolCall, ToolDef, Usage
from loomflow.tools.registry import Tool
from loomflow.tools.search import (
    SEARCH_TOOL_NAME,
    estimate_tool_def_tokens,
    make_search_tools_tool,
    rank_tools,
    stub_defs,
)

pytestmark = pytest.mark.anyio

_STUB_SCHEMA = {"type": "object", "additionalProperties": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fat_tool(i: int, executed: list[tuple[str, dict[str, Any]]] | None = None) -> Tool:
    """A tool with a deliberately heavy schema + long description."""
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

    def _fn(**kwargs: Any) -> str:
        if executed is not None:
            executed.append((f"fat_tool_{i}", dict(kwargs)))
        return f"ran fat_tool_{i}"

    return Tool(
        name=f"fat_tool_{i}",
        description=(
            f"Fat tool number {i} that does specialised work. "
            + "It supports many modes and has extensive documentation. " * 6
        ),
        fn=_fn,
        input_schema={"type": "object", "properties": props, "required": []},
    )


def _defs(n: int = 30) -> list[ToolDef]:
    return [_fat_tool(i).to_def() for i in range(n)]


class RecordingModel:
    """Wraps a ScriptedModel and records the tool defs of every call."""

    name = "recording"

    def __init__(self, inner: ScriptedModel) -> None:
        self._inner = inner
        self.tools_per_call: list[list[ToolDef]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[ToolCall], Usage, str]:
        self.tools_per_call.append(list(tools or []))
        return await self._inner.complete(messages, tools=tools, **kwargs)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.tools_per_call.append(list(tools or []))
        return self._inner.stream(messages, tools=tools, **kwargs)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


def test_estimator_is_chars_over_four_of_name_description_schema() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    d = ToolDef(name="abcd", description="x" * 40, input_schema=schema)
    expected = (
        len("abcd") + 40 + len(json.dumps(schema, separators=(",", ":")))
    ) // 4
    assert estimate_tool_def_tokens([d]) == expected


def test_estimator_empty_schema_counts_zero_schema_chars() -> None:
    d = ToolDef(name="ab", description="cd")
    assert estimate_tool_def_tokens([d]) == 1  # 4 chars // 4


def test_estimator_grows_with_schema_size() -> None:
    small = ToolDef(name="t", description="d", input_schema={"type": "object"})
    big = _fat_tool(0).to_def()
    assert estimate_tool_def_tokens([big]) > estimate_tool_def_tokens([small])


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def test_stub_defs_shapes() -> None:
    defs = _defs(3)
    stubbed = stub_defs(defs, keep=["fat_tool_1"])
    by_name = {d.name: d for d in stubbed}
    # keep name preserved untouched (same object contents).
    assert by_name["fat_tool_1"] == defs[1]
    # Others: permissive marker schema + first-sentence description + hint.
    stub = by_name["fat_tool_0"]
    assert stub.input_schema == _STUB_SCHEMA
    assert stub.description.startswith(
        "Fat tool number 0 that does specialised work."
    )
    assert "full parameters load after first use" in stub.description
    # Order preserved.
    assert [d.name for d in stubbed] == [d.name for d in defs]


def test_stub_defs_preserves_destructive_and_server() -> None:
    d = ToolDef(
        name="rm",
        description="Delete things. Dangerous stuff.",
        input_schema={"type": "object", "properties": {"p": {"type": "string"}}},
        server="fs",
        destructive=True,
    )
    (stub,) = stub_defs([d], keep=())
    assert stub.destructive is True
    assert stub.server == "fs"


def test_stub_defs_never_stubs_search_tools_itself() -> None:
    search_def = make_search_tools_tool([]).to_def()
    (out,) = stub_defs([search_def], keep=())
    assert out == search_def


# ---------------------------------------------------------------------------
# Ranking + the search_tools Tool
# ---------------------------------------------------------------------------


def test_rank_tools_name_match_beats_description_match() -> None:
    defs = [
        ToolDef(name="send_email", description="Send an email to someone."),
        ToolDef(name="fetch_page", description="Fetch a web page, or email it."),
        ToolDef(name="unrelated", description="Nothing to see here."),
    ]
    matches = rank_tools(defs, "email")
    assert [m["name"] for m in matches] == ["send_email", "fetch_page"]
    # Name + one-liner only — no schema in match payloads.
    assert set(matches[0]) == {"name", "description"}


def test_rank_tools_no_match_returns_empty_and_limit_caps() -> None:
    defs = _defs(30)
    assert rank_tools(defs, "zzzzqqq") == []
    assert len(rank_tools(defs, "fat tool", limit=5)) == 5


async def test_search_tools_tool_executes_over_static_defs() -> None:
    tool = make_search_tools_tool(_defs(10))
    assert tool.name == SEARCH_TOOL_NAME
    out = json.loads(await tool.execute({"query": "fat_tool_7"}))
    assert out["matches"], "expected at least one match"
    assert out["matches"][0]["name"] == "fat_tool_7"


async def test_search_tools_tool_async_provider_and_self_exclusion() -> None:
    provider_defs = [*_defs(2), make_search_tools_tool([]).to_def()]

    async def _provider() -> list[ToolDef]:
        return provider_defs

    tool = make_search_tools_tool(_provider)
    out = json.loads(await tool.execute({"query": "search tools"}))
    assert all(m["name"] != SEARCH_TOOL_NAME for m in out["matches"])


async def test_search_tools_tool_no_match_carries_hint() -> None:
    tool = make_search_tools_tool(_defs(2))
    out = json.loads(await tool.execute({"query": "zzzzqqq"}))
    assert out["matches"] == []
    assert "hint" in out


# ---------------------------------------------------------------------------
# End-to-end via ReAct
# ---------------------------------------------------------------------------


async def test_e2e_enabled_shrinks_stub_executes_hydrates_keeps() -> None:
    executed: list[tuple[str, dict[str, Any]]] = []
    fat = [_fat_tool(i, executed) for i in range(30)]
    tools: list[Tool | Any] = list(fat)
    full_estimate = estimate_tool_def_tokens([t.to_def() for t in fat])

    model = RecordingModel(
        ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(tool="fat_tool_3", args={"param_0": "x"})
                    ]
                ),
                ScriptedTurn(text="done"),
            ]
        )
    )
    agent = Agent(
        "test",
        model=model,  # type: ignore[arg-type]
        tools=tools,
        tuning=Tuning(
            tool_search=True,
            tool_search_threshold_tokens=500,
            keep_tools=["fat_tool_0"],
        ),
    )
    result = await agent.run("go")
    assert result.output == "done"
    assert len(model.tools_per_call) == 2

    first = {d.name: d for d in model.tools_per_call[0]}
    # search_tools ships with its full def.
    assert SEARCH_TOOL_NAME in first
    assert "query" in first[SEARCH_TOOL_NAME].input_schema["properties"]
    # keep_tools stay full; everything else is stubbed on turn 1.
    assert "properties" in first["fat_tool_0"].input_schema
    assert first["fat_tool_3"].input_schema == _STUB_SCHEMA
    # >70% token drop on the first-request tool block.
    first_estimate = estimate_tool_def_tokens(model.tools_per_call[0])
    assert first_estimate < 0.3 * full_estimate

    # The stubbed tool still executed (server-side dispatch hits the
    # REAL tool regardless of the schema the model saw).
    assert ("fat_tool_3", {"param_0": "x"}) in executed

    # Hydration: the called tool ships its FULL schema on turn 2;
    # untouched tools remain stubs.
    second = {d.name: d for d in model.tools_per_call[1]}
    assert "properties" in second["fat_tool_3"].input_schema
    assert second["fat_tool_5"].input_schema == _STUB_SCHEMA


async def test_e2e_below_threshold_ships_full_defs() -> None:
    tools: list[Tool | Any] = [_fat_tool(i) for i in range(3)]
    model = RecordingModel(ScriptedModel([ScriptedTurn(text="done")]))
    agent = Agent(
        "test",
        model=model,  # type: ignore[arg-type]
        tools=tools,
        tuning=Tuning(tool_search=True, tool_search_threshold_tokens=10_000_000),
    )
    await agent.run("go")
    names = {d.name for d in model.tools_per_call[0]}
    # search_tools is installed, but nothing is stubbed.
    assert SEARCH_TOOL_NAME in names
    for d in model.tools_per_call[0]:
        if d.name != SEARCH_TOOL_NAME:
            assert "properties" in d.input_schema


async def test_e2e_disabled_is_identical_to_full_defs() -> None:
    fat = [_fat_tool(i) for i in range(30)]
    tools: list[Tool | Any] = list(fat)
    model = RecordingModel(ScriptedModel([ScriptedTurn(text="done")]))
    agent = Agent("test", model=model, tools=tools)  # type: ignore[arg-type]
    await agent.run("go")
    assert model.tools_per_call[0] == [t.to_def() for t in fat]


def test_tuning_defaults_off() -> None:
    t = Tuning()
    assert t.tool_search is False
    assert t.tool_search_threshold_tokens == 10_000
    assert tuple(t.keep_tools) == ()
