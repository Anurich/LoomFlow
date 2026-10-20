"""MCP interop against a real server built with the installed ``mcp`` SDK.

Fake sessions accept whatever attribute names the code reads, so they
could not catch mcp 2.x renaming every protocol field to snake_case.
These tests run a real stdio server and the real client stack.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
if int(importlib.metadata.version("mcp").split(".")[0]) < 2:
    pytest.skip("the test server uses the mcp 2.x API", allow_module_level=True)

from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec  # noqa: E402

pytestmark = pytest.mark.anyio

SERVER = str(Path(__file__).with_name("_mcp_sdk2_server.py"))


def _sampling(messages: Any, preferences: Any) -> str:
    hints = [h.name for h in (preferences.hints or [])] if preferences else []
    return f"HELLO|hints={','.join(hints)}"


async def _registry() -> MCPRegistry:
    spec = MCPServerSpec.stdio("srv", sys.executable, [SERVER], sampling_handler=_sampling)
    return MCPRegistry([MCPClient(spec)])


async def test_real_server_fields_survive_the_sdk_rename() -> None:
    reg = await _registry()
    try:
        defs = {d.name: d for d in await reg.list_tools()}
        # Page 2 of tools/list is followed (nextCursor).
        assert {"echo", "fail", "rich", "paged_tool", "sample"} <= set(defs)
        assert defs["echo"].input_schema["required"] == ["msg"]
        assert defs["fail"].destructive is True

        failed = await reg.call("fail", {}, call_id="c1")
        assert not failed.ok and "tool exploded" in (failed.error or "")

        rich = await reg.call("rich", {}, call_id="c2")
        assert rich.ok
        assert "[image: image/png" in rich.output
        assert "EMBEDDED-TEXT-PAYLOAD" in rich.output
        assert "[resource_link: file:///big.log]" in rich.output

        sampled = await reg.call("sample", {}, call_id="c3")
        assert sampled.ok and "hints=claude" in sampled.output
    finally:
        await reg.aclose()


async def test_disambiguated_names_are_provider_safe() -> None:
    a = MCPServerSpec.stdio("city.api", sys.executable, [SERVER], sampling_handler=_sampling)
    b = MCPServerSpec.stdio("noaa", sys.executable, [SERVER], sampling_handler=_sampling)
    reg = MCPRegistry([MCPClient(a), MCPClient(b)])
    try:
        names = [d.name for d in await reg.list_tools()]
        assert "noaa__echo" in names
        assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", n) for n in names), names
        # The dotted form from earlier releases still routes.
        result = await reg.call("noaa.echo", {"msg": "hi"}, call_id="c")
        assert result.ok and result.output == "echo:hi"
    finally:
        await reg.aclose()
