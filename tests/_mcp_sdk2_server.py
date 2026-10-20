"""Real stdio MCP server (official ``mcp`` 2.x low-level API) for the interop tests
in ``test_validation_protocols.py``.

- tools/list is paginated: page 1 = [echo, fail, rich], page 2 (cursor "p2") = [paged_tool, notify, sample]
- echo: text content; input schema with required 'msg'
- fail: isError=True
- rich: image + embedded text resource + resource_link + structuredContent
- notify: sends notifications/tools/list_changed and then adds 'late_tool'
- sample: asks the client for sampling/createMessage and returns the text
- annotations: 'fail' is destructiveHint=True
"""
from __future__ import annotations

import os
import warnings

import anyio
import mcp.types as t
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server

warnings.simplefilter("ignore")
EXTRA = {"added": False}

PAGE1 = [
    t.Tool(name="echo", description="Echo a message",
           input_schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}),
    t.Tool(name="fail", description="Always fails", input_schema={"type": "object"},
           annotations=t.ToolAnnotations(destructive_hint=True)),
    t.Tool(name="rich", description="Mixed content", input_schema={"type": "object"}),
]
PAGE2 = [
    t.Tool(name="paged_tool", description="Only on page 2", input_schema={"type": "object"}),
    t.Tool(name="notify", description="Triggers list_changed", input_schema={"type": "object"}),
    t.Tool(name="sample", description="Uses sampling", input_schema={"type": "object"}),
]


NOPAGE = os.environ.get("NOPAGE") == "1"


async def list_tools(ctx, params):
    cursor = getattr(params, "cursor", None) if params else None
    if NOPAGE:
        tools = list(PAGE1) + list(PAGE2) + ([t.Tool(name="late_tool", description="late", input_schema={"type": "object"})]
                                             if EXTRA["added"] else [])
        return t.ListToolsResult(tools=tools, next_cursor=None)
    if cursor == "p2":
        tools = list(PAGE2) + ([t.Tool(name="late_tool", description="late", input_schema={"type": "object"})]
                               if EXTRA["added"] else [])
        return t.ListToolsResult(tools=tools, next_cursor=None)
    return t.ListToolsResult(tools=PAGE1, next_cursor="p2")


async def call_tool(ctx, params):
    name = params.name
    if name == "echo":
        return t.CallToolResult(content=[t.TextContent(type="text", text=f"echo:{(params.arguments or {}).get('msg')}")])
    if name == "fail":
        return t.CallToolResult(content=[t.TextContent(type="text", text="tool exploded")], is_error=True)
    if name == "rich":
        return t.CallToolResult(content=[
            t.ImageContent(type="image", data="iVBORw0KGgo=", mime_type="image/png"),
            t.EmbeddedResource(type="resource", resource=t.TextResourceContents(
                uri="file:///notes.txt", mime_type="text/plain", text="EMBEDDED-TEXT-PAYLOAD")),
            t.ResourceLink(type="resource_link", uri="file:///big.log", name="big.log", mime_type="text/plain"),
        ])
    if name == "notify":
        EXTRA["added"] = True
        await ctx.session.send_tool_list_changed()
        return t.CallToolResult(content=[t.TextContent(type="text", text="notified")])
    if name == "sample":
        res = await ctx.session.create_message(
            messages=[t.SamplingMessage(role="user", content=t.TextContent(type="text", text="say hi"))],
            max_tokens=10,
            model_preferences=t.ModelPreferences(hints=[t.ModelHint(name="claude")]),
        )
        return t.CallToolResult(content=[t.TextContent(type="text", text=f"sampled:{res.content.text}|model={res.model}")])
    if name == "paged_tool":
        return t.CallToolResult(content=[t.TextContent(type="text", text="page2 ok")])
    raise ValueError(f"unknown tool {name}")


server = Server("interop-server", on_list_tools=list_tools, on_call_tool=call_tool)


async def main() -> None:
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options(
            notification_options=NotificationOptions(tools_changed=True)))


if __name__ == "__main__":
    anyio.run(main)
