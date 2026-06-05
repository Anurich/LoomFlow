"""22 — Lazy tool loading: cut per-turn tool-schema tokens.

A coding agent often carries 15-20 tools. Every request normally ships
*every* tool's full JSON schema in the ``tools`` field — re-sent each
turn. ``Tuning(lazy_tools=True)`` sends only an ``expand_tool`` meta-tool
plus a compact catalog (in the cached system prompt); the model calls
``expand_tool(name)`` to see a tool's arguments on demand.

Crucially, the exposed tool list stays byte-stable across turns, so the
prompt-cache tool breakpoint is preserved — you get the schema-token
savings WITHOUT thrashing the cache.

Run: ``python examples/22_lazy_tools.py`` (offline — uses EchoModel).
"""

import asyncio
import json

from loomflow import Agent, Tuning
from loomflow.model.echo import EchoModel
from loomflow.tools import tool

TOOL_SPECS = [
    ("read_file", "Read a file from disk and return its contents."),
    ("write_file", "Write content to a file, creating directories as needed."),
    ("edit_file", "Apply a surgical search/replace edit to a file."),
    ("bash", "Run a shell command and return stdout/stderr."),
    ("grep", "Search file contents by regex across the tree."),
    ("glob", "Find files by glob pattern."),
    ("ls", "List a directory."),
    ("web_fetch", "Fetch a URL and return readable text."),
    ("codebase_search", "Semantic search over the indexed codebase."),
    ("go_to_definition", "LSP: jump to a symbol's definition."),
    ("find_references", "LSP: find all references to a symbol."),
    ("run_tests", "Run the project's test suite and report failures."),
]


def _make_tools() -> list:
    tools = []
    for name, desc in TOOL_SPECS:
        def factory(n: str, d: str):
            @tool(name=n, description=d)
            def _f(path: str = "", pattern: str = "", command: str = "") -> str:
                return "ok"
            return _f
        tools.append(factory(name, desc))
    return tools


def _schema_tokens(defs) -> int:
    blob = json.dumps(
        [{"name": d.name, "description": d.description, "input_schema": d.input_schema}
         for d in defs]
    )
    return len(blob) // 4  # ~4 chars/token proxy


async def main() -> None:
    tools = _make_tools()
    off = Agent("You help.", model=EchoModel(), tools=tools)
    on = Agent("You help.", model=EchoModel(), tools=tools, tuning=Tuning(lazy_tools=True))

    off_defs = await off._tool_host.list_tools()
    on_defs = await on._tool_host.list_tools()

    off_tok = _schema_tokens(off_defs)
    on_tok = _schema_tokens(on_defs)

    print(f"Roster: {len(tools)} tools\n")
    print(f"lazy OFF — tools array sent EVERY turn : {off_tok} tokens")
    print(f"lazy ON  — tools array sent EVERY turn : {on_tok} tokens "
          f"({[d.name for d in on_defs]})")
    print(f"\nPer-turn tools-array reduction: "
          f"{100 * (off_tok - on_tok) / off_tok:.0f}%")
    print("\nThe full schemas move into expand_tool (on demand) + a compact")
    print("catalog in the cached system prompt — the cache breakpoint holds.")


if __name__ == "__main__":
    asyncio.run(main())
