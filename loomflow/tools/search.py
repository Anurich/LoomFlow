"""Tool Search / deferred tool loading (G1).

When an agent carries dozens of tools (multiple MCP servers, big
in-process registries), shipping every full JSON schema on every model
call costs tens of thousands of tokens *and lowers accuracy*. This
module implements progressive disclosure for tool definitions —
mirroring the pattern the skills system already uses for playbooks:

* :func:`estimate_tool_def_tokens` — cheap chars/4 heuristic over
  name + description + schema JSON, used by the ReAct loop to decide
  whether the tool block is heavy enough to bother stubbing.
* :func:`stub_defs` — reduce a def list to "stubs": name + one-line
  description + a minimal permissive object schema. A stub stays
  *callable* — the real tool's own arg coercion / validation still
  runs server-side on dispatch, so correctness is preserved; the
  full schema simply isn't paid for until the tool proves useful.
* :func:`make_search_tools_tool` — a local ``search_tools(query)``
  tool the model uses to browse the catalogue by keyword (substring
  + keyword scoring; deliberately no embeddings — deterministic,
  zero-dependency, fast).

The ReAct loop (see :mod:`loomflow.architecture.react`) keeps a
per-session ``hydrated`` set: once the model calls a stubbed tool,
its FULL definition ships on every subsequent turn.

Everything here is opt-in via ``Tuning(tool_search=True)`` — the
default path is byte-identical to pre-G1 behaviour.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import Any

from ..core.types import ToolDef
from .registry import Tool

SEARCH_TOOL_NAME = "search_tools"
"""Name of the auto-injected catalogue-search tool. Always shipped
with its full definition (never stubbed) when tool search is active."""

_STUB_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}
"""Minimal permissive schema stubs carry — valid JSON Schema, so a
stubbed tool remains directly callable by any provider. Server-side
dispatch validates/coerces against the REAL tool's schema regardless."""

_STUB_HINT = (
    " (call search_tools or call directly; full parameters load after first use)"
)

_SENTENCE_END = re.compile(r"(.+?[.!?])(?:\s|$)")


def _first_sentence(description: str) -> str:
    """First sentence (or first line, whichever ends sooner) of a
    description — the "one-liner" tier of progressive disclosure."""
    first_line = description.strip().split("\n", 1)[0].strip()
    m = _SENTENCE_END.match(first_line)
    return m.group(1).strip() if m else first_line


def estimate_tool_def_tokens(defs: Sequence[ToolDef]) -> int:
    """Estimate the token weight of a tool-def block.

    Chars/4 heuristic over ``name + description + json(input_schema)``
    per def — the same cheap estimator used elsewhere in the framework
    (tool-result caps, memory budgeting). Deliberately provider-
    agnostic: this gates a *threshold* decision, not billing.
    """
    total_chars = 0
    for d in defs:
        total_chars += len(d.name) + len(d.description)
        if d.input_schema:
            total_chars += len(json.dumps(d.input_schema, separators=(",", ":")))
    return total_chars // 4


def stub_defs(
    defs: Sequence[ToolDef], keep: Collection[str]
) -> list[ToolDef]:
    """Replace every def NOT named in ``keep`` with a callable stub.

    A stub keeps the tool's name (so the model can still call it
    directly), truncates the description to its first sentence plus a
    hint about hydration, and swaps the input schema for a minimal
    permissive ``{"type": "object", "additionalProperties": true}``.
    ``destructive`` and ``server`` are preserved — the approval-gate
    backstop and MCP attribution must not change under stubbing.

    ``keep`` names (plus :data:`SEARCH_TOOL_NAME`, always) pass
    through untouched. Order is preserved.
    """
    keep_set = set(keep) | {SEARCH_TOOL_NAME}
    out: list[ToolDef] = []
    for d in defs:
        if d.name in keep_set:
            out.append(d)
            continue
        out.append(
            ToolDef(
                name=d.name,
                description=_first_sentence(d.description) + _STUB_HINT,
                input_schema=dict(_STUB_SCHEMA),
                server=d.server,
                destructive=d.destructive,
            )
        )
    return out


def rank_tools(
    defs: Sequence[ToolDef], query: str, *, limit: int = 10
) -> list[dict[str, str]]:
    """Rank tool defs against ``query`` — substring + keyword scoring.

    Scoring (deliberately simple, no embeddings):

    * whole query == / substring-of the tool NAME — strongest signal;
    * whole query substring of the description — medium;
    * per keyword: exact-name > name-substring > description hit.

    Only positive-score defs are returned, best first (name as the
    deterministic tie-break), capped at ``limit``. Each match is
    ``{"name": ..., "description": <first sentence>}`` — the cheap
    tier only; full parameters hydrate when the tool is called.
    """
    q = query.lower().strip()
    terms = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    scored: list[tuple[float, str, str]] = []
    for d in defs:
        name = d.name.lower()
        desc = d.description.lower()
        score = 0.0
        if q:
            if q == name:
                score += 20.0
            elif q in name:
                score += 10.0
            if q in desc:
                score += 3.0
        for t in terms:
            if t == name:
                score += 8.0
            elif t in name:
                score += 4.0
            if t in desc:
                score += 1.0
        if score > 0:
            scored.append((score, d.name, _first_sentence(d.description)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"name": name, "description": desc}
        for _score, name, desc in scored[:limit]
    ]


DefsSource = Sequence[ToolDef] | Callable[[], Awaitable[Sequence[ToolDef]]]
"""What :func:`make_search_tools_tool` searches over: either a static
def list (tests, fixed registries) or an async zero-arg provider that
re-lists the live host on every search (MCP hosts whose catalogue can
change mid-run)."""


def make_search_tools_tool(defs: DefsSource) -> Tool:
    """Build the local ``search_tools`` :class:`Tool`.

    Executes entirely in-process (no model, no network): ranks the
    current catalogue against the query via :func:`rank_tools` and
    returns a JSON payload of ``{name, description}`` matches. The
    tool never lists itself.
    """

    async def _search_tools(query: str, limit: int = 10) -> str:
        if callable(defs):
            current = list(await defs())
        else:
            current = list(defs)
        current = [d for d in current if d.name != SEARCH_TOOL_NAME]
        matches = rank_tools(current, query, limit=limit)
        payload: dict[str, Any] = {"matches": matches}
        if not matches:
            payload["hint"] = (
                "no tools matched; try broader or different keywords"
            )
        return json.dumps(payload)

    return Tool(
        name=SEARCH_TOOL_NAME,
        description=(
            "Search the tool catalogue by keyword. Returns ranked "
            "{name, description} matches. Any listed tool can be "
            "called directly by name — its full parameter schema "
            "loads automatically after the first call."
        ),
        fn=_search_tools,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords describing the capability you need."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    )


__all__ = [
    "SEARCH_TOOL_NAME",
    "DefsSource",
    "estimate_tool_def_tokens",
    "make_search_tools_tool",
    "rank_tools",
    "stub_defs",
]
