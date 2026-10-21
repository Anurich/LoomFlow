"""What Studio records about an agent or workflow: its definition.

A definition is the part of an agent that changes its behaviour —
instructions, model, tools, architecture, limits, guards, and its
structure (sub-agents, routes, workflow graph, as drawn by
:mod:`loomflow.graph`) — gathered from the running object, so nothing
has to be declared. Its
fingerprint (a hash of the definition) identifies a version: the
store gives each new fingerprint the next version number.

An agent's catalog key is its ``name=``. Unnamed agents are keyed by
the first line of their instructions, so editing that line starts a
new catalog entry; give agents a name to keep their history together.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _fingerprint(definition: dict[str, Any]) -> str:
    canonical = json.dumps(definition, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def agent_identity(agent: Any) -> tuple[str, str]:
    """``(catalog key, display name)`` for an Agent."""
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name:
        return name, name
    line = _first_line(getattr(agent, "_instructions", "") or "")
    label = line if len(line) <= 48 else line[:47].rstrip() + "…"
    digest = hashlib.sha1(line.encode("utf-8")).hexdigest()[:8]
    return f"agent:{digest}", label or "agent"


def _permissions(permissions: Any) -> str:
    kind = type(permissions).__name__
    mode = getattr(permissions, "_mode", None) or getattr(permissions, "mode", None)
    mode_name = getattr(mode, "value", None) or getattr(mode, "name", None)
    return f"{kind} ({mode_name})" if mode_name else kind


async def structure(target: Any, title: str) -> dict[str, Any] | None:
    """The agent's or workflow's structure as plain data (nodes keyed
    by the names runs record), or ``None`` if it can't be walked."""
    from ..graph import build_graph

    try:
        graph = await build_graph(target, title=title, key_for=lambda agent: agent_identity(agent)[0])
    except Exception:  # noqa: BLE001 — a diagram must never break a run
        return None
    return graph.to_dict()


def _schema_name(schema: Any) -> str | None:
    if schema is None:
        return None
    return getattr(schema, "__name__", None) or str(schema)


async def describe_agent(agent: Any) -> dict[str, Any]:
    """The catalog record (``key``, ``name``, ``kind``, ``definition``,
    ``fingerprint``) for an Agent, as it is configured right now."""
    key, label = agent_identity(agent)
    tools: list[dict[str, Any]] = []
    host = getattr(agent, "_tool_host", None)
    if host is not None:
        try:
            defs = await host.list_tools()
        except Exception:  # noqa: BLE001 — an unreachable MCP server mustn't break the run
            defs = []
        tools = sorted(
            ({"name": d.name, "destructive": bool(getattr(d, "destructive", False))} for d in defs),
            key=lambda t: t["name"],
        )
    model = getattr(agent, "_model", None)
    architecture = getattr(agent, "_architecture", None)
    definition: dict[str, Any] = {
        "kind": "agent",
        "model": getattr(model, "name", None),
        "architecture": getattr(architecture, "name", None) or type(architecture).__name__,
        "instructions": getattr(agent, "_instructions", ""),
        "tools": tools,
        "max_turns": getattr(agent, "_max_turns", None),
        "permissions": _permissions(getattr(agent, "_permissions", None)),
        "guardrails": [type(g).__name__ for g in getattr(agent, "_guardrails", ())],
        "memory": type(getattr(agent, "_memory", None)).__name__,
        "runtime": type(getattr(agent, "_runtime", None)).__name__,
        "output_schema": _schema_name(getattr(agent, "_default_output_schema", None)),
        "effort": getattr(agent, "_default_effort", None),
        "structure": await structure(agent, label),
    }
    return {
        "key": key,
        "name": label,
        "kind": "agent",
        "definition": definition,
        "fingerprint": _fingerprint(definition),
    }


async def describe_workflow(workflow: Any) -> dict[str, Any]:
    """The catalog record for a Workflow: its nodes and graph."""
    name = str(getattr(workflow, "name", "") or "workflow")
    try:
        graph = workflow.to_mermaid()
    except Exception:  # noqa: BLE001 — the diagram is a nicety
        graph = ""
    definition: dict[str, Any] = {
        "kind": "workflow",
        "nodes": sorted(getattr(workflow, "_nodes", {})),
        "edges": graph,
        "max_steps": getattr(workflow, "_max_steps", None),
        "structure": await structure(workflow, name),
        "runtime": type(getattr(workflow, "_runtime", None)).__name__
        if getattr(workflow, "_runtime", None) is not None
        else None,
    }
    return {
        "key": f"workflow:{name}",
        "name": name,
        "kind": "workflow",
        "definition": definition,
        "fingerprint": _fingerprint(definition),
    }
