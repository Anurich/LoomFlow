"""Agent structure visualization — Mermaid graph generator.

.. important::

   **This module is visualization-only. Nothing in here executes.**
   :func:`build_graph` / :func:`write_graph` walk an existing
   :class:`Agent` tree and render a *picture* of it. If you came
   from LangGraph looking for an **executable** graph runtime
   (nodes + edges + routers that actually run), that primitive is
   :class:`loomflow.Workflow`; LLM-driven multi-agent execution is
   :class:`loomflow.Team` / ``Agent(architecture=...)``.

LangGraph established the de facto pattern: emit a Mermaid text
diagram describing the agent's structure, optionally render it to
PNG via ``mermaid.ink``. We follow the same shape, extended to
walk our nested architectures and surface tool attachments.

Public surface
--------------

* :func:`build_graph` — walk an :class:`Agent` (or a
  :class:`~loomflow.Workflow`) and return an :class:`AgentGraph`
  (nodes + edges + subgraphs).
* :meth:`AgentGraph.to_mermaid` — render the graph to Mermaid text;
  :meth:`AgentGraph.to_dict` — the same graph as plain data (what
  Loomflow Studio stores and draws).
* :meth:`Agent.generate_graph` — high-level convenience that walks
  the tree, renders, and dispatches to disk by file extension:

  * ``.mmd`` — raw Mermaid source
  * ``.md``  — Markdown with the diagram in a ``mermaid`` fence
    (renders natively on GitHub, in IDE markdown previews, in
    Jupyter via ``IPython.display.Markdown``)
  * ``.png`` / ``.svg`` — fetched from ``https://mermaid.ink`` via
    ``urllib`` (no extra deps); falls back to writing ``.mmd``
    next to the requested path if the network call fails

What's captured
---------------

* The top-level Agent and its architecture
* Every sub-agent declared by the architecture's
  :meth:`Architecture.declared_workers`
* Tool attachments per agent (filesystem tools, custom ``@tool``
  functions, MCP-bound tools)
* Architecture-specific structural relationships:

  * **Supervisor**: coordinator → workers via ``delegate``
  * **Router**: classifier → routes via ``classifies``
  * **Swarm**: peer → peer via ``handoff``
  * **ActorCritic**: actor ⇄ critic loop
  * **MultiAgentDebate**: debaters → judge
  * **Blackboard**: agents ↔ shared workspace ↔ decider
  * **Reflexion**: wraps the base architecture in a retry loop

* Recursive composition: nested architectures (Reflexion of
  Supervisor, Supervisor whose worker is itself a Supervisor)
  render as nested subgraphs.
* Workflows: start and end, each node (an Agent node with its own
  tools and team), plain edges, router branches, ``parallel``
  branches, and nested workflows as subgraphs.

Agents are labelled with their ``name=``, or else the first line of
their instructions. Each node carries ``keys`` (an agent's name, a
tool's name, a workflow node's name) so a recorded run can be matched
to the parts of the diagram it used.

Design notes
------------

* **Pure stdlib.** No httpx / graphviz / playwright dependency.
  Mermaid is emitted as f-strings; PNG/SVG fetches use ``urllib``.
* **Async**, like the rest of the framework, because tool
  enumeration goes through ``ToolHost.list_tools()`` (the MCP path
  is async).
* **Stable node ids**: a deterministic counter prefixed by node
  kind (``a0``, ``a1`` for agents, ``t0``, ``t1`` for tools, etc.)
  so diffing the output across runs is meaningful.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from .agent.api import Agent
    from .architecture.base import Architecture


# ---------------------------------------------------------------------------
# Internal IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Node:
    id: str
    label: str
    # "agent" — top-level coordinator / single agent
    # "subagent" — worker / specialist / debater
    # "tool" — a registered Tool
    # "memory" — a memory backend
    # "store" — a vector store / lesson store
    # "blackboard" — the Blackboard's shared workspace
    # "step" — a workflow node that's a plain function
    # "workflow" — a nested workflow
    # "start" / "end" — a workflow's entry and exit
    kind: str
    # Names a recorded run uses for this part (agent / tool / node).
    keys: tuple[str, ...] = ()
    # A second line for the drawing: the architecture, a node name.
    detail: str = ""


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    label: str = ""
    # "solid" (default), "dashed" (optional / late-binding),
    # "dotted" (data flow)
    style: str = "solid"


@dataclass
class _Subgraph:
    id: str
    label: str
    members: list[str] = field(default_factory=list)


@dataclass
class AgentGraph:
    """Renderable graph of an agent's structure.

    **Visualization-only** — this is a drawing IR (nodes + edges +
    subgraphs) rendered to Mermaid text; it is never executed. For
    an executable graph, see :class:`loomflow.Workflow`.
    """

    nodes: list[_Node] = field(default_factory=list)
    edges: list[_Edge] = field(default_factory=list)
    subgraphs: list[_Subgraph] = field(default_factory=list)
    title: str = "Agent"

    # ---- rendering --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The graph as plain, JSON-ready data: ``title``, ``nodes``
        (``id``, ``label``, ``kind``, ``keys``, ``detail``), ``edges``
        (``source``, ``target``, ``label``, ``style``) and
        ``subgraphs`` (``id``, ``label``, ``members``)."""
        return {
            "title": self.title,
            "nodes": [
                {"id": n.id, "label": n.label, "kind": n.kind, "keys": list(n.keys), "detail": n.detail}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "label": e.label, "style": e.style}
                for e in self.edges
            ],
            "subgraphs": [
                {"id": g.id, "label": g.label, "members": list(g.members)} for g in self.subgraphs
            ],
        }

    def to_mermaid(self) -> str:
        """Render to a Mermaid ``flowchart TB`` block.

        Output is plain Mermaid (no ``%%{init}%%`` directives) so it
        renders consistently across GitHub, IDE previews, and
        ``mermaid.ink``.
        """
        lines: list[str] = ["flowchart TB"]

        # classDef styling — distinct colors per node kind so the
        # diagram is readable at a glance without a legend.
        lines.extend(
            [
                "  classDef agent fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,font-weight:bold",
                "  classDef subagent fill:#d5e8d4,stroke:#82b366",
                "  classDef tool fill:#fff2cc,stroke:#d6b656",
                "  classDef memory fill:#f8cecc,stroke:#b85450",
                "  classDef store fill:#e1d5e7,stroke:#9673a6",
                "  classDef blackboard fill:#fff2cc,stroke:#d6b656,stroke-dasharray:5 5",
                "  classDef step fill:#f5f5f5,stroke:#666666",
                "  classDef workflow fill:#dae8fc,stroke:#6c8ebf,stroke-dasharray:5 5",
                # One class for start and end: ``end`` is a Mermaid keyword.
                "  classDef terminal fill:#ffffff,stroke:#666666",
            ]
        )

        # Members of any subgraph go inside that subgraph block;
        # everyone else is at the top level.
        members_in_sg: set[str] = set()
        for sg in self.subgraphs:
            members_in_sg.update(sg.members)

        nodes_by_id = {n.id: n for n in self.nodes}

        # Top-level nodes first.
        for node in self.nodes:
            if node.id in members_in_sg:
                continue
            lines.append(f"  {_render_node(node)}")

        # Subgraphs.
        for sg in self.subgraphs:
            lines.append(f'  subgraph {sg.id}["{_escape(sg.label)}"]')
            lines.append("    direction TB")
            for member_id in sg.members:
                member_node = nodes_by_id.get(member_id)
                if member_node is None:
                    continue
                lines.append(f"    {_render_node(member_node)}")
            lines.append("  end")

        # Edges.
        # Edge labels are wrapped in double quotes — that's the
        # Mermaid-canonical form, and it makes mermaid.ink's
        # stricter parser accept labels containing parens / colons /
        # arrows / other special characters that the bare-label form
        # rejects with HTTP 400.
        for edge in self.edges:
            arrow = {
                "solid": "-->",
                "dashed": "-.->",
                "dotted": "~~~",
            }.get(edge.style, "-->")
            if edge.label:
                lines.append(
                    f'  {edge.source} {arrow}|"{_escape(edge.label)}"| {edge.target}'
                )
            else:
                lines.append(f"  {edge.source} {arrow} {edge.target}")

        # classDef applications.
        for node in self.nodes:
            kind = "terminal" if node.kind in ("start", "end") else node.kind
            lines.append(f"  class {node.id} {kind}")

        return "\n".join(lines)


def _render_node(node: _Node) -> str:
    label = _escape(node.label)
    # Different node shapes per kind for additional visual cues.
    if node.kind == "tool":
        return f'{node.id}[/"{label}"\\]'  # parallelogram-ish
    if node.kind in {"memory", "store"}:
        return f'{node.id}[("{label}")]'  # cylinder (database)
    if node.kind == "blackboard":
        return f'{node.id}{{{{"{label}"}}}}'  # hexagon
    if node.kind in {"start", "end"}:
        return f'{node.id}(["{label}"])'  # stadium
    return f'{node.id}["{label}"]'


def _escape(s: str) -> str:
    """Escape characters that confuse the Mermaid parser inside
    bracketed labels."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', "&quot;")
        .replace("\n", " ")
        .replace("[", "(")
        .replace("]", ")")
    )


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------


class _Builder:
    """Walks an Agent + its architecture tree, building an
    :class:`AgentGraph`. Stateful so node ids stay unique."""

    def __init__(self, *, key_for: Callable[[Any], str] | None = None) -> None:
        self.graph = AgentGraph()
        self._counters: dict[str, int] = {}
        self._agent_ids: dict[int, str] = {}  # id(agent) -> node id
        # How an agent is named in recorded runs (its ``keys``).
        self._key_for = key_for or (lambda agent: getattr(agent, "name", None) or "")

    def _new_id(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        return f"{prefix}{n}"

    async def visit_root(self, agent: Agent, *, title: str) -> None:
        """Entry point: visit the top-level agent."""
        self.graph.title = title
        await self._visit_agent(agent, kind="agent", parent_sg=None)

    async def _visit_agent(
        self,
        agent: Agent,
        *,
        kind: str,
        parent_sg: _Subgraph | None,
    ) -> str:
        """Visit an Agent: add a node for it, recurse into its
        architecture, attach its tools. Returns the node id."""
        # De-duplicate: same Agent instance reached twice gets the
        # same id (avoids double-rendering shared workers).
        if id(agent) in self._agent_ids:
            return self._agent_ids[id(agent)]

        node_id = self._new_id("a")
        self._agent_ids[id(agent)] = node_id

        # Its name, else a short label from its instructions.
        label = getattr(agent, "name", None) or _summarize(agent.instructions, fallback="agent")
        key = self._key_for(agent)
        architecture = getattr(agent.architecture, "name", "") or ""
        self.graph.nodes.append(
            _Node(
                id=node_id,
                label=label,
                kind=kind,
                keys=(key,) if key else (),
                detail=architecture if architecture not in ("", "react") else "",
            )
        )
        if parent_sg is not None:
            parent_sg.members.append(node_id)

        # Tools: enumerate from the tool host. ``list_tools`` is the
        # async public surface; works for InProcessToolHost and MCP.
        try:
            tool_defs = await agent.tool_host.list_tools()
        except Exception:  # noqa: BLE001 — tool host may be partially set up
            tool_defs = []

        for tdef in tool_defs:
            tool_id = self._new_id("t")
            self.graph.nodes.append(
                _Node(id=tool_id, label=tdef.name, kind="tool", keys=(tdef.name,))
            )
            if parent_sg is not None:
                parent_sg.members.append(tool_id)
            self.graph.edges.append(
                _Edge(source=node_id, target=tool_id, label="uses")
            )

        # Architecture-specific structure.
        await self._visit_architecture(agent.architecture, parent_id=node_id)
        return node_id

    async def _visit_architecture(
        self,
        arch: Architecture,
        *,
        parent_id: str,
    ) -> None:
        """Dispatch by architecture class. Imports are local to keep
        the graph module's import surface small."""
        from .architecture.actor_critic import ActorCritic
        from .architecture.blackboard import BlackboardArchitecture
        from .architecture.debate import MultiAgentDebate
        from .architecture.react import ReAct
        from .architecture.reflexion import Reflexion
        from .architecture.router import Router
        from .architecture.supervisor import Supervisor
        from .architecture.swarm import Swarm

        if isinstance(arch, Supervisor):
            await self._visit_supervisor(arch, parent_id)
        elif isinstance(arch, Router):
            await self._visit_router(arch, parent_id)
        elif isinstance(arch, Swarm):
            await self._visit_swarm(arch, parent_id)
        elif isinstance(arch, ActorCritic):
            await self._visit_actor_critic(arch, parent_id)
        elif isinstance(arch, MultiAgentDebate):
            await self._visit_debate(arch, parent_id)
        elif isinstance(arch, BlackboardArchitecture):
            await self._visit_blackboard(arch, parent_id)
        elif isinstance(arch, Reflexion):
            await self._visit_reflexion(arch, parent_id)
        elif isinstance(arch, ReAct):
            # Single-agent default; no extra structure beyond tools.
            return
        else:
            # Unknown architecture: fall back to declared_workers.
            for name, worker in arch.declared_workers().items():
                worker_id = await self._visit_agent(
                    worker, kind="subagent", parent_sg=None
                )
                self.graph.edges.append(
                    _Edge(source=parent_id, target=worker_id, label=name)
                )

    # ---- per-architecture renderers ------------------------------------

    async def _visit_supervisor(self, arch: Any, parent_id: str) -> None:
        sg = self._make_subgraph("workers")
        for name, worker in arch.declared_workers().items():
            worker_id = await self._visit_agent(
                worker, kind="subagent", parent_sg=sg
            )
            self.graph.edges.append(
                _Edge(source=parent_id, target=worker_id, label=f"delegate: {name}")
            )
            self.graph.edges.append(
                _Edge(
                    source=parent_id,
                    target=worker_id,
                    label=f"forward: {name}",
                    style="dashed",
                )
            )

    async def _visit_router(self, arch: Any, parent_id: str) -> None:
        sg = self._make_subgraph("routes (1 of N runs)")
        for name, route_agent in arch.declared_workers().items():
            route_id = await self._visit_agent(
                route_agent, kind="subagent", parent_sg=sg
            )
            self.graph.edges.append(
                _Edge(
                    source=parent_id,
                    target=route_id,
                    label=f"classify: {name}",
                )
            )

    async def _visit_swarm(self, arch: Any, parent_id: str) -> None:
        sg = self._make_subgraph("peers")
        peer_ids: dict[str, str] = {}
        entry_name = getattr(arch, "_entry_agent", None)
        for name, peer in arch.declared_workers().items():
            kind = "agent" if name == entry_name else "subagent"
            pid = await self._visit_agent(peer, kind=kind, parent_sg=sg)
            peer_ids[name] = pid
        # An entry edge from the parent (the swarm's outer Agent
        # which is essentially nominal — the architecture is what
        # runs) to the entry peer.
        if entry_name and entry_name in peer_ids:
            self.graph.edges.append(
                _Edge(
                    source=parent_id,
                    target=peer_ids[entry_name],
                    label="entry",
                )
            )
        # Peer→peer handoff edges (each can hand off to any other).
        # Render as dashed since the actual edges are runtime-decided.
        for src in peer_ids:
            for dst in peer_ids:
                if src == dst:
                    continue
                self.graph.edges.append(
                    _Edge(
                        source=peer_ids[src],
                        target=peer_ids[dst],
                        label="handoff",
                        style="dashed",
                    )
                )

    async def _visit_actor_critic(
        self, arch: Any, parent_id: str
    ) -> None:
        actor = arch._actor
        critic = arch._critic
        actor_id = await self._visit_agent(
            actor, kind="subagent", parent_sg=None
        )
        critic_id = await self._visit_agent(
            critic, kind="subagent", parent_sg=None
        )
        self.graph.edges.append(
            _Edge(source=parent_id, target=actor_id, label="generate")
        )
        self.graph.edges.append(
            _Edge(source=actor_id, target=critic_id, label="critique")
        )
        self.graph.edges.append(
            _Edge(
                source=critic_id,
                target=actor_id,
                label="refine if below threshold",
                style="dashed",
            )
        )

    async def _visit_debate(self, arch: Any, parent_id: str) -> None:
        sg = self._make_subgraph("debaters (parallel rounds)")
        debater_ids: list[str] = []
        for name, dbt in arch.declared_workers().items():
            if name == "judge":
                continue
            did = await self._visit_agent(
                dbt, kind="subagent", parent_sg=sg
            )
            debater_ids.append(did)
            self.graph.edges.append(
                _Edge(source=parent_id, target=did, label="round")
            )

        judge = arch.declared_workers().get("judge")
        if judge is not None:
            jid = await self._visit_agent(
                judge, kind="subagent", parent_sg=None
            )
            for did in debater_ids:
                self.graph.edges.append(
                    _Edge(source=did, target=jid, label="responses")
                )

    async def _visit_blackboard(
        self, arch: Any, parent_id: str
    ) -> None:
        # The shared workspace is conceptually central — give it a
        # named hexagon node every agent connects to.
        bb_id = self._new_id("bb")
        self.graph.nodes.append(
            _Node(id=bb_id, label="blackboard", kind="blackboard")
        )

        sg = self._make_subgraph("agents")
        for name, ag in arch.declared_workers().items():
            if name in {"__coordinator", "__decider"}:
                continue
            aid = await self._visit_agent(
                ag, kind="subagent", parent_sg=sg
            )
            self.graph.edges.append(
                _Edge(source=aid, target=bb_id, label="reads/writes")
            )

        coord = arch.declared_workers().get("__coordinator")
        if coord is not None:
            cid = await self._visit_agent(
                coord, kind="subagent", parent_sg=None
            )
            self.graph.edges.append(
                _Edge(
                    source=parent_id, target=cid, label="coordinates"
                )
            )
            self.graph.edges.append(
                _Edge(source=cid, target=bb_id, label="reads")
            )

        decider = arch.declared_workers().get("__decider")
        if decider is not None:
            did = await self._visit_agent(
                decider, kind="subagent", parent_sg=None
            )
            self.graph.edges.append(
                _Edge(source=bb_id, target=did, label="synthesizes")
            )

    async def _visit_reflexion(
        self, arch: Any, parent_id: str
    ) -> None:
        # Wrap the base architecture in a "Reflexion loop" subgraph.
        sg = self._make_subgraph("Reflexion retry loop")
        sg.members.append(parent_id)  # outer Agent IS inside the loop
        # Recurse into the base architecture; its sub-nodes go into
        # the same subgraph so the user sees the wrapping clearly.
        # We do this by stashing the subgraph as the current group;
        # recursion just appends nodes to the global graph and we
        # add the new node ids to the subgraph members afterwards.
        before_ids = {n.id for n in self.graph.nodes}
        await self._visit_architecture(arch._base, parent_id=parent_id)
        new_ids = [n.id for n in self.graph.nodes if n.id not in before_ids]
        sg.members.extend(new_ids)

        # Lesson store gets its own node (cylinder) outside the loop.
        if arch._lesson_store is not None:
            store_id = self._new_id("st")
            self.graph.nodes.append(
                _Node(
                    id=store_id, label="lesson_store", kind="store"
                )
            )
            self.graph.edges.append(
                _Edge(
                    source=parent_id,
                    target=store_id,
                    label="recall lessons",
                    style="dashed",
                )
            )
            self.graph.edges.append(
                _Edge(
                    source=parent_id,
                    target=store_id,
                    label="persist lesson",
                    style="dashed",
                )
            )

    # ---- workflows -----------------------------------------------------

    async def visit_workflow(self, workflow: Any, *, parent_sg: _Subgraph | None = None) -> str:
        """Visit a Workflow: its start, nodes, edges and end. Returns
        the start node's id."""
        from .agent.api import Agent
        from .workflow import Workflow, _Router, _Sentinel

        start_id = self._new_id("s")
        self.graph.nodes.append(_Node(id=start_id, label="start", kind="start"))
        if parent_sg is not None:
            parent_sg.members.append(start_id)
        end_id: str | None = None

        def end() -> str:
            nonlocal end_id
            if end_id is None:
                end_id = self._new_id("e")
                self.graph.nodes.append(_Node(id=end_id, label="end", kind="end"))
                if parent_sg is not None:
                    parent_sg.members.append(end_id)
            return end_id

        ids: dict[str, str] = {}
        raw_nodes: dict[str, Any] = getattr(workflow, "_raw_nodes", {})
        for name in getattr(workflow, "_nodes", {}):
            raw = raw_nodes.get(name)
            if isinstance(raw, Agent):
                agent_id = await self._visit_agent(raw, kind="subagent", parent_sg=parent_sg)
                node = next(n for n in self.graph.nodes if n.id == agent_id)
                detail = node.detail if node.label == name else f"node {name}"
                self._replace(node, keys=(*node.keys, name), detail=detail)
                ids[name] = agent_id
            elif isinstance(raw, Workflow):
                inner = self._make_subgraph(f"workflow {raw.name}")
                wf_id = self._new_id("w")
                self.graph.nodes.append(_Node(id=wf_id, label=name, kind="workflow", keys=(name, raw.name)))
                if parent_sg is not None:
                    parent_sg.members.append(wf_id)
                inner_start = await self.visit_workflow(raw, parent_sg=inner)
                self.graph.edges.append(_Edge(source=wf_id, target=inner_start, label="runs", style="dashed"))
                ids[name] = wf_id
            else:
                step_id = self._new_id("n")
                self.graph.nodes.append(_Node(id=step_id, label=name, kind="step", keys=(name,)))
                if parent_sg is not None:
                    parent_sg.members.append(step_id)
                ids[name] = step_id
                for branch_name, branch in getattr(raw, "_loom_branches", ()):
                    branch_id = await self._visit_branch(branch_name, branch, parent_sg)
                    self.graph.edges.append(_Edge(source=step_id, target=branch_id, label="in parallel"))

        def target_id(dst: Any) -> str | None:
            if isinstance(dst, _Sentinel) or dst == "__END__":
                return end()
            return ids.get(str(dst))

        def add_router(src: str, router: Any) -> None:
            for label, dst in router.routes.items():
                tid = target_id(dst)
                if tid is not None:
                    self.graph.edges.append(_Edge(source=src, target=tid, label=str(label)))
            if router.default is not None:
                tid = target_id(router.default)
                if tid is not None:
                    self.graph.edges.append(_Edge(source=src, target=tid, label="default", style="dashed"))

        entry = getattr(workflow, "_entry_router", None)
        if entry is not None:
            add_router(start_id, entry)
        elif getattr(workflow, "_start", None) in ids:
            self.graph.edges.append(_Edge(source=start_id, target=ids[workflow._start]))
        for src, targets in getattr(workflow, "_edges", {}).items():
            if src not in ids:
                continue
            for target in targets:
                if isinstance(target, _Router):
                    add_router(ids[src], target)
                else:
                    tid = target_id(target)
                    if tid is not None:
                        self.graph.edges.append(_Edge(source=ids[src], target=tid))
        return start_id

    async def _visit_branch(self, name: str, branch: Any, parent_sg: _Subgraph | None) -> str:
        """One ``Workflow.parallel`` branch: an Agent, a Workflow or a
        function."""
        from .agent.api import Agent
        from .workflow import Workflow

        if isinstance(branch, Agent):
            branch_id = await self._visit_agent(branch, kind="subagent", parent_sg=parent_sg)
            node = next(n for n in self.graph.nodes if n.id == branch_id)
            self._replace(node, keys=(*node.keys, name, f"fan_out.{name}"))
            return branch_id
        if isinstance(branch, Workflow):
            inner = self._make_subgraph(f"workflow {branch.name}")
            return await self.visit_workflow(branch, parent_sg=inner)
        step_id = self._new_id("n")
        self.graph.nodes.append(_Node(id=step_id, label=name, kind="step", keys=(name, f"fan_out.{name}")))
        if parent_sg is not None:
            parent_sg.members.append(step_id)
        return step_id

    def _replace(self, node: _Node, **changes: Any) -> None:
        if "keys" in changes:
            changes["keys"] = tuple(dict.fromkeys(changes["keys"]))
        index = self.graph.nodes.index(node)
        self.graph.nodes[index] = replace(node, **changes)

    def _make_subgraph(self, label: str) -> _Subgraph:
        sg = _Subgraph(id=self._new_id("sg"), label=label)
        self.graph.subgraphs.append(sg)
        return sg


def _summarize(text: str, *, fallback: str = "agent") -> str:
    """First line of the agent's instructions, capped at ~60 chars."""
    if not text:
        return fallback
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) > 60:
        first_line = first_line[:57] + "..."
    return first_line or fallback


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def build_graph(
    agent: Any,
    *,
    title: str = "Agent",
    key_for: Callable[[Any], str] | None = None,
) -> AgentGraph:
    """Walk an :class:`Agent` or a :class:`~loomflow.Workflow` and
    return its renderable :class:`AgentGraph`.

    ``key_for(agent)`` names each agent in the nodes' ``keys`` (default:
    its ``name=``); Studio passes its catalog key.
    """
    from .workflow import Workflow

    builder = _Builder(key_for=key_for)
    if isinstance(agent, Workflow):
        builder.graph.title = title
        await builder.visit_workflow(agent)
    else:
        await builder.visit_root(agent, title=title)
    return builder.graph


# ---------------------------------------------------------------------------
# File output: dispatch by extension
# ---------------------------------------------------------------------------


async def write_graph(
    agent: Agent,
    path: str | Path,
    *,
    title: str | None = None,
) -> str:
    """Walk the agent, render to Mermaid, write to ``path``.

    Extension dispatch:

    * ``.mmd`` — raw Mermaid source
    * ``.md``  — Markdown with the diagram in a ``mermaid`` fence
    * ``.png`` / ``.svg`` — fetched from ``mermaid.ink``; on
      network failure, writes ``.mmd`` next to the requested path
      and returns the Mermaid text anyway

    Returns the Mermaid text in every case.
    """
    target = Path(path)
    graph = await build_graph(agent, title=title or target.stem or "Agent")
    mermaid = graph.to_mermaid()
    suffix = target.suffix.lower()

    # Sync path I/O is fine here — graph generation is a one-shot
    # operation, and the mermaid.ink fetch (when used) dominates
    # any wall-clock cost.
    if suffix == ".mmd":
        target.write_text(mermaid, encoding="utf-8")  # noqa: ASYNC240
    elif suffix == ".md":
        target.write_text(  # noqa: ASYNC240
            f"# {graph.title}\n\n```mermaid\n{mermaid}\n```\n",
            encoding="utf-8",
        )
    elif suffix in {".png", ".svg"}:
        fmt = "png" if suffix == ".png" else "svg"
        try:
            data = _fetch_mermaid_ink(mermaid, fmt=fmt)
            target.write_bytes(data)  # noqa: ASYNC240
        except (URLError, OSError, TimeoutError) as exc:
            # Network unavailable — degrade gracefully by writing
            # the Mermaid source next to the requested path so the
            # user still has something they can paste into a
            # renderer.
            fallback = target.with_suffix(".mmd")
            fallback.write_text(mermaid, encoding="utf-8")  # noqa: ASYNC240
            raise RuntimeError(
                f"Could not reach mermaid.ink ({exc}). "
                f"Wrote Mermaid source to {fallback} as a fallback. "
                f"Open it on https://mermaid.live to render."
            ) from exc
    else:
        # Unknown extension: write Mermaid source.
        target.write_text(mermaid, encoding="utf-8")  # noqa: ASYNC240

    return mermaid


def _fetch_mermaid_ink(
    mermaid: str, *, fmt: str, timeout: float = 10.0
) -> bytes:
    """Render Mermaid via ``mermaid.ink``. ``fmt`` is ``"png"`` or
    ``"svg"``. Raises on network error.

    The endpoint shape:

    * ``GET /img/<base64>?type=png`` → returns PNG bytes
      (the bare ``/img/<base64>`` returns JPEG by default — that's
      why ``?type=png`` is required for proper PNG output)
    * ``GET /svg/<base64>``         → returns SVG bytes
    """
    encoded = base64.urlsafe_b64encode(mermaid.encode("utf-8")).decode(
        "ascii"
    )
    if fmt == "svg":
        url = f"https://mermaid.ink/svg/{encoded}"
    else:
        url = f"https://mermaid.ink/img/{encoded}?type=png"
    request = Request(  # noqa: S310 — fixed scheme, user-supplied data is base64
        url,
        headers={"User-Agent": "Loom/graph"},
    )
    with urlopen(request, timeout=timeout) as resp:  # noqa: S310
        return bytes(resp.read())
