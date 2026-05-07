"""Tests for the agent graph visualization module.

Covers:

* The Mermaid emitter renders nodes / edges / subgraphs / classDef
  in the expected order and form.
* Each architecture produces structurally correct output:
  - ReAct: only the agent + its tools
  - Supervisor: coordinator → workers via delegate / forward_message
  - Router: classifier → routes via classify
  - Swarm: peer↔peer handoff edges
  - ActorCritic: actor ↔ critic loop
  - Debate: debaters → judge
  - Blackboard: agents ↔ blackboard ↔ decider
  - Reflexion: base architecture wrapped in a Reflexion subgraph
* Recursive composition: Supervisor whose worker is itself a
  Supervisor renders nested subgraphs.
* File extension dispatch: ``.mmd`` / ``.md`` / ``.png`` / ``.svg``
  produce the right artefact (PNG path is mocked to avoid network).
* ``Agent.generate_graph(path=None)`` returns the Mermaid text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from jeevesagent import (
    Agent,
    HashEmbedder,
    InMemoryVectorStore,
    Reflexion,
    RouterRoute,
    ScriptedModel,
    ScriptedTurn,
    Supervisor,
    Team,
    build_graph,
    tool,
    write_graph,
)
from jeevesagent.graph import _Builder, _escape

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(label: str, *, tools: list[Any] | None = None) -> Agent:
    return Agent(
        label,
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        tools=tools,
    )


# ---------------------------------------------------------------------------
# Mermaid emitter primitives
# ---------------------------------------------------------------------------


def test_escape_handles_brackets_and_quotes() -> None:
    assert _escape('a "b" c') == "a &quot;b&quot; c"
    assert _escape("[x]") == "(x)"
    assert _escape("line1\nline2") == "line1 line2"


# ---------------------------------------------------------------------------
# ReAct (single-agent default)
# ---------------------------------------------------------------------------


async def test_react_graph_has_agent_and_tools() -> None:
    @tool
    async def my_tool(x: str) -> str:
        return x

    a = _agent("solo", tools=[my_tool])
    graph = await build_graph(a)
    text = graph.to_mermaid()

    assert "flowchart TB" in text
    assert "solo" in text
    # Tool appears with its name.
    assert "my_tool" in text
    # An edge from agent to tool labelled "uses".
    assert "uses" in text


# ---------------------------------------------------------------------------
# Supervisor — coordinator → workers
# ---------------------------------------------------------------------------


async def test_supervisor_graph_renders_workers_in_subgraph() -> None:
    researcher = _agent("Research the question")
    writer = _agent("Write the report")
    team = Team.supervisor(
        workers={"researcher": researcher, "writer": writer},
        instructions="Manage the pipeline",
        model="echo",
    )
    graph = await build_graph(team)
    text = graph.to_mermaid()

    # Coordinator label.
    assert "Manage the pipeline" in text
    # Workers in a subgraph block.
    assert "subgraph" in text
    assert "Research the question" in text
    assert "Write the report" in text
    # Edge labels.
    assert "delegate: researcher" in text
    assert "delegate: writer" in text
    # Forward edges (dashed). Label wrapped in quotes for mermaid.ink
    # parser compatibility.
    assert "forward: researcher" in text
    assert '-.->|"forward' in text


# ---------------------------------------------------------------------------
# Router — classifier → routes
# ---------------------------------------------------------------------------


async def test_router_graph_renders_routes() -> None:
    a = _agent("billing specialist")
    b = _agent("technical specialist")
    team = Team.router(
        routes=[
            RouterRoute(name="billing", agent=a, description="..."),
            RouterRoute(name="technical", agent=b, description="..."),
        ],
        instructions="triage",
        model="echo",
    )
    graph = await build_graph(team)
    text = graph.to_mermaid()
    assert "classify: billing" in text
    assert "classify: technical" in text


# ---------------------------------------------------------------------------
# Swarm — peer↔peer handoffs
# ---------------------------------------------------------------------------


async def test_swarm_graph_includes_handoff_edges_between_peers() -> None:
    a = _agent("triage agent")
    b = _agent("billing agent")
    c = _agent("tech agent")
    team = Team.swarm(
        agents={"triage": a, "billing": b, "tech": c},
        entry_agent="triage",
        instructions="hub",
        model="echo",
    )
    graph = await build_graph(team)
    text = graph.to_mermaid()
    # Handoff edges show up as dashed.
    assert text.count('|"handoff"|') >= 6  # 3 peers × 2 directions
    assert '|"entry"|' in text


# ---------------------------------------------------------------------------
# ActorCritic — actor ↔ critic loop
# ---------------------------------------------------------------------------


async def test_actor_critic_graph_renders_loop() -> None:
    actor = _agent("Actor")
    critic = _agent("Critic")
    team = Team.actor_critic(
        actor=actor,
        critic=critic,
        instructions="root",
        model="echo",
    )
    graph = await build_graph(team)
    text = graph.to_mermaid()
    assert '|"generate"|' in text
    assert '|"critique"|' in text
    assert "refine" in text


# ---------------------------------------------------------------------------
# Debate — debaters → judge
# ---------------------------------------------------------------------------


async def test_debate_graph_renders_judge_synthesis() -> None:
    d1 = _agent("Optimist")
    d2 = _agent("Skeptic")
    judge = _agent("Judge")
    team = Team.debate(
        debaters=[d1, d2],
        judge=judge,
        instructions="moderator",
        model="echo",
    )
    graph = await build_graph(team)
    text = graph.to_mermaid()
    assert "Optimist" in text
    assert "Skeptic" in text
    assert "Judge" in text
    assert '|"responses"|' in text
    assert '|"round"|' in text


# ---------------------------------------------------------------------------
# Blackboard — agents ↔ blackboard ↔ decider
# ---------------------------------------------------------------------------


async def test_blackboard_graph_renders_central_workspace() -> None:
    a = _agent("hypothesis")
    b = _agent("evidence")
    coord = _agent("coord")
    decider = _agent("decider")
    team = Team.blackboard(
        agents={"hypothesis": a, "evidence": b},
        coordinator=coord,
        decider=decider,
        instructions="root",
        model="echo",
    )
    graph = await build_graph(team)
    text = graph.to_mermaid()
    # Hexagon node for the workspace.
    assert "blackboard" in text
    # Reads/writes edges from each agent to the workspace.
    assert text.count("reads/writes") >= 2
    # Decider receives synthesized state from the workspace.
    assert '|"synthesizes"|' in text


# ---------------------------------------------------------------------------
# Reflexion — wraps a base architecture
# ---------------------------------------------------------------------------


async def test_reflexion_wraps_base_in_subgraph() -> None:
    worker = _agent("worker")
    base_team = Team.supervisor(
        workers={"a": worker}, instructions="base", model="echo"
    )
    # Wrap the same constructor's inner architecture in Reflexion.
    base_arch = base_team.architecture
    assert isinstance(base_arch, Supervisor)

    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=32))
    wrapped = Agent(
        "outer",
        model="echo",
        architecture=Reflexion(base=base_arch, lesson_store=store),
    )
    graph = await build_graph(wrapped)
    text = graph.to_mermaid()

    assert "Reflexion retry loop" in text
    # Lesson store cylinder.
    assert "lesson_store" in text
    # Both recall + persist edges.
    assert "recall lessons" in text
    assert "persist lesson" in text


# ---------------------------------------------------------------------------
# Recursive composition — Supervisor whose worker is itself a Supervisor
# ---------------------------------------------------------------------------


async def test_nested_supervisor_renders_nested_subgraphs() -> None:
    inner_a = _agent("inner agent A")
    inner_b = _agent("inner agent B")
    inner_team = Team.supervisor(
        workers={"a": inner_a, "b": inner_b},
        instructions="inner manager",
        model="echo",
    )
    outer_team = Team.supervisor(
        workers={"inner": inner_team},
        instructions="outer manager",
        model="echo",
    )
    graph = await build_graph(outer_team)
    text = graph.to_mermaid()
    # Three levels of agents visible.
    assert "outer manager" in text
    assert "inner manager" in text
    assert "inner agent A" in text
    assert "inner agent B" in text
    # Two subgraph blocks (one per supervisor).
    assert text.count("subgraph") >= 2


# ---------------------------------------------------------------------------
# Same Agent reached twice → de-duplicated (shared workers across teams)
# ---------------------------------------------------------------------------


async def test_shared_worker_appears_once_in_graph() -> None:
    shared = _agent("Shared specialist")
    team = Team.supervisor(
        workers={"a": shared, "b": shared},  # same instance
        instructions="root",
        model="echo",
    )
    graph = await build_graph(team)
    # Only one node for the shared agent (de-dupe by id()).
    shared_nodes = [
        n for n in graph.nodes if n.label == "Shared specialist"
    ]
    assert len(shared_nodes) == 1


# ---------------------------------------------------------------------------
# Agent.generate_graph (no path) returns Mermaid text
# ---------------------------------------------------------------------------


async def test_agent_generate_graph_returns_mermaid_text() -> None:
    agent = _agent("solo")
    text = await agent.generate_graph()
    assert text.startswith("flowchart TB")
    assert "solo" in text


# ---------------------------------------------------------------------------
# File extension dispatch
# ---------------------------------------------------------------------------


async def test_generate_graph_writes_mmd(tmp_path: Path) -> None:
    agent = _agent("solo")
    out = tmp_path / "graph.mmd"
    text = await agent.generate_graph(out)
    assert out.exists()
    assert out.read_text() == text
    assert text.startswith("flowchart TB")


async def test_generate_graph_writes_md_with_fence(
    tmp_path: Path,
) -> None:
    agent = _agent("solo")
    out = tmp_path / "graph.md"
    await agent.generate_graph(out, title="My Team")
    body = out.read_text()
    assert body.startswith("# My Team")
    assert "```mermaid" in body
    assert "flowchart TB" in body
    assert body.rstrip().endswith("```")


async def test_generate_graph_unknown_extension_writes_mermaid(
    tmp_path: Path,
) -> None:
    agent = _agent("solo")
    out = tmp_path / "graph.txt"
    await agent.generate_graph(out)
    assert out.exists()
    assert out.read_text().startswith("flowchart TB")


async def test_generate_graph_png_falls_back_on_network_error(
    tmp_path: Path,
) -> None:
    """If mermaid.ink is unreachable, we write the Mermaid source
    next to the requested path and raise a helpful error."""
    from urllib.error import URLError

    agent = _agent("solo")
    out = tmp_path / "graph.png"

    with patch(
        "jeevesagent.graph._fetch_mermaid_ink",
        side_effect=URLError("mock offline"),
    ):
        with pytest.raises(RuntimeError, match="mermaid.ink"):
            await agent.generate_graph(out)

    fallback = out.with_suffix(".mmd")
    assert fallback.exists()
    assert fallback.read_text().startswith("flowchart TB")


async def test_generate_graph_png_writes_bytes_on_success(
    tmp_path: Path,
) -> None:
    """When the PNG fetch succeeds, the returned bytes are written
    to the requested path."""
    fake_png = b"\x89PNG\r\n\x1a\n"  # PNG magic header
    agent = _agent("solo")
    out = tmp_path / "graph.png"

    with patch(
        "jeevesagent.graph._fetch_mermaid_ink", return_value=fake_png
    ):
        await agent.generate_graph(out)

    assert out.exists()
    assert out.read_bytes() == fake_png


# ---------------------------------------------------------------------------
# Builder primitives
# ---------------------------------------------------------------------------


def test_builder_assigns_unique_ids() -> None:
    b = _Builder()
    assert b._new_id("a") == "a0"
    assert b._new_id("a") == "a1"
    assert b._new_id("t") == "t0"
    assert b._new_id("a") == "a2"


# ---------------------------------------------------------------------------
# write_graph public function works without going through Agent
# ---------------------------------------------------------------------------


async def test_write_graph_top_level(tmp_path: Path) -> None:
    agent = _agent("solo")
    out = tmp_path / "g.md"
    text = await write_graph(agent, out)
    assert "solo" in text
    assert out.exists()
