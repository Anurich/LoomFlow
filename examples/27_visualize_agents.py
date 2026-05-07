"""27_visualize_agents — Render an agent's structure as a Mermaid graph.

Two scenarios in one file:

1. **A flat Supervisor team** — researcher + writer + reviewer, each
   with its own tools. Shows the basic graph: coordinator at the
   top, workers in a subgraph, tools attached to each worker.
2. **Recursive composition** — a Supervisor whose ``writer`` worker
   is *itself* a `Team.actor_critic` team (an actor + critic pair).
   Shows how nested architectures render as nested subgraphs and
   why this composition is the framework's differentiator over
   sibling-only frameworks.

Each scenario:

* Prints the Mermaid text to stdout (so you can see it immediately)
* Writes ``graph.md`` (Markdown with a ``mermaid`` code fence —
  renders natively on GitHub / IDE preview / Jupyter)
* Writes ``graph.mmd`` (raw Mermaid source — paste into
  https://mermaid.live to render interactively)
* Optionally tries to render PNG via ``mermaid.ink`` (skip with
  ``--no-png`` if you're offline)

This example uses the ``"echo"`` zero-key model so it runs without
``OPENAI_API_KEY``. Graph generation is a pure-introspection pass
— no LLM calls, no network (unless you ask for PNG).

Run::

    pip install -e '.[dev]'
    python examples/27_visualize_agents.py
    python examples/27_visualize_agents.py --no-png
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from jeevesagent import Agent, Team, tool

# ---------------------------------------------------------------------------
# Some custom tools so the graph has tool nodes worth looking at.
# ---------------------------------------------------------------------------


@tool
async def search_kb(query: str) -> str:
    """Search the internal knowledge base."""
    return f"results for {query!r}"


@tool
async def fetch_doc(doc_id: str) -> str:
    """Fetch the full text of a document."""
    return f"contents of {doc_id}"


@tool
async def lint_markdown(text: str) -> str:
    """Run a markdown linter; returns warnings as JSON."""
    return "[]"


# ---------------------------------------------------------------------------
# Scenario 1: a flat Supervisor team
# ---------------------------------------------------------------------------


def _build_flat_supervisor() -> Agent:
    researcher = Agent(
        "You research topics in the knowledge base. Use search_kb "
        "to find relevant material, then fetch_doc to read full text.",
        model="echo",
        tools=[search_kb, fetch_doc],
    )
    writer = Agent(
        "You write structured markdown reports.",
        model="echo",
    )
    reviewer = Agent(
        "You review markdown for issues. Use lint_markdown.",
        model="echo",
        tools=[lint_markdown],
    )
    return Team.supervisor(
        workers={
            "researcher": researcher,
            "writer": writer,
            "reviewer": reviewer,
        },
        instructions="Manage the research-write-review pipeline",
        model="echo",
    )


# ---------------------------------------------------------------------------
# Scenario 2: recursive composition (Supervisor whose writer is a team)
# ---------------------------------------------------------------------------


def _build_nested_supervisor() -> Agent:
    researcher = Agent(
        "You research topics in the knowledge base.",
        model="echo",
        tools=[search_kb, fetch_doc],
    )

    # The writer is no longer a single agent — it's an actor-critic
    # team where one model drafts and another critiques. This is
    # exactly the kind of nesting that makes our framework shine:
    # the outer Supervisor doesn't know or care that one of its
    # workers is a team — it just calls the writer and gets a final
    # output back.
    actor = Agent(
        "You draft markdown reports from research findings.",
        model="echo",
    )
    critic = Agent(
        "You critique drafts. Output JSON {issues, score, summary}.",
        model="echo",
    )
    writer_team = Team.actor_critic(
        actor=actor,
        critic=critic,
        max_rounds=2,
        approval_threshold=0.85,
        instructions="Code-quality coordinator for written drafts",
        model="echo",
    )

    reviewer = Agent(
        "You review markdown for issues. Use lint_markdown.",
        model="echo",
        tools=[lint_markdown],
    )

    return Team.supervisor(
        workers={
            "researcher": researcher,
            "writer": writer_team,  # ← a Team is a regular Agent
            "reviewer": reviewer,
        },
        instructions="Manage the research-write-review pipeline (nested)",
        model="echo",
    )


# ---------------------------------------------------------------------------
# Render one scenario to stdout + disk
# ---------------------------------------------------------------------------


async def _render_scenario(
    title: str,
    agent: Agent,
    out_dir: Path,
    *,
    try_png: bool,
) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

    # 1. Mermaid text to stdout — same call without a path returns
    #    the Mermaid source so you can pipe it / log it / paste it.
    mermaid = await agent.generate_graph(title=title)
    print()
    print(mermaid)
    print()

    # 2. Write Markdown with a mermaid fence — drop this into a
    #    GitHub README and the diagram renders natively.
    md_path = out_dir / "graph.md"
    await agent.generate_graph(md_path, title=title)
    print(f"  Wrote Markdown:    {md_path}")

    # 3. Write raw Mermaid source — paste into https://mermaid.live
    #    for an interactive editor view.
    mmd_path = out_dir / "graph.mmd"
    await agent.generate_graph(mmd_path, title=title)
    print(f"  Wrote raw Mermaid: {mmd_path}")

    # 4. PNG via mermaid.ink — needs network. Degrade gracefully if
    #    offline (the framework writes a .mmd next to the .png path
    #    and raises RuntimeError naming the fallback).
    if try_png:
        png_path = out_dir / "graph.png"
        try:
            await agent.generate_graph(png_path, title=title)
            print(f"  Wrote PNG:         {png_path}")
        except RuntimeError as exc:
            print(f"  PNG render failed: {exc}")
    else:
        print("  PNG: skipped (--no-png)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    try_png = "--no-png" not in sys.argv

    base_dir = Path(  # noqa: ASYNC240 — demo startup
        tempfile.mkdtemp(prefix="jeeves_graph_")
    ).resolve()

    print("Generating agent graphs...")
    print(f"Output directory: {base_dir}")
    if not try_png:
        print("(PNG rendering disabled via --no-png)")

    # Scenario 1
    flat_dir = base_dir / "01_flat_supervisor"
    flat_dir.mkdir()
    await _render_scenario(
        "Flat Supervisor (researcher + writer + reviewer)",
        _build_flat_supervisor(),
        flat_dir,
        try_png=try_png,
    )

    # Scenario 2 (nested)
    nested_dir = base_dir / "02_nested_supervisor"
    nested_dir.mkdir()
    await _render_scenario(
        "Nested Supervisor (writer = ActorCritic team)",
        _build_nested_supervisor(),
        nested_dir,
        try_png=try_png,
    )

    print()
    print("=" * 78)
    print(f"  Done. Inspect outputs in {base_dir}")
    print("=" * 78)
    print()
    print("Tips:")
    print(
        f"  • Open {base_dir}/01_flat_supervisor/graph.md in any "
        "markdown viewer (GitHub, VS Code preview, Jupyter)."
    )
    print(
        "  • Paste the contents of any *.mmd into "
        "https://mermaid.live for an interactive view."
    )
    print(
        "  • Add 'await agent.generate_graph(\"graph.md\")' to "
        "your own setup to capture its current shape."
    )


if __name__ == "__main__":
    asyncio.run(main())
