"""22_rag_with_loader — Full RAG pipeline: loader → chunker → vector store → multi-agent.

What it shows
-------------

The end-to-end production-shape RAG pipeline using every piece of
the framework that ships in v0.5+:

* **Loaders** — :func:`jeevesagent.loader.load` auto-detects the
  format and converts to markdown. We show ``.md`` here, but the
  same line works for ``.pdf`` / ``.docx`` / ``.xlsx`` / ``.csv``
  / ``.html`` (just install the matching extra).
* **Chunking** — :class:`MarkdownChunker` splits the converted
  markdown on heading boundaries and **preserves the header trail
  in each chunk's metadata**. Retrieval surfaces section context.
* **Vector store** — :class:`InMemoryVectorStore` with a pluggable
  :class:`Embedder`, built in one line via :meth:`from_chunks`.
  Swap the class for :class:`ChromaVectorStore` /
  :class:`PostgresVectorStore` / :class:`FAISSVectorStore` for
  production with no other code changes.
* **Hybrid search** — :meth:`search_hybrid` combines BM25 (for
  exact-term hits like "ReWOO" / model names / error codes) with
  cosine similarity, fused via Reciprocal Rank Fusion. Better
  recall than pure embedding search on technical content.
* **Mongo-style filters** — search results can be restricted by
  metadata via ``filter={"source": {"$in": [...]}}`` etc.,
  translated per-backend so the same filter expression works on
  every store.
* **Tools** — Two RAG tools (``search_kb`` for hybrid search,
  ``search_kb_in`` for filtered search) for the researcher; plus
  the framework's **built-in filesystem tools**
  (:func:`read_tool`, :func:`write_tool`, :func:`edit_tool`) for
  the writer + reviewer. Everything shares the same default
  workdir, so write→edit→read across workers Just Works.
* **Multi-agent** — :class:`Supervisor` coordinates a
  ``researcher`` + ``writer`` + ``reviewer`` pipeline. Worker
  events stream end-to-end to the outer ``agent.stream(...)``.

What it produces
----------------

A polished comparison report on disk at
``${TMPDIR}/jeeves_agent_*/comparison.md``, written and reviewed
by the agent team using ONLY the framework's built-in tools.

Run::

    pip install -e '.[dev,openai,loader]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/22_rag_with_loader.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ OPENAI_API_KEY not set — skipping this example.\n"
        "    Add OPENAI_API_KEY=sk-... to .env at repo root to run.\n"
    )
    sys.exit(0)

from jeevesagent import (  # noqa: E402
    Agent,
    HashEmbedder,
    InMemoryVectorStore,
    Team,
    edit_tool,
    read_tool,
    tool,
    write_tool,
)
from jeevesagent.loader import (  # noqa: E402
    MarkdownChunker,
    load,
)

# ---------------------------------------------------------------------------
# Stage 1: Generate sample knowledge-base documents on disk.
#
# In production these would be your actual .pdf / .docx / .xlsx
# files. The loader handles all of them via the same `load(path)`
# call. For this example we write 4 markdown files so the demo is
# zero-key and self-contained.
# ---------------------------------------------------------------------------

KB_DOCS: dict[str, str] = {
    "react.md": (
        "# ReAct\n\n"
        "## Overview\n\n"
        "ReAct is the canonical observe-think-act loop. The model "
        "calls tools when needed and produces a final response when "
        "it doesn't.\n\n"
        "## Cost\n\n"
        "1 LLM call per turn. Total cost scales linearly with the "
        "number of tool calls the model makes.\n\n"
        "## When to use\n\n"
        "General agents with mixed tool use. ReAct is the default "
        "in JeevesAgent.\n\n"
        "## Failure modes\n\n"
        "Goal drift on long tasks (>10 turns). The model loses "
        "track of the original goal as the message history grows.\n"
    ),
    "router.md": (
        "# Router\n\n"
        "## Overview\n\n"
        "Router classifies the user's request once via a small fast "
        "model and dispatches to ONE specialist Agent.\n\n"
        "## Cost\n\n"
        "1 classifier call + 1 specialist run. The cheapest "
        "multi-agent pattern. Router is genuinely cheaper than "
        "single-agent ReAct on tasks with clear specialist "
        "boundaries.\n\n"
        "## When to use\n\n"
        "Customer support, helpdesk, API gateways. Anywhere with "
        "clear specialist boundaries.\n\n"
        "## Failure modes\n\n"
        "Routing errors are total — there's no recovery if the "
        "wrong route is picked. Use confidence thresholds + a "
        "fallback route to mitigate.\n"
    ),
    "supervisor.md": (
        "# Supervisor\n\n"
        "## Overview\n\n"
        "Supervisor coordinates a team of worker Agents via a "
        "`delegate(worker, instructions)` tool injected into its "
        "loop.\n\n"
        "## Cost\n\n"
        "1.5-3x single-agent. Multiple delegations in one "
        "supervisor turn run in parallel via anyio task groups.\n\n"
        "## When to use\n\n"
        "Multi-domain tasks needing specialist roles. Research + "
        "code + review pipelines. Anthropic reports +90.2% over "
        "single-agent on their MA Research benchmark.\n\n"
        "## Failure modes\n\n"
        "Supervisor is a bottleneck if the model emits poor "
        "delegation instructions. Workers don't see the user's "
        "original message.\n"
    ),
    "rewoo.md": (
        "# ReWOO\n\n"
        "## Overview\n\n"
        "ReWOO emits a structured plan of tool calls upfront with "
        "`{{En}}` placeholder substitution. Independent steps run "
        "in parallel.\n\n"
        "## Cost\n\n"
        "2 LLM calls + N tool calls. ReAct on the same task would "
        "be roughly N+1 LLM calls. 30-50% cheaper for tool-heavy "
        "predictable workloads.\n\n"
        "## When to use\n\n"
        "Tool-heavy multi-step tasks where the planner can predict "
        "the call sequence upfront. Multi-source data lookups + "
        "synthesis.\n\n"
        "## Failure modes\n\n"
        "Planner must predict accurately upfront — no replanning "
        "on failure in v1. Hallucinated tool names produce errors "
        "at dispatch time.\n"
    ),
}


def _materialize_kb(workdir: Path) -> Path:
    """Write the 4 KB markdown files to ``workdir/kb/`` and return
    the kb path. In production these would already exist on disk."""
    kb = workdir / "kb"
    kb.mkdir(parents=True, exist_ok=True)
    for filename, content in KB_DOCS.items():
        (kb / filename).write_text(content)
    return kb


# ---------------------------------------------------------------------------
# Stage 2: Load + chunk + build the vector store in one line.
#
# `load(path)` auto-detects format and converts to markdown.
# `MarkdownChunker` splits on heading boundaries; each chunk's
# metadata.headers is the path of parent headers ([file, "ReAct",
# "Cost"] etc.) so retrieval surfaces section context.
#
# `InMemoryVectorStore.from_chunks(chunks, embedder=...)` is the
# one-shot factory: construct + embed + store + return. No parallel
# lists, no manual cosine. Swap the class for `ChromaVectorStore`,
# `PostgresVectorStore`, or `FAISSVectorStore` and nothing else
# changes.
# ---------------------------------------------------------------------------


# Module-global; populated by main() so the @tool functions below
# can reach it without thread-local context plumbing.
_STORE: InMemoryVectorStore | None = None


async def _build_index_from_kb(kb_path: Path) -> InMemoryVectorStore:
    """Load every file under kb_path, chunk it, build a vector store
    from the chunks via the from_chunks factory.

    We tag each chunk's metadata with both ``source`` (full path,
    set by the chunker) and ``source_name`` (basename) so the
    ``$in`` filter can match on filenames the LLM names directly.
    """
    chunker = MarkdownChunker(chunk_size=600, chunk_overlap=80)
    all_chunks = []
    for doc_path in sorted(kb_path.iterdir()):  # noqa: ASYNC240 — demo startup
        if not doc_path.is_file():
            continue
        # `load(path)` dispatches by extension. Same line works for
        # .pdf, .docx, .xlsx, .csv, .html.
        document = load(doc_path)
        for chunk in chunker.split(
            document.content, source=str(doc_path)
        ):
            chunk.metadata["source_name"] = doc_path.name
            all_chunks.append(chunk)
    # One-shot factory: embed all chunks + return a populated store.
    return await InMemoryVectorStore.from_chunks(
        all_chunks, embedder=HashEmbedder(dimensions=256)
    )


# ---------------------------------------------------------------------------
# Stage 3: RAG tools — hybrid search (vector + BM25) + filtered search.
#
# `search_hybrid` is the killer feature here: pure embedding search
# misses exact-term queries ("ReWOO", "MMR", error codes, model
# names) because tokenization smears them across hash dimensions.
# BM25 catches them, RRF fuses the rankings, the model gets the
# best of both worlds. Try the same code with .search() instead and
# watch recall drop on technical queries.
# ---------------------------------------------------------------------------


def _format_results(results: list) -> str:
    parts = []
    for r in results:
        source = Path(r.chunk.metadata.get("source", "")).name
        headers = r.chunk.metadata.get("headers") or []
        trail = " > ".join(headers) if headers else "(no headers)"
        parts.append(
            f"[{source}] {trail} (score={r.score:.3f})\n"
            f"{r.chunk.content.strip()}\n"
        )
    return "\n---\n".join(parts)


@tool
async def search_kb(query: str) -> str:
    """Hybrid search (BM25 + vector via RRF) over the indexed
    knowledge base. Returns the top-3 chunks with their source
    path, header trail, and content.

    Hybrid search catches both semantic matches AND exact-term hits
    (architecture names like "ReWOO", error codes, etc.) — better
    recall than pure embedding search on technical content.
    """
    assert _STORE is not None
    results = await _STORE.search_hybrid(query, k=3)
    if not results:
        return "ERROR: no results (was the index built?)"
    return _format_results(results)


@tool
async def search_kb_in(query: str, sources: list[str]) -> str:
    """Search restricted to a given set of source files. ``sources``
    is a list of filenames (e.g. ``["router.md", "rewoo.md"]``).
    Useful when you want to compare claims across two specific docs.

    Demonstrates the framework's Mongo-style filter operators —
    ``filter={"source_name": {"$in": [...]}}`` translates per-
    backend so this same line works on InMemory / Chroma /
    Postgres / FAISS.
    """
    assert _STORE is not None
    results = await _STORE.search(
        query, k=3, filter={"source_name": {"$in": sources}}
    )
    if not results:
        return f"(no matches in {sources!r})"
    return _format_results(results)


# ---------------------------------------------------------------------------
# Stage 4: Workers — researcher, writer, reviewer.
#
# Researcher uses the RAG tool only. Writer + reviewer use the
# framework's built-in filesystem tools, all sharing the same
# default workdir so write→edit→read works across workers.
# ---------------------------------------------------------------------------


researcher = Agent(
    instructions=(
        "You research the user's question by querying a "
        "knowledge base. Process:\n"
        "1. Call `search_kb(query)` (in parallel where possible — "
        "issue multiple searches in one turn for different sub-"
        "topics) for each topic the user mentions. This uses hybrid "
        "search (BM25 + vector) so exact terms like 'ReWOO' or "
        "'Router' hit reliably.\n"
        "2. If two specific architectures must be compared head-"
        "to-head, also call `search_kb_in(query, sources)` with "
        "the matching filenames (e.g. `['router.md', 'rewoo.md']`) "
        "to keep the result set focused.\n"
        "3. Each result includes the source file, header trail, "
        "score, and chunk content. Quote the chunk content "
        "verbatim in your findings — the writer will rely on your "
        "exact text.\n"
        "4. Return a tight summary: one paragraph per topic, "
        "ending each with the source path so the writer can cite."
    ),
    model="gpt-4.1-mini",
    tools=[search_kb, search_kb_in],
)

writer = Agent(
    instructions=(
        "You write structured markdown reports.\n\n"
        "Tools:\n"
        "* `write(path, content)` — create or fully overwrite.\n"
        "* `read(path)` — verify what's on disk (line-numbered).\n"
        "* `edit(path, old_string, new_string)` — in-place fix. "
        "old_string must match the file's contents EXACTLY and "
        "must be unique (or pass replace_all=true).\n\n"
        "Always use the file path the supervisor names. Verify "
        "with `read` after writing."
    ),
    model="gpt-4.1-mini",
    tools=[write_tool(), read_tool(), edit_tool()],
)

reviewer = Agent(
    instructions=(
        "You review markdown reports for completeness and "
        "accuracy. Process:\n"
        "1. Call `read(path)` on the file the supervisor names.\n"
        "2. List specific issues. For each, give the writer the "
        "exact old_string to find (copy from `read`'s output, "
        "minus the line-number prefix) and the proposed "
        "replacement.\n"
        "Be specific. The writer will pass your suggestions "
        "verbatim to ``edit``."
    ),
    model="gpt-4.1-mini",
    tools=[read_tool()],
)


# ---------------------------------------------------------------------------
# Stage 5: Top-level supervisor.
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = Path(  # noqa: ASYNC240 — demo startup, sync OK
        tempfile.mkdtemp(prefix="jeeves_rag_")
    ).resolve()
    print("=" * 70)
    print("RAG with loader + chunker + multi-agent")
    print("=" * 70)
    print(f"Workdir: {workdir}\n")

    # Stage 1 — write source files
    kb_path = _materialize_kb(workdir)
    print(f"KB has {len(KB_DOCS)} source file(s) at {kb_path}\n")

    # Stage 2 — load, chunk, embed (single line via from_chunks).
    print("Indexing...", end=" ", flush=True)
    global _STORE
    _STORE = await _build_index_from_kb(kb_path)
    chunk_count = await _STORE.count()
    print(f"✓ ({chunk_count} chunks indexed)\n")

    # Stage 3-4 — run the multi-agent pipeline.
    # Team.supervisor produces an Agent whose architecture is a
    # configured Supervisor; same effect as the explicit
    # ``Agent(architecture=Supervisor(...))`` form, just with a
    # builder shape that mirrors LangGraph's create_supervisor /
    # CrewAI's hierarchical Crew / AutoGen's GroupChatManager.
    agent = Team.supervisor(
        workers={
            "researcher": researcher,
            "writer": writer,
            "reviewer": reviewer,
        },
        instructions=(
            "You are a research project manager. You answer "
            "questions about agent architectures by coordinating "
            "a team.\n\n"
            "Pipeline:\n"
            "1. Delegate to `researcher` — give specific search "
            "queries. The researcher will search the KB and quote "
            "verbatim chunks.\n"
            "2. Delegate to `writer` — paste the researcher's "
            "verbatim findings into the writer's instructions and "
            "tell them to save to `comparison.md`.\n"
            "3. Delegate to `reviewer` — paste the file path; ask "
            "for issues + EXACT replacement text.\n"
            "4. Delegate to `writer` again to apply fixes via "
            "`edit`. Pass the reviewer's verbatim suggestions.\n"
            "5. Briefly summarize the final report (and its file "
            "path) in your response."
        ),
        model="gpt-4.1-mini",
    )

    prompt = (
        "Compare Router and ReWOO architectures: design, cost, "
        "when to use each, and failure modes. Save to "
        "`comparison.md` and have it reviewed before returning."
    )
    print(f"Question: {prompt}\n")

    delegation_count = 0
    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk_data = ev.payload.get("chunk", {})
            if (
                chunk_data.get("kind") == "text"
                and chunk_data.get("text")
            ):
                print(chunk_data["text"], end="", flush=True)
        elif kind == "tool_call":
            call = ev.payload.get("call", {})
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            if tool_name == "delegate":
                delegation_count += 1
                worker = args.get("worker", "?")
                preview = args.get("instructions", "")[:80]
                print(
                    f"\n\n┌─ [supervisor → {worker}] (delegation "
                    f"#{delegation_count})"
                )
                print(f"│  {preview}...")
                print("└─")
            else:
                arg_str = ", ".join(
                    f"{k}={str(v)[:50]!r}"
                    for k, v in args.items()
                )
                print(f"\n  ◇ [tool] {tool_name}({arg_str})")
        elif kind == "tool_result":
            output = (ev.payload.get("result", {}).get("output") or "")[:120]
            print(f"  → {output}")
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n\n" + "=" * 70)
            print("FINAL SUMMARY")
            print("=" * 70)
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}"
            )

    # Show the produced report.
    # The built-in tools default to the framework's tempdir, so the
    # writer's output is at default_workdir() / "comparison.md".
    from jeevesagent import default_workdir

    output_path = default_workdir() / "comparison.md"
    print("\n" + "=" * 70)
    print(f"OUTPUT FILE ({output_path})")
    print("=" * 70)
    if output_path.exists():
        print(output_path.read_text())
    else:
        # Defensive: maybe the writer wrote to the kb workdir or
        # elsewhere; scan both.
        candidates = list(default_workdir().rglob("*.md")) + list(
            workdir.rglob("comparison.md")
        )
        for c in candidates:
            print(f"\n--- {c} ---")
            print(c.read_text())

    print(
        f"\n(KB at {kb_path}, output at {default_workdir()} — "
        "kept for inspection.)"
    )


if __name__ == "__main__":
    asyncio.run(main())
