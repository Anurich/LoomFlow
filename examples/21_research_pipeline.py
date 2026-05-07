"""21_research_pipeline — Plan, parallel-research, write, review, update.

What it shows
-------------

This is the showcase example: a "Claude-Code-shaped" research agent
that plans, decomposes, executes work in parallel where possible,
mutates real files on disk, reviews its own output, and iterates
on fixes.

Architecture stack:

* **Top:** ``Supervisor`` orchestrating three workers.
* **Workers:** each a single-agent ``ReAct`` with its own tool host.
    - ``researcher`` — RAG over an in-memory corpus
        tools: ``search_corpus``, ``fetch_doc``
    - ``writer`` — real markdown file I/O via the **built-in
      filesystem tools**: ``read``, ``write``, ``edit``
    - ``reviewer`` — reads the writer's output and produces a
      critique
        tools: ``read``
* **Built-in tools.** This example uses
  :func:`~jeevesagent.read_tool`, :func:`~jeevesagent.write_tool`,
  and :func:`~jeevesagent.edit_tool` — the canonical
  Claude-Code-shaped tool set shipped in v0.4. They share the
  framework's :func:`~jeevesagent.default_workdir` automatically,
  so write→edit→read across multiple workers all hit the same
  file. No path-resolution boilerplate; no hand-rolled section
  editing — ``edit`` does exact-string find-and-replace, the same
  approach Claude Code takes.
* **Parallel execution:** when a worker's model emits multiple tool
  calls in one turn (e.g. researcher calls ``search_corpus`` for
  several topics simultaneously), they fire in parallel via
  ReAct's ``anyio.create_task_group`` dispatch — for free.
* **Streaming + observability:** every model token, tool call, tool
  result, and architecture event flows through ``agent.stream(...)``
  — the typewriter UI works end-to-end across all three nesting
  levels.

What it produces
----------------

A polished comparison report on disk at
``${TMPDIR}/jeeves_agent_*/research_report.md``, with sections for
each architecture, cross-references, and a final summary. The
example prints the path so you can ``cat`` the result.

Run::

    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/21_research_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "\n  ✗ OPENAI_API_KEY required. "
        "Add OPENAI_API_KEY=sk-... to .env at repo root.\n"
    )

from jeevesagent import (  # noqa: E402
    Agent,
    HashEmbedder,
    InMemoryVectorStore,
    Team,
    default_workdir,
    edit_tool,
    read_tool,
    tool,
    write_tool,
)
from jeevesagent.loader import Chunk  # noqa: E402

# ---------------------------------------------------------------------------
# In-memory corpus — fake "JeevesAgent v0.3 architecture docs". In
# production, swap for a real vector index (ChromaMemory / PostgresMemory
# with OpenAIEmbedder).
# ---------------------------------------------------------------------------

CORPUS: dict[str, str] = {
    "react.md": (
        "# ReAct\n"
        "ReAct is the canonical observe-think-act loop. The model\n"
        "calls tools when needed and produces a final response when\n"
        "it doesn't. Cost: 1 LLM call per turn. Best for: general\n"
        "agents with mixed tool use. Failure mode: goal drift on\n"
        "long tasks (>10 turns)."
    ),
    "router.md": (
        "# Router\n"
        "Router classifies the user's request once via a small fast\n"
        "model and dispatches to ONE specialist Agent. Cost: 1\n"
        "classifier call + 1 specialist run. Best for: customer\n"
        "support / helpdesk / API gateway. Failure mode: routing\n"
        "errors are total; no recovery if the wrong route is picked."
    ),
    "supervisor.md": (
        "# Supervisor\n"
        "Supervisor coordinates a team of worker Agents via a\n"
        "`delegate(worker, instructions)` tool. Multiple delegations\n"
        "in one supervisor turn run in parallel. Cost: 1.5-3x\n"
        "single-agent. Best for: multi-domain tasks needing\n"
        "specialist roles. Failure mode: supervisor is a bottleneck\n"
        "if the model emits poor delegation instructions."
    ),
    "rewoo.md": (
        "# ReWOO\n"
        "ReWOO emits a structured plan of tool calls upfront with\n"
        "{{En}} placeholder substitution. Independent steps run in\n"
        "parallel. Cost: 2 LLM calls + N tool calls (vs ReAct's N+1\n"
        "LLM calls). 30-50% cheaper for tool-heavy predictable\n"
        "workloads. Failure mode: planner must predict accurately;\n"
        "no replanning on failure in v1."
    ),
    "actor_critic.md": (
        "# ActorCritic\n"
        "ActorCritic requires two separate Agents: actor generates,\n"
        "critic reviews adversarially with structured JSON output\n"
        "(issues + 0-1 score). Below threshold, actor refines. Cost:\n"
        "2-5x single-pass. Best for: code generation, security-\n"
        "critical writing, important documentation. Use DIFFERENT\n"
        "models for genuine blind-spot diversity."
    ),
    "reflexion.md": (
        "# Reflexion\n"
        "Reflexion wraps any base architecture with verbal RL: an\n"
        "evaluator scores each attempt, a reflector emits a one-\n"
        "sentence lesson on failure, the lesson persists to a\n"
        "memory block visible to the next attempt. Cross-session\n"
        "learning when paired with a persistent memory backend.\n"
        "Cost: 1.5-3x base."
    ),
}


# ---------------------------------------------------------------------------
# Vector store — single-line setup via the framework's
# ``InMemoryVectorStore``. Replaces the manual cosine + parallel-list
# scaffolding the older examples carried. Swap the class for
# ``ChromaVectorStore`` / ``PostgresVectorStore`` / ``FAISSVectorStore``
# for production with no other code changes.
# ---------------------------------------------------------------------------

_STORE = InMemoryVectorStore(embedder=HashEmbedder(dimensions=256))


async def _build_index() -> None:
    """Embed every doc once at startup."""
    chunks = [
        Chunk(content=content, metadata={"doc_id": doc_id})
        for doc_id, content in CORPUS.items()
    ]
    await _STORE.add(chunks)


# ---------------------------------------------------------------------------
# Researcher tools (RAG over the in-memory corpus)
# ---------------------------------------------------------------------------


@tool
async def search_corpus(query: str) -> str:
    """Search the architecture corpus by semantic similarity.
    Returns the top-3 matching doc IDs with cosine scores and a
    one-line snippet from each."""
    results = await _STORE.search(query, k=3)
    if not results:
        return "ERROR: index empty"
    lines = []
    for r in results:
        doc_id = r.chunk.metadata["doc_id"]
        first_para = CORPUS[doc_id].split("\n", 2)[1]
        lines.append(
            f"  - {doc_id} (score={r.score:.3f}): {first_para[:120]}..."
        )
    return "\n".join(lines)


@tool
def fetch_doc(doc_id: str) -> str:
    """Fetch the full text of a corpus doc by id."""
    if doc_id not in CORPUS:
        return f"ERROR: unknown doc_id {doc_id!r}"
    return CORPUS[doc_id]


# ---------------------------------------------------------------------------
# Workers — each gets its own slice of the built-in tool set.
#
# Critical: read_tool() / write_tool() / edit_tool() called WITHOUT
# arguments all share `default_workdir()` — a single tempdir under
# /tmp/jeeves_agent_*. So when the writer creates a file and the
# reviewer reads it, they're hitting the same file on disk. No
# explicit path threading required.
# ---------------------------------------------------------------------------


researcher = Agent(
    instructions=(
        "You research questions by searching the architecture "
        "corpus. Process:\n"
        "1. Call `search_corpus` (in parallel where possible — "
        "you can issue multiple calls in one turn) for each topic "
        "you need to cover.\n"
        "2. For each promising hit, call `fetch_doc` to read the "
        "full text.\n"
        "3. Return a tight summary of findings: doc IDs + key "
        "passages (verbatim, in quotes), one paragraph per topic."
    ),
    model="gpt-4.1-mini",
    tools=[search_corpus, fetch_doc],
)

writer = Agent(
    instructions=(
        "You write structured markdown reports.\n\n"
        "Tool playbook:\n"
        "* `write(path, content)` — for creating a new file or "
        "fully overwriting an existing one.\n"
        "* `read(path)` — to verify what's on disk. Output is "
        "line-numbered (format `   N\\tline`), but the line "
        "numbers are NOT part of the file content; ignore them "
        "when planning edits.\n"
        "* `edit(path, old_string, new_string)` — for in-place "
        "fixes. ``old_string`` must match the file's contents "
        "EXACTLY (whitespace, indentation, line breaks). It must "
        "also be UNIQUE in the file (or pass `replace_all=true`).\n"
        "  - To get the exact old_string, ALWAYS `read` the file "
        "first and copy the relevant lines verbatim (without the "
        "leading line-number / tab).\n"
        "  - If the same string appears multiple times, include "
        "more surrounding context to make the match unique.\n\n"
        "Always verify with `read` after writing or editing. Keep "
        "report files focused — under 200 lines."
    ),
    model="gpt-4.1-mini",
    tools=[write_tool(), read_tool(), edit_tool()],
)

reviewer = Agent(
    instructions=(
        "You review markdown reports for completeness, accuracy, "
        "and structural quality. Process:\n"
        "1. Call `read(path)` on the file the supervisor named.\n"
        "2. Return a bulleted list of issues. For each issue, "
        "give the writer everything they need to fix it via "
        "`edit`:\n"
        "   - the EXACT current text to replace (copy from `read`, "
        "minus the line-number prefix), and\n"
        "   - the proposed replacement text.\n"
        "Be specific. Don't paraphrase the existing text; the "
        "writer needs an exact substring to feed to ``edit``."
    ),
    model="gpt-4.1-mini",
    tools=[read_tool()],
)


# ---------------------------------------------------------------------------
# Top-level supervisor
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = default_workdir()  # shared tempdir; created lazily on first call

    print("=" * 70)
    print("Research Pipeline — Plan, parallel-research, write, review, update")
    print("=" * 70)
    print(f"Workdir (shared by writer + reviewer): {workdir}")

    print("Building semantic index...", end=" ", flush=True)
    await _build_index()
    print(f"✓ ({len(CORPUS)} docs)\n")

    agent = Team.supervisor(
        workers={
            "researcher": researcher,
            "writer": writer,
            "reviewer": reviewer,
        },
        instructions=(
            "You are a research project manager. You answer "
            "research questions by coordinating a team of "
            "specialists.\n\n"
            "Pipeline:\n"
            "1. Delegate to `researcher` first — give a specific "
            "search query for the topics in the user's question.\n"
            "2. Delegate to `writer` once findings are in — paste "
            "the researcher's findings into the writer's "
            "instructions and tell them which file path to write "
            "(e.g. `research_report.md`). Ask the writer to use "
            "the `write` tool.\n"
            "3. Delegate to `reviewer` — paste the file path and "
            "ask for issues + EXACT replacement text the writer "
            "should pass to `edit`.\n"
            "4. Delegate to `writer` again to apply the reviewer's "
            "fixes via `edit`. Pass the reviewer's verbatim "
            "old_string / new_string suggestions through unchanged.\n"
            "5. Briefly summarize the final report in your last "
            "response, including the file path.\n\n"
            "Be specific in delegation instructions; workers do NOT "
            "see the user's original question, only what you write "
            "to them."
        ),
        model="gpt-4.1-mini",
    )

    prompt = (
        "Write a comparison of three JeevesAgent v0.3 architectures: "
        "Router, Supervisor, and ReWOO. Focus on the cost/quality "
        "trade-off and when to use each. Save the report at "
        "`research_report.md` and have it reviewed before returning."
    )

    print(f"Question: {prompt}\n")

    delegation_count = 0
    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
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
            result = ev.payload.get("result", {})
            output = (result.get("output") or "")[:100]
            err = result.get("error")
            if err:
                print(f"  ✗ ERROR: {err}")
            else:
                print(f"  → {output}")
        elif kind == "architecture_event":
            name = ev.payload.get("name", "")
            if name == "supervisor.workers_ready":
                workers = ev.payload.get("workers", [])
                print(f"[team ready: {workers}]\n")
            elif name == "supervisor.completed":
                print("\n\n[supervisor pipeline complete]")
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

    # Show what was actually produced on disk
    print("\n" + "=" * 70)
    print(f"FILES PRODUCED in {workdir}")
    print("=" * 70)
    files = sorted(p for p in workdir.rglob("*") if p.is_file())
    for f in files:
        size = f.stat().st_size
        rel = f.relative_to(workdir)
        print(f"  {rel}  ({size} bytes)")
        print("  " + "-" * 60)
        for line in f.read_text().splitlines()[:80]:
            print(f"    {line}")
        if len(f.read_text().splitlines()) > 80:
            extra = len(f.read_text().splitlines()) - 80
            print(f"    ... ({extra} more lines)")
        print("  " + "-" * 60)
    print(f"\n(Workdir kept at {workdir} for inspection.)")


if __name__ == "__main__":
    asyncio.run(main())
