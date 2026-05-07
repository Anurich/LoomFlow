"""20_rag_supervisor — Production-shape RAG with three agents.

Three worker Agents coordinated by a :class:`Supervisor`:

* **Researcher** — searches an in-memory corpus via cosine similarity
  over :class:`HashEmbedder` vectors, then fetches full doc text on
  demand.
* **Curator** — verifies that quotes the researcher found actually
  appear in the cited docs. Catches the LLM-hallucinating-citations
  failure mode that breaks production RAG.
* **Synthesizer** — composes the final markdown brief using only
  verified material.

The supervisor delegates Researcher → Curator → Synthesizer in turn.
Each worker has its own tool host; tool calls inside each worker
fire in parallel where the model emits multiple in one turn.

The corpus contains a deliberate inconsistency — a DRAFT and a
FINAL meeting note with different engineering budget numbers —
so the Curator's role visibly matters. A naive RAG implementation
would mix the two.

Why this example REQUIRES a real LLM
------------------------------------

Coordinating three workers + a supervisor with deterministic
:class:`ScriptedModel` would mean ~15 pre-baked turns (each
worker + supervisor + tool calls) — fragile and unreadable. Set
``OPENAI_API_KEY`` in your environment (or in a ``.env`` file at
the repo root); the other 19 examples in this folder demonstrate
the framework's behaviour without API keys.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/20_rag_supervisor.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Optional: load OPENAI_API_KEY from .env at the repo root if present.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from jeevesagent import (
    Agent,
    HashEmbedder,
    InMemoryVectorStore,
    Team,
    tool,
)
from jeevesagent.core.types import ToolCall  # noqa: F401 — used in docstrings
from jeevesagent.loader import Chunk

# ---------------------------------------------------------------------------
# Fake corpus — 6 internal docs from "Acme Corp" Q3 planning.
#
# Note the meeting_q3_planning_DRAFT vs meeting_q3_planning_FINAL —
# the draft has $1.2M for engineering; the final has $1.5M. This is
# the test case for the Curator's role.
# ---------------------------------------------------------------------------

CORPUS: dict[str, str] = {
    "meeting_q3_planning_DRAFT.md": (
        "STATUS: DRAFT — superseded by meeting_q3_planning_FINAL.md\n"
        "Date: 2026-04-10\n\n"
        "Q3 budget proposal (DRAFT):\n"
        "- Engineering: $1.2M\n"
        "- Marketing: $400K\n"
        "- Operations: $200K\n\n"
        "Action items: revisit engineering allocation given hiring plan."
    ),
    "meeting_q3_planning_FINAL.md": (
        "STATUS: FINAL — approved 2026-04-15\n"
        "Date: 2026-04-15\n\n"
        "Q3 budget (FINAL, approved):\n"
        "- Engineering: $1.5M\n"
        "- Marketing: $400K\n"
        "- Operations: $200K\n\n"
        "Approved by: CEO, CTO, head of finance.\n"
        "Engineering allocation revised upward from draft to fund "
        "agent-harness team expansion."
    ),
    "product_roadmap_q3.md": (
        "STATUS: FINAL\n"
        "Q3 product priorities:\n"
        "1. Ship JeevesAgent harness v1.0 (engineering-led).\n"
        "2. Launch beta program (marketing-led, 50 design partners).\n"
        "3. Stabilize MCP gateway (engineering-led, sub-priority)."
    ),
    "finance_q2_summary.md": (
        "STATUS: FINAL\n"
        "Q2 actuals:\n"
        "- Revenue: $480K (vs $400K target; +20%)\n"
        "- Burn rate: $850K/month\n"
        "- Cash on hand at end of Q2: $4.2M\n"
        "- Runway: ~5 months at current burn"
    ),
    "hiring_policy.md": (
        "STATUS: FINAL\n"
        "Engineering hires require sign-off from BOTH the head of "
        "engineering AND the CTO. Marketing and ops hires require "
        "the CEO's approval only. All offers must be approved before "
        "extending."
    ),
    "strategy_2027.md": (
        "STATUS: FINAL\n"
        "Three-year goal: be the default agent harness for "
        "production teams. North-star metric: weekly active "
        "production agents using JeevesAgent."
    ),
}


# ---------------------------------------------------------------------------
# Vector store. ``InMemoryVectorStore`` owns the embed/store/search loop —
# no manual cosine, no parallel lists. We tag each chunk's metadata with
# its ``status`` (DRAFT / FINAL) so callers can pre-filter the search via
# Mongo-style operators (``filter={"status": "FINAL"}``). Swap the class
# for ``ChromaVectorStore`` / ``PostgresVectorStore`` / ``FAISSVectorStore``
# for production with no other code changes.
# ---------------------------------------------------------------------------


_STORE = InMemoryVectorStore(embedder=HashEmbedder(dimensions=256))


def _status_of(content: str) -> str:
    """Pull the STATUS line so we can index it as metadata."""
    return "DRAFT" if "STATUS: DRAFT" in content else "FINAL"


async def _build_index() -> None:
    """Embed every doc once at startup. Each chunk's metadata carries
    the doc_id + status so search results can be filtered by status."""
    chunks = [
        Chunk(
            content=content,
            metadata={"doc_id": doc_id, "status": _status_of(content)},
        )
        for doc_id, content in CORPUS.items()
    ]
    await _STORE.add(chunks)


# ---------------------------------------------------------------------------
# Tracking what each agent has fetched (so curator can list it).
# ---------------------------------------------------------------------------

_LOADED_DOCS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Researcher tools
# ---------------------------------------------------------------------------


@tool
async def search_corpus(query: str, finals_only: bool = False) -> str:
    """Search the internal doc corpus by semantic similarity.

    Returns the top-3 matching doc IDs with similarity scores and a
    snippet from each. Pass ``finals_only=true`` to skip docs marked
    STATUS: DRAFT — useful when you only want approved material.
    """
    filter_expr = {"status": "FINAL"} if finals_only else None
    results = await _STORE.search(query, k=3, filter=filter_expr)
    if not results:
        return "(no matches)"
    lines = []
    for r in results:
        doc_id = r.chunk.metadata["doc_id"]
        snippet = r.chunk.content.replace("\n", " ")[:120]
        status = r.chunk.metadata.get("status", "?")
        lines.append(
            f"  - {doc_id} [{status}] (score={r.score:.3f}): {snippet}..."
        )
    return "Matches:\n" + "\n".join(lines)


@tool
async def fetch_full_doc(doc_id: str) -> str:
    """Get the full text of a document by id. Records that this
    doc has been loaded so the curator can reference it."""
    if doc_id not in CORPUS:
        return f"ERROR: unknown doc_id {doc_id!r}"
    _LOADED_DOCS[doc_id] = CORPUS[doc_id]
    return f"=== {doc_id} ===\n{CORPUS[doc_id]}"


# ---------------------------------------------------------------------------
# Curator tools
# ---------------------------------------------------------------------------


@tool
async def list_loaded_docs() -> str:
    """List the doc IDs the researcher has fetched so far."""
    if not _LOADED_DOCS:
        return "(no docs loaded yet)"
    return "Loaded docs:\n" + "\n".join(
        f"  - {d}" for d in _LOADED_DOCS
    )


@tool
async def verify_quote(doc_id: str, quote: str) -> str:
    """Verify that a quote literally appears in the named doc.
    Returns 'verified' with context if found, 'not found' otherwise.
    Substring match is case-insensitive and whitespace-tolerant."""
    if doc_id not in CORPUS:
        return f"not found: unknown doc_id {doc_id!r}"
    haystack = " ".join(CORPUS[doc_id].split()).lower()
    needle = " ".join(quote.split()).lower()
    if needle in haystack:
        return f"verified: '{quote}' appears in {doc_id}"
    return f"not found: '{quote}' is NOT in {doc_id}"


# ---------------------------------------------------------------------------
# Synthesizer tools
# ---------------------------------------------------------------------------


@tool
async def format_brief(
    headline: str, body: str, citations: str
) -> str:
    """Compose a structured markdown brief with citations."""
    return (
        f"# {headline}\n\n"
        f"{body}\n\n"
        f"## Citations\n{citations}"
    )


# ---------------------------------------------------------------------------
# Build the four agents.
# ---------------------------------------------------------------------------


def _build_agents() -> Agent:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "\n  ✗ OPENAI_API_KEY is not set.\n\n"
            "  This example requires a real LLM to coordinate the\n"
            "  three workers + supervisor. Set OPENAI_API_KEY in your\n"
            "  environment, or add OPENAI_API_KEY=sk-... to a .env\n"
            "  file at the repo root.\n\n"
            "  For an API-key-free example, see examples/12_supervisor.py.\n"
        )
        sys.exit(1)

    model_name = "gpt-4.1-mini"

    researcher = Agent(
        instructions=(
            "You research questions by searching the internal "
            "doc corpus.\n\n"
            "Process:\n"
            "1. Call `search_corpus(query)` with a focused query "
            "from the supervisor's instruction. Pass "
            "`finals_only=true` if the question wants approved / "
            "current numbers (skips DRAFT docs at the index level).\n"
            "2. For each promising hit, call `fetch_full_doc(doc_id)` "
            "to read the full text. Always fetch the FULL doc — "
            "snippets are previews only.\n"
            "3. Return: a list of doc IDs you fetched, plus the "
            "specific passages (verbatim, in quotes) that answer "
            "the question. Note any doc that says STATUS: DRAFT — "
            "drafts may be superseded.\n\n"
            "Be precise; the curator will verify every quote you "
            "claim, so don't paraphrase."
        ),
        model=model_name,
        tools=[search_corpus, fetch_full_doc],
    )

    curator = Agent(
        instructions=(
            "You are a citation curator. The researcher has fetched "
            "documents and proposed quotes from them. Your job:\n\n"
            "1. Call `list_loaded_docs()` to see what was fetched.\n"
            "2. For EACH quote the researcher proposed, call "
            "`verify_quote(doc_id, quote)`. If verification fails, "
            "say so explicitly.\n"
            "3. Flag any quote that comes from a doc marked "
            "STATUS: DRAFT — drafts are superseded; prefer FINAL.\n\n"
            "Return a list of verified, FINAL-status quotes with "
            "their doc IDs. Drop anything unverified or from a "
            "draft."
        ),
        model=model_name,
        tools=[verify_quote, list_loaded_docs],
    )

    synthesizer = Agent(
        instructions=(
            "You compose final briefs using only verified, "
            "FINAL-status material from the curator. Use the "
            "`format_brief(headline, body, citations)` tool to "
            "produce the final markdown.\n\n"
            "Cite every factual claim with the doc ID in brackets "
            "like [meeting_q3_planning_FINAL.md]. Don't invent "
            "facts; if the verified material is insufficient, say "
            "so plainly."
        ),
        model=model_name,
        tools=[format_brief],
    )

    return Team.supervisor(
        workers={
            "researcher": researcher,
            "curator": curator,
            "synthesizer": synthesizer,
        },
        instructions=(
            "You answer questions about Acme Corp's internal "
            "documents by coordinating a research team.\n\n"
            "Workflow:\n"
            "1. Delegate to `researcher` first — give it a "
            "specific search query derived from the user's question.\n"
            "2. Once the researcher returns findings (doc IDs + "
            "quotes), delegate to `curator` with those quotes for "
            "verification. PASTE THE RESEARCHER'S OUTPUT into the "
            "curator's instructions so the curator has something to "
            "verify.\n"
            "3. Once the curator returns verified quotes, delegate "
            "to `synthesizer` to compose the final answer. Pass "
            "the curator's verified quotes to the synthesizer.\n"
            "4. Return the synthesizer's final brief as your answer."
        ),
        model=model_name,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=== Building corpus index... ===")
    await _build_index()
    print(f"  ✓ indexed {len(CORPUS)} docs\n")

    agent = _build_agents()

    question = (
        "What's our engineering budget for Q3? "
        "Cite the source doc."
    )
    print(f"=== Question ===\n{question}\n")

    print("=== Streaming events ===")
    final_result: dict[str, object] = {}
    async for event in agent.stream(question):
        if event.kind == "tool_call":
            call = event.payload.get("call", {})
            tool_name = call.get("tool", "?")
            args = call.get("args", {})
            if tool_name == "delegate":
                worker = args.get("worker", "?")
                preview = args.get("instructions", "")[:60]
                print(f"\n[supervisor → {worker}] {preview}...")
            else:
                arg_preview = ", ".join(
                    f"{k}={str(v)[:40]!r}" for k, v in args.items()
                )
                print(f"  [tool] {tool_name}({arg_preview})")
        elif event.kind == "tool_result":
            result = event.payload.get("result", {})
            output = (result.get("output") or "")[:120]
            print(f"  [→] {output}...")
        elif event.kind == "completed":
            # The COMPLETED event's payload carries the full RunResult
            # dump (output + turns + tokens + cost). Capture so we can
            # print the final answer without re-running (LLM
            # nondeterminism would otherwise produce a different answer
            # on a second run).
            final_result = event.payload.get("result", {}) or {}

    print("\n=== Final answer ===")
    print(final_result.get("output", "(no output)"))
    print(
        f"\nTurns: {final_result.get('turns', '?')}  "
        f"Tokens: in={final_result.get('tokens_in', '?')} "
        f"out={final_result.get('tokens_out', '?')}  "
        f"Cost: ${float(final_result.get('cost_usd', 0) or 0):.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
