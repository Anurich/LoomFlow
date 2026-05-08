"""Example 1 — RAG over a folder of PDFs.

End-to-end pipeline using only JeevesAgent's own building blocks:

    PDFs in a folder
       │
       ▼
    jeevesagent.loader.load(pdf_path)            ← built-in PDF loader
       │  → Document(content=markdown, metadata)
       ▼
    RecursiveChunker.split(content)              ← built-in chunker
       │  → list[Chunk]
       ▼
    ChromaVectorStore.add(chunks)                ← built-in vector store
       │     (embedder = OpenAIEmbedder)
       ▼
    @tool search_docs(query)                     ← retriever tool
       │     wraps store.search(query, k=4)
       ▼
    Agent("answer using the retriever", tools=[search_docs])
       │
       ▼
    answer

Run::

    OPENAI_API_KEY=sk-... python examples/01_rag_pdf.py

The first run generates four sample PDFs in ``examples/data/general/``
(via ``reportlab``) and indexes them into a persistent Chroma
collection. Re-runs reuse the on-disk index, so only the agent loop
re-runs against OpenAI.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ Skipping: OPENAI_API_KEY is not set. "
        "Export it (or add it to .env) to run this example.\n"
    )
    sys.exit(0)


# --------------------------------------------------------------------
# 0. Sample PDFs — generated once, cached on disk
# --------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data" / "general"

# Each entry → one PDF. ``filename: paragraphs``.
SAMPLE_PDFS: dict[str, list[str]] = {
    "company_handbook.pdf": [
        "Acme Corp was founded in 2008 and is headquartered in Berlin.",
        "Acme builds open-source observability tools for distributed systems. "
        "Our flagship product, AcmeTrace, is used by over 4,000 organizations.",
        "Acme's CEO is Mira Castellanos. Our annual offsite is held in Lisbon "
        "every September.",
    ],
    "engineering_guide.pdf": [
        "All production deployments at Acme go through our internal CI/CD "
        "system, called Forge, which runs on Kubernetes.",
        "Forge enforces three required gates: (1) unit tests must pass, "
        "(2) static analysis must report zero high-severity findings, and "
        "(3) at least one human reviewer must approve the change.",
        "Hotfix deploys bypass the static-analysis gate but still require "
        "tests + a reviewer. They are tracked under the 'hotfix' label "
        "and reviewed weekly by the platform-reliability team.",
    ],
    "security_policy.pdf": [
        "Acme's security policy mandates that all employee laptops use "
        "full-disk encryption (FileVault on macOS, BitLocker on Windows).",
        "Production database access requires hardware-backed MFA and is "
        "logged in the audit-trail system. Access expires after 14 days "
        "and must be re-requested via the access portal.",
        "Reporting a security incident: email security@acme.example "
        "or page the on-call security engineer via PagerDuty.",
    ],
    "support_runbook.pdf": [
        "Customer support tickets are triaged into P1 (production down), "
        "P2 (major feature broken), P3 (minor issue), and P4 (question / "
        "feature request).",
        "P1 tickets must be acknowledged within 15 minutes and resolved or "
        "have a workaround within 4 hours, around the clock.",
        "Escalation path for P1: on-call support engineer → support manager "
        "→ VP of customer experience.",
    ],
}


def _ensure_sample_pdfs() -> Path:
    """Generate sample PDFs into DATA_DIR if missing. Returns the dir."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        f for f in SAMPLE_PDFS if not (DATA_DIR / f).exists()
    ]
    if not missing:
        return DATA_DIR

    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ImportError(
            "reportlab is required to generate sample PDFs. "
            "Install with: pip install reportlab"
        ) from exc

    print(f"  Generating {len(missing)} sample PDF(s) in {DATA_DIR}...")
    for filename in missing:
        paragraphs = SAMPLE_PDFS[filename]
        path = DATA_DIR / filename
        c = canvas.Canvas(str(path), pagesize=LETTER)
        width, height = LETTER
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 72, filename.replace("_", " ").rsplit(".", 1)[0].title())
        c.setFont("Helvetica", 11)
        y = height - 110
        for para in paragraphs:
            for line in _wrap(para, 90):
                c.drawString(72, y, line)
                y -= 16
            y -= 8
        c.showPage()
        c.save()
    return DATA_DIR


def _wrap(text: str, width: int) -> list[str]:
    """Naive word-wrap — keeps the example dependency-free."""
    out: list[str] = []
    line = ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word) if line else word
    if line:
        out.append(line)
    return out


# --------------------------------------------------------------------
# 1. Build the index
# --------------------------------------------------------------------

from jeevesagent import Agent, tool  # noqa: E402
from jeevesagent.loader import RecursiveChunker, load  # noqa: E402
from jeevesagent.memory.embedder import OpenAIEmbedder  # noqa: E402
from jeevesagent.vectorstore import ChromaVectorStore  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parent / "data" / ".chroma_general"


async def build_or_load_index() -> ChromaVectorStore:
    pdf_dir = _ensure_sample_pdfs()
    embedder = OpenAIEmbedder("text-embedding-3-small")
    store = ChromaVectorStore(
        embedder=embedder,
        collection_name="general_docs",
        persist_directory=str(INDEX_DIR),
    )

    # Skip re-indexing if the collection already has rows on disk.
    existing = store._collection.count()  # type: ignore[attr-defined]
    if existing:
        print(f"  Reusing on-disk Chroma index with {existing} chunks.")
        return store

    chunker = RecursiveChunker(chunk_size=600, chunk_overlap=80)
    all_chunks = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = load(pdf)
        chunks = chunker.split(doc.content, source=str(pdf))
        for ch in chunks:
            ch.metadata["source_file"] = pdf.name
        all_chunks.extend(chunks)
        print(f"  {pdf.name:30s} → {len(chunks)} chunks")

    await store.add(all_chunks)
    print(f"  Indexed {len(all_chunks)} chunks into Chroma.")
    return store


# --------------------------------------------------------------------
# 2. Retriever tool
# --------------------------------------------------------------------


def make_retriever(store: ChromaVectorStore):
    @tool(name="search_docs")
    async def search_docs(query: str) -> str:
        """Search the company knowledge base.

        Returns the top 4 passages most relevant to ``query``,
        each prefixed with its source filename. Use this whenever
        the user asks about company policies, products, or
        procedures.
        """
        results = await store.search(query, k=4)
        if not results:
            return "(no matching passages)"
        out: list[str] = []
        for i, r in enumerate(results, 1):
            src = r.chunk.metadata.get("source_file", "?")
            out.append(f"[{i}] ({src})\n{r.chunk.content.strip()}")
        return "\n\n".join(out)

    return search_docs


# --------------------------------------------------------------------
# 3. The agent
# --------------------------------------------------------------------


async def main() -> None:
    print("\n  Example 1 — RAG over a folder of PDFs\n")
    store = await build_or_load_index()
    retriever = make_retriever(store)

    agent = Agent(
        "You are a precise assistant for Acme Corp. Always call "
        "search_docs to ground your answer in the indexed knowledge "
        "base; quote the source filename for any factual claim. If "
        "the docs do not cover the question, say so.",
        model="gpt-4.1-mini",
        tools=[retriever],
    )

    questions = [
        "When was Acme founded and where is it headquartered?",
        "What are the three required gates for a production deploy?",
        "How fast must a P1 support ticket be acknowledged, and who is "
        "the final escalation step?",
    ]

    for q in questions:
        print("─" * 72)
        print(f"Q: {q}")
        result = await agent.run(q)
        print(f"A: {result.output}")
        print(f"   ({result.turns} turns, {result.tokens_in}+{result.tokens_out} tokens)")
    print("─" * 72)


if __name__ == "__main__":
    asyncio.run(main())
