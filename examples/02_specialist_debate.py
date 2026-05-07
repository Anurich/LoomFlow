"""Example 2 — Five domain specialists, each with their own RAG, debating.

Five specialist sub-Agents — IT technician, physicist, physician,
financial analyst, lawyer — each grounded in their own folder of PDFs:

    examples/data/it/         → indexed → Chroma collection 'it_docs'
    examples/data/physics/    → indexed → Chroma collection 'physics_docs'
    examples/data/medicine/   → indexed → Chroma collection 'medicine_docs'
    examples/data/finance/    → indexed → Chroma collection 'finance_docs'
    examples/data/law/        → indexed → Chroma collection 'law_docs'

Each specialist gets a retriever tool scoped to its own collection
and a system prompt forbidding it from answering outside its
domain. The five specialists are then composed via the **debate**
architecture: each takes a turn, they cross-react, and a judge
agent synthesises a single grounded answer.

    Team.debate(
        debaters=[it, phys, med, fin, law],
        judge=judge,
        rounds=2,
    )

Run::

    OPENAI_API_KEY=sk-... python examples/02_specialist_debate.py

The first run generates one PDF per domain (via reportlab) and
indexes each into its own persistent Chroma collection. Re-runs
reuse the on-disk indices.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print("\n  ✗ Set OPENAI_API_KEY in your environment or .env file.\n")
    sys.exit(1)


# --------------------------------------------------------------------
# 0. Domain definitions + sample-PDF content
# --------------------------------------------------------------------


@dataclass
class Domain:
    key: str               # short id, used for paths + collection name
    role: str              # human-readable role for the system prompt
    pdf_filename: str      # the one PDF that gets generated for the domain
    pdf_paragraphs: list[str]


DOMAINS: list[Domain] = [
    Domain(
        key="it",
        role="senior IT support technician",
        pdf_filename="it_runbook.pdf",
        pdf_paragraphs=[
            "When a user reports their laptop will not turn on, the standard "
            "first-line check is: (1) confirm the power adapter LED is lit, "
            "(2) try a known-good charging cable, (3) hold the power button "
            "for 30 seconds to discharge residual power, then reattach "
            "the adapter and retry.",
            "For corporate Windows laptops with BitLocker, the recovery key "
            "is escrowed in the IT identity portal under 'Devices > "
            "Recovery Keys'. Provide the recovery key to the user only "
            "after verifying their identity via a second channel (Slack "
            "DM plus phone call).",
            "If a laptop powers on but the display is blank, the most "
            "common cause is a stuck firmware state. Connect an external "
            "monitor to confirm the GPU is alive; if the external display "
            "works, reseat the internal display ribbon (technician-only) "
            "or schedule a depot repair.",
        ],
    ),
    Domain(
        key="physics",
        role="research physicist",
        pdf_filename="physics_notes.pdf",
        pdf_paragraphs=[
            "Newton's second law states that the net force on an object is "
            "equal to the time derivative of its momentum. For an object "
            "of constant mass this reduces to F = m*a, where F is force "
            "in newtons, m is mass in kilograms, and a is acceleration "
            "in metres per second squared.",
            "Conservation of energy in a closed system means the total "
            "energy — kinetic plus potential plus thermal plus other "
            "forms — does not change over time. Energy may be converted "
            "between forms but is never created or destroyed.",
            "The speed of light in vacuum, denoted c, is exactly "
            "299,792,458 metres per second. It is a fundamental constant "
            "and the upper bound on the speed at which information can "
            "travel through space.",
        ],
    ),
    Domain(
        key="medicine",
        role="general-practice physician",
        pdf_filename="clinical_guidelines.pdf",
        pdf_paragraphs=[
            "Adult resting heart rate is typically 60-100 beats per "
            "minute. Sustained values below 50 (bradycardia) or above "
            "120 at rest (tachycardia) warrant evaluation, particularly "
            "if accompanied by dizziness, chest pain, or syncope.",
            "Acetaminophen (paracetamol) at the standard adult dose of "
            "500-1000 mg every 6 hours is generally well tolerated. The "
            "maximum recommended daily dose is 4 g for healthy adults; "
            "patients with hepatic impairment or chronic alcohol use "
            "should be capped at 2 g per day to reduce hepatotoxicity "
            "risk.",
            "First-line management of mild hypertension (140-159 / 90-99 "
            "mmHg without end-organ damage) is lifestyle modification: "
            "reduced sodium intake, regular aerobic exercise, weight "
            "management, and reducing alcohol consumption. Medication is "
            "added if blood pressure is not controlled after 3 months.",
        ],
    ),
    Domain(
        key="finance",
        role="financial analyst",
        pdf_filename="finance_primer.pdf",
        pdf_paragraphs=[
            "Net present value (NPV) discounts a stream of future cash "
            "flows back to today's value using a chosen discount rate, "
            "then subtracts the initial investment. A positive NPV "
            "indicates the project is expected to create value at the "
            "discount rate used.",
            "The Sharpe ratio measures the excess return of a portfolio "
            "per unit of total volatility (standard deviation). It is "
            "computed as (Rp - Rf) / sigma_p, where Rp is the portfolio "
            "return, Rf is the risk-free rate, and sigma_p is the "
            "portfolio's standard deviation of returns.",
            "Dollar-cost averaging is the practice of investing a fixed "
            "amount on a fixed schedule regardless of price. It reduces "
            "the impact of market timing on long-term returns but does "
            "not, on average, beat lump-sum investing in a rising "
            "market.",
        ],
    ),
    Domain(
        key="law",
        role="corporate lawyer",
        pdf_filename="legal_overview.pdf",
        pdf_paragraphs=[
            "Under the General Data Protection Regulation (GDPR), a data "
            "controller must report a personal-data breach to the "
            "supervisory authority within 72 hours of becoming aware "
            "of it, unless the breach is unlikely to result in a risk "
            "to the rights and freedoms of natural persons.",
            "An NDA (non-disclosure agreement) is generally enforceable "
            "if it (1) is supported by consideration, (2) defines the "
            "confidential information with reasonable specificity, and "
            "(3) sets a duration that is not unreasonably long. Courts "
            "may refuse to enforce overly broad NDAs.",
            "Limited liability for shareholders of a corporation means "
            "that, absent fraud or commingling of personal and corporate "
            "assets ('piercing the corporate veil'), shareholders are "
            "generally not personally liable for the corporation's "
            "debts beyond their investment.",
        ],
    ),
]


DATA_ROOT = Path(__file__).resolve().parent / "data"


def _wrap(text: str, width: int) -> list[str]:
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


def _ensure_pdf(domain: Domain) -> Path:
    folder = DATA_ROOT / domain.key
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / domain.pdf_filename
    if path.exists():
        return path

    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ImportError(
            "reportlab is required to generate sample PDFs. "
            "Install with: pip install reportlab"
        ) from exc

    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    c.setFont("Helvetica-Bold", 14)
    title = domain.pdf_filename.replace("_", " ").rsplit(".", 1)[0].title()
    c.drawString(72, height - 72, title)
    c.setFont("Helvetica", 11)
    y = height - 110
    for para in domain.pdf_paragraphs:
        for line in _wrap(para, 90):
            c.drawString(72, y, line)
            y -= 16
        y -= 8
    c.showPage()
    c.save()
    return path


# --------------------------------------------------------------------
# 1. Per-domain index + retriever tool
# --------------------------------------------------------------------

from jeevesagent import Agent, tool  # noqa: E402
from jeevesagent.loader import RecursiveChunker, load  # noqa: E402
from jeevesagent.memory.embedder import OpenAIEmbedder  # noqa: E402
from jeevesagent.team import Team  # noqa: E402
from jeevesagent.vectorstore import ChromaVectorStore  # noqa: E402


async def build_domain_store(
    domain: Domain, embedder: OpenAIEmbedder
) -> ChromaVectorStore:
    pdf = _ensure_pdf(domain)
    persist = DATA_ROOT / f".chroma_{domain.key}"
    store = ChromaVectorStore(
        embedder=embedder,
        collection_name=f"{domain.key}_docs",
        persist_directory=str(persist),
    )

    if store._collection.count():  # type: ignore[attr-defined]
        return store

    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=60)
    doc = load(pdf)
    chunks = chunker.split(doc.content, source=str(pdf))
    for ch in chunks:
        ch.metadata["source_file"] = pdf.name
        ch.metadata["domain"] = domain.key
    await store.add(chunks)
    print(f"  [{domain.key:9s}] indexed {len(chunks)} chunks from {pdf.name}")
    return store


def make_retriever_tool(domain: Domain, store: ChromaVectorStore):
    """Return a @tool that searches *only* this domain's collection."""

    @tool(name=f"search_{domain.key}_docs")
    async def search(query: str) -> str:
        """Search this specialist's knowledge base.

        Returns up to 3 passages from the indexed documents most
        relevant to ``query``, each prefixed with its source
        filename. Always call this before answering.
        """
        results = await store.search(query, k=3)
        if not results:
            return "(no matching passages)"
        return "\n\n".join(
            f"[{i}] ({r.chunk.metadata.get('source_file', '?')})\n"
            f"{r.chunk.content.strip()}"
            for i, r in enumerate(results, 1)
        )

    return search


def build_specialist(domain: Domain, retriever) -> Agent:
    return Agent(
        instructions=(
            f"You are a {domain.role}. Stay strictly within your "
            f"domain — if the question falls outside {domain.role} "
            "expertise, say so and decline to speculate. Always "
            f"call search_{domain.key}_docs to ground any factual "
            "claim you make. Be concise; one paragraph at most."
        ),
        model="gpt-4.1-mini",
        tools=[retriever],
    )


# --------------------------------------------------------------------
# 2. Compose them into a debate team
# --------------------------------------------------------------------


async def main() -> None:
    print("\n  Example 2 — Five-specialist debate with per-agent RAG\n")

    embedder = OpenAIEmbedder("text-embedding-3-small")

    specialists: list[Agent] = []
    for d in DOMAINS:
        store = await build_domain_store(d, embedder)
        retriever = make_retriever_tool(d, store)
        specialists.append(build_specialist(d, retriever))

    judge = Agent(
        instructions=(
            "You are an impartial judge. You are given the responses "
            "of five domain specialists (IT, physics, medicine, "
            "finance, law). Your job is to synthesise a single "
            "answer by combining only the parts each specialist is "
            "qualified to address; ignore any specialist who declined "
            "or strayed outside their domain. Quote source filenames "
            "where the specialists provided them. Be concise."
        ),
        model="gpt-4.1-mini",
    )

    team = Team.debate(
        debaters=specialists,
        judge=judge,
        model="gpt-4.1-mini",
        rounds=1,
        convergence_check=False,
    )

    questions = [
        # A real cross-domain question — touches medicine + finance + law.
        "An employee's BitLocker-encrypted laptop won't turn on, and "
        "they say the laptop contains some unencrypted personal-health "
        "notes about a customer. The customer is in the EU. What should "
        "be done first, and what reporting obligations might apply?",
        # A more focused question — only one domain truly applies.
        "A user reports their corporate Windows laptop powers on but "
        "the display stays black. What is the recommended diagnostic "
        "sequence?",
    ]

    for q in questions:
        print("─" * 72)
        print(f"Q: {q}\n")
        result = await team.run(q)
        print(f"A: {result.output}\n")
        print(f"   ({result.turns} turns, {result.tokens_in}+{result.tokens_out} tokens)")
    print("─" * 72)


if __name__ == "__main__":
    asyncio.run(main())
