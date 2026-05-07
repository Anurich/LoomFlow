"""25_document_pipeline — Batch-extract structured records from documents.

Real workflow: a folder full of invoice PDFs (or receipts, contracts,
forms, etc.). For each one, you want the same set of fields out —
amount, date, vendor, line items. The classic ETL-with-LLMs job.

What this example shows
-----------------------

* **Loader** — :func:`jeevesagent.loader.load` reads each file
  regardless of format (.md / .pdf / .docx / .xlsx / .csv / .html
  with the matching extras installed). The demo drops markdown
  receipts into the workdir so it runs zero-key on the loader
  side.
* **Structured output via Pydantic** — each record conforms to
  :class:`InvoiceRecord`. The extractor agent emits JSON; we parse
  + validate. Bad output retries via Reflexion (it learns from
  the failed attempt's evaluator score).
* **Supervisor with parallel delegations** — the manager fans out
  one ``extractor`` worker per document. Anyio's structured
  concurrency runs them all concurrently — the wall-clock cost is
  ~1× a single extraction, not N×.
* **CSV output** at the end so the result is machine-readable.

Run::

    pip install -e '.[dev,openai,loader]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/25_document_pipeline.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "\n  ✗ OPENAI_API_KEY required. "
        "Add OPENAI_API_KEY=sk-... to .env at repo root.\n"
    )

from jeevesagent import Agent, Team, Tool, tool  # noqa: E402
from jeevesagent.loader import load  # noqa: E402

# ---------------------------------------------------------------------------
# Sample receipts. In production these would be actual PDFs/scans.
# ---------------------------------------------------------------------------

SAMPLE_RECEIPTS: dict[str, str] = {
    "receipt_001.md": (
        "**TechCorp Inc.**\n"
        "123 Innovation Way\n\n"
        "Invoice #TC-9842\n"
        "Date: 2026-04-12\n\n"
        "| Item                  | Qty | Price |\n"
        "|-----------------------|-----|-------|\n"
        "| API credits (10K)     |  1  | 250.00 |\n"
        "| Premium support       |  1  | 99.00  |\n\n"
        "Subtotal: $349.00\n"
        "Tax: $28.42\n"
        "**Total: $377.42**"
    ),
    "receipt_002.md": (
        "**CloudVendor Ltd**\n"
        "Quarterly bill\n\n"
        "Invoice: CV-2026-Q2-447\n"
        "Date: 2026-05-01\n\n"
        "Compute hours .... 1,247 @ $0.08 = $99.76\n"
        "Storage TB-mo .... 4.2 @ $25.00 = $105.00\n"
        "Egress GB ........ 880 @ $0.09 = $79.20\n\n"
        "**Grand Total: $283.96**"
    ),
    "receipt_003.md": (
        "**Acme Office Supplies**\n\n"
        "Sales receipt — invoice 88-2271\n"
        "Date 04/28/2026\n\n"
        "10x USB-C cables @ $9.99 ............ $99.90\n"
        "2x ergonomic keyboards @ $149.50 ..... $299.00\n"
        "1x box of pens (50ct) ................ $24.99\n\n"
        "Tax (7.5%): $31.79\n"
        "Total due: $455.68"
    ),
    "receipt_004.md": (
        "ZephyrAirlines\n"
        "E-receipt for booking #ZA-LX-99812\n"
        "Issued 2026-03-15\n\n"
        "Round-trip JFK ↔ LHR\n"
        "Passenger: business class\n\n"
        "Base fare: $1,890.00\n"
        "Fees & taxes: $310.50\n"
        "Total charged: $2,200.50"
    ),
    "receipt_005.md": (
        "**Joe's Coffee**\n"
        "Order #JC-440217 — 2026-04-30\n\n"
        "2x latte @ $5.50 = $11.00\n"
        "1x croissant = $4.25\n"
        "1x americano = $4.00\n\n"
        "Subtotal $19.25\n"
        "Tip $4.00\n"
        "Total $23.25"
    ),
}


# ---------------------------------------------------------------------------
# Target schema for extracted records.
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    line_total: float | None = None


class InvoiceRecord(BaseModel):
    """One extracted invoice/receipt — the format we want out the
    other end of the pipeline."""

    invoice_id: str
    vendor: str
    date: str
    line_items: list[LineItem]
    subtotal: float | None = None
    tax: float | None = None
    total: float


# ---------------------------------------------------------------------------
# Tools — read documents + record extracted JSON.
# ---------------------------------------------------------------------------


# In-memory results store. The supervisor's workers each write
# one record here as they finish. We aggregate to CSV at the end.
_RESULTS: dict[str, dict[str, object]] = {}


def _make_load_doc_tool(workdir: Path) -> Tool:
    @tool
    async def load_document(filename: str) -> str:
        """Load a document from the input folder by filename. Returns
        the markdown-converted text — supports .md/.pdf/.docx/.xlsx/
        .csv/.html with the matching loader extras installed."""
        full = (workdir / filename).resolve()
        # Sync filesystem stat is fine — single one-shot path
        # validation; the loader's read dominates wall time.
        if not str(full).startswith(str(workdir.resolve())):  # noqa: ASYNC240
            return f"ERROR: {filename} escapes workdir"
        if not full.exists():  # noqa: ASYNC240
            return f"ERROR: {filename} not found"
        document = load(full)
        return document.content

    return load_document


def _make_record_extracted_tool() -> Tool:
    @tool
    async def record_extracted(filename: str, payload_json: str) -> str:
        """Submit an extracted ``InvoiceRecord`` for the given source
        filename. ``payload_json`` must be valid JSON parseable into
        the schema. Pydantic validation errors are returned to the
        agent so it can self-correct."""
        try:
            data = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return f"ERROR: invalid JSON: {exc}"
        try:
            record = InvoiceRecord.model_validate(data)
        except ValidationError as exc:
            return (
                f"ERROR: schema validation failed:\n"
                f"{exc}\n\n"
                "Re-extract with the correct fields. Required: "
                "invoice_id, vendor, date, line_items, total. "
                "Optional: subtotal, tax."
            )
        _RESULTS[filename] = record.model_dump()
        return (
            f"OK: recorded {filename} → "
            f"vendor={record.vendor!r}, total=${record.total}"
        )

    return record_extracted


# ---------------------------------------------------------------------------
# Build agents.
# ---------------------------------------------------------------------------


def _build_pipeline(workdir: Path) -> Agent:
    load_doc = _make_load_doc_tool(workdir)
    record = _make_record_extracted_tool()

    extractor = Agent(
        instructions=(
            "You extract structured invoice data from documents. "
            "For each filename the supervisor names: "
            "1) call `load_document(filename)` to read the file, "
            "2) parse the content for invoice_id, vendor, date, "
            "line items (description + quantity + unit_price + "
            "line_total), subtotal, tax, total, "
            "3) call `record_extracted(filename, payload_json)` "
            "with the JSON exactly matching this schema:\n\n"
            '{"invoice_id": str, "vendor": str, "date": str (ISO 8601 if possible), '
            '"line_items": [{"description": str, "quantity": number, "unit_price": number, "line_total": number}], '
            '"subtotal": number|null, "tax": number|null, "total": number}\n\n'
            "If validation fails, retry with corrected JSON. "
            "Return a one-line confirmation when done."
        ),
        model="gpt-4.1-mini",
        tools=[load_doc, record],
    )

    # Team.supervisor is the ergonomic facade — equivalent to
    # ``Agent(instructions=..., model=..., architecture=Supervisor(
    # workers={"extractor": extractor}))`` but reads like the
    # supervisor builders in LangGraph / CrewAI / AutoGen.
    return Team.supervisor(
        workers={"extractor": extractor},
        instructions=(
            "You manage a document-processing pipeline. The user "
            "names a list of input filenames. Delegate ONE call to "
            "`extractor` per filename — issue ALL the delegations "
            "in a single turn so they run in parallel. After all "
            "extractors return, summarize: how many succeeded, "
            "any that failed, total invoiced amount across all."
        ),
        model="gpt-4.1-mini",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = Path(  # noqa: ASYNC240 — demo startup
        tempfile.mkdtemp(prefix="jeeves_docs_")
    ).resolve()

    print("=" * 70)
    print("Document pipeline — batch invoice extraction")
    print("=" * 70)
    print(f"Workdir: {workdir}\n")

    # Materialise the input documents.
    for name, body in SAMPLE_RECEIPTS.items():
        (workdir / name).write_text(body)
    print(f"Pre-seeded {len(SAMPLE_RECEIPTS)} receipt(s) in {workdir}\n")

    pipeline = _build_pipeline(workdir)
    files_listing = ", ".join(SAMPLE_RECEIPTS.keys())
    prompt = (
        f"Extract structured records from these {len(SAMPLE_RECEIPTS)} "
        f"files: {files_listing}. Delegate one extractor per file in "
        "parallel."
    )

    print(f"Goal: {prompt}\n")
    print("─" * 70)

    delegation_count = 0
    async for ev in pipeline.stream(prompt):
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
                preview = args.get("instructions", "")[:80]
                print(
                    f"\n  ┌─ delegation #{delegation_count}: "
                    f"{preview}..."
                )
            elif tool_name == "record_extracted":
                fname = args.get("filename", "?")
                print(f"\n  · recording extraction for {fname}")
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n" + "─" * 70)
            print("\nMANAGER'S SUMMARY:")
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}  "
                f"Cost: ${float(result.get('cost_usd', 0) or 0):.4f}"
            )

    # Aggregate the extracted records → CSV.
    print(f"\n{'═' * 70}")
    print("EXTRACTED RECORDS")
    print(f"{'═' * 70}")

    if not _RESULTS:
        print("(No records extracted — see the trace above for issues.)")
        return

    csv_path = workdir / "invoices.csv"
    with csv_path.open("w", newline="") as f:  # noqa: ASYNC230 — final sync write
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "invoice_id",
                "vendor",
                "date",
                "line_count",
                "subtotal",
                "tax",
                "total",
            ],
        )
        writer.writeheader()
        for filename, rec in sorted(_RESULTS.items()):
            line_items = rec.get("line_items") or []
            assert isinstance(line_items, list)
            writer.writerow(
                {
                    "filename": filename,
                    "invoice_id": rec.get("invoice_id"),
                    "vendor": rec.get("vendor"),
                    "date": rec.get("date"),
                    "line_count": len(line_items),
                    "subtotal": rec.get("subtotal"),
                    "tax": rec.get("tax"),
                    "total": rec.get("total"),
                }
            )

    # Pretty-print the table.
    print(
        f"{'filename':<20} {'vendor':<22} {'date':<12} "
        f"{'lines':>6} {'total':>10}"
    )
    print("-" * 72)
    grand_total = 0.0
    for filename, rec in sorted(_RESULTS.items()):
        total_raw = rec.get("total") or 0.0
        total = float(total_raw)  # type: ignore[arg-type]
        grand_total += total
        line_items = rec.get("line_items") or []
        assert isinstance(line_items, list)
        print(
            f"{filename:<20} "
            f"{str(rec.get('vendor', '?'))[:22]:<22} "
            f"{str(rec.get('date', '?'))[:12]:<12} "
            f"{len(line_items):>6d} "
            f"{total:>10.2f}"
        )
    print("-" * 72)
    print(f"{'GRAND TOTAL':<60} {grand_total:>10.2f}")
    print(f"\nCSV written to {csv_path}")
    print(f"(Workdir kept at {workdir} for inspection.)")


if __name__ == "__main__":
    asyncio.run(main())
