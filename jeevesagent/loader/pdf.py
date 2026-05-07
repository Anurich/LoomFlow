"""PDF loader → markdown.

Uses ``pypdf`` (lazy import). Each page becomes a section ``#
Page N`` in the markdown output. Page-level whitespace is
normalized; otherwise the text comes through as the PDF's
extractable layer reports it.

PDFs vary wildly in extractability — scanned image PDFs return
empty text; layout-heavy PDFs lose column structure. For
production use cases needing OCR / table extraction, swap this
loader for ``pdfplumber`` or ``unstructured`` (kept out of the
default dependency footprint).
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import Document

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_page_text(text: str) -> str:
    """Strip absurd spacing PDFs sometimes have."""
    # Collapse any run of 3+ whitespace into a paragraph break.
    out = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    # Strip trailing whitespace from each line.
    out = "\n".join(line.rstrip() for line in out.splitlines())
    return out.strip()


def load_pdf(path: str | Path) -> Document:
    """Load a PDF, convert to markdown.

    Each page becomes ``## Page N`` followed by the extracted text.
    Requires ``pypdf``: ``pip install 'jeevesagent[loader-pdf]'``.
    """
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pypdf is not installed. "
            "Install with: pip install 'jeevesagent[loader-pdf]' "
            "(or 'jeevesagent[loader]' for all loader extras)."
        ) from exc

    p = Path(path)
    reader = PdfReader(str(p))

    # Document-level metadata from the PDF (title, author, etc.).
    title = ""
    try:
        raw_title = reader.metadata.get("/Title", "") if reader.metadata else ""
        title = (raw_title or "").strip() if isinstance(raw_title, str) else ""
    except (TypeError, AttributeError):
        title = ""

    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")
    elif p.stem:
        parts.append(f"# {p.stem}\n")

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — PDFs are messy
            text = ""
        text = _normalize_page_text(text)
        parts.append(f"## Page {i}\n")
        parts.append(text or "(no extractable text)")
        parts.append("")  # blank line between pages

    content = "\n".join(parts)
    return Document(
        content=content,
        metadata={
            "source": str(p),
            "format": "pdf",
            "page_count": len(reader.pages),
            "title": title,
        },
    )
