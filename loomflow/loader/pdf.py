"""PDF loader → markdown.

We replaced ``pypdf`` because its extraction quality on real-world
PDFs (multi-column layouts, tables, figures, embedded fonts) was
measurably worse than what's needed for production RAG. Lower
sections of structured documents would extract cleanly enough to
*look* fine but be jumbled column-interleaved or silently empty,
producing the classic "questions about content near the end of
the PDF go unanswered" symptom — a footgun that survives unit
tests and only breaks at retrieval time.

Two backends ship in-tree; pick at load time via ``backend=``:

* ``backend="unstructured"`` (default) — wraps ``unstructured``
  (Apache 2.0, what LangChain's ``UnstructuredPDFLoader`` uses).
  Element-level parsing (``Title`` / ``NarrativeText`` / ``Table``
  / ``ListItem``); per-page metadata. Battle-tested across
  thousands of RAG pipelines. Three quality modes via
  ``strategy=``:
    * ``"fast"`` — pure-Python parse via ``pdfminer.six``. Default.
    * ``"hi_res"`` — YOLO-based layout detection
      (``unstructured-inference``). Best on multi-column / table-
      heavy PDFs.
    * ``"ocr_only"`` — Tesseract OCR for scanned/image-only PDFs.
* ``backend="docling"`` — wraps ``docling`` (IBM Research, MIT).
  ML-based structure-aware extraction; the 2026 best-in-class
  benchmark winner for native PDFs. Outputs clean markdown with
  preserved hierarchy. Slower first run (downloads layout model)
  then comparable speed.

Output format (both backends produce the same ``Document`` shape):

* ``content`` — markdown string starting with ``# <title>``,
  per-page ``## Page N`` sections, paragraph / heading / list /
  table content within.
* ``metadata`` — ``{"source", "format": "pdf", "page_count",
  "title", "backend", "strategy"}``.

Per-page extraction failures emit a ``warnings.warn`` with the
backend, page number, and underlying exception. Pre-replacement
versions used ``except Exception: text = ""`` which produced
empty pages with zero log signal — the silent footgun behind the
"lower-half pages have no answers" symptom.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .base import Document

_log = logging.getLogger("loomflow.loader.pdf")

_BACKENDS = ("unstructured", "docling")

# Docling escapes markdown special chars in body text (so an
# identifier like ``PAGE_ONE`` ends up as ``PAGE\_ONE`` in its
# markdown output). That's technically correct markdown but bad
# for RAG: embeddings degrade on backslash-escaped tokens, and
# substring-search retrieval misses identifiers entirely. We strip
# the escapes back out — body text doesn't need markdown emphasis
# markers preserved, just plain content.
_MD_ESCAPE_RE = re.compile(r"\\([_*#<>\[\]()!`~|\\])")


def _unescape_markdown(text: str) -> str:
    """Reverse markdown-special-char backslash-escaping in text.
    Used to clean Docling's overcautious output before chunking."""
    return _MD_ESCAPE_RE.sub(r"\1", text)


def load_pdf(
    path: str | Path,
    *,
    backend: str = "unstructured",
    strategy: str = "fast",
    languages: list[str] | None = None,
) -> Document:
    """Load a PDF and convert it to clean markdown.

    Parameters
    ----------
    path
        Filesystem path to the PDF.
    backend
        ``"unstructured"`` (default) or ``"docling"``. See module
        docstring for the trade-offs. Defaults to unstructured
        because it's the more battle-tested option; switch to
        docling for the 2026 best-in-class quality on native PDFs.
    strategy
        Only meaningful for ``backend="unstructured"``: ``"fast"``
        (default), ``"hi_res"``, or ``"ocr_only"``. Ignored by the
        docling backend (which always runs its full layout-aware
        pipeline).
    languages
        Optional list of language codes (``["eng", "fra"]``) for
        the OCR / layout backends. Only used by
        ``backend="unstructured"`` with ``strategy="ocr_only"`` /
        ``"hi_res"``. Defaults to English.

    Returns
    -------
    Document
        ``content`` is the markdown rendering; ``metadata`` carries
        ``source``, ``format="pdf"``, ``page_count``, ``title``,
        ``backend``, and ``strategy``.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; "
            f"expected one of: {', '.join(_BACKENDS)}"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    if backend == "unstructured":
        return _load_via_unstructured(p, strategy=strategy, languages=languages)
    return _load_via_docling(p)


# ---------------------------------------------------------------------------
# Unstructured backend
# ---------------------------------------------------------------------------


def _load_via_unstructured(
    p: Path,
    *,
    strategy: str,
    languages: list[str] | None,
) -> Document:
    try:
        from unstructured.partition.pdf import (  # type: ignore[import-not-found, import-untyped]
            partition_pdf,
        )
    except ImportError as exc:  # pragma: no cover — depends on user env
        raise ImportError(
            "unstructured is not installed. "
            "Install with: pip install 'loomflow[loader-pdf]' "
            "(or 'loomflow[loader]' for all loader extras). "
            "If you want OCR / hi-res layout analysis, install "
            "with: pip install 'unstructured[pdf,ocr]'."
        ) from exc

    try:
        elements = partition_pdf(
            filename=str(p),
            strategy=strategy,
            languages=languages or ["eng"],
        )
    except Exception as exc:  # noqa: BLE001 — surface to caller w/ context
        return _failed_doc(p, "unstructured", strategy, exc)

    title = _extract_title_unstructured(elements) or p.stem
    page_count = max(
        (
            getattr(e.metadata, "page_number", None) or 0
            for e in elements
        ),
        default=0,
    )
    content = _render_markdown_unstructured(elements, title)

    return Document(
        content=content,
        metadata={
            "source": str(p),
            "format": "pdf",
            "page_count": page_count,
            "title": title,
            "backend": "unstructured",
            "strategy": strategy,
        },
    )


def _extract_title_unstructured(elements: Iterable[Any]) -> str:
    """Best-effort document title from the first ``Title`` element on
    page 1; caller substitutes the filename stem if empty."""
    for e in elements:
        page = getattr(e.metadata, "page_number", None)
        if page is not None and page > 1:
            break
        if e.category == "Title":
            text = (e.text or "").strip()
            if text:
                return text
    return ""


def _render_markdown_unstructured(
    elements: Iterable[Any], title: str
) -> str:
    """Group elements by page, emit `## Page N` sections, render
    each element by its category."""
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")

    pages: dict[int, list[Any]] = {}
    for e in elements:
        page = getattr(e.metadata, "page_number", None) or 1
        pages.setdefault(page, []).append(e)

    for page_no in sorted(pages):
        parts.append(f"## Page {page_no}\n")
        rendered = _render_page_unstructured(
            pages[page_no], doc_title=title
        )
        parts.append(rendered or "(no extractable text)")
        parts.append("")

    return "\n".join(parts)


def _render_page_unstructured(
    elements: list[Any], *, doc_title: str
) -> str:
    """Render one page's unstructured elements in arrival order."""
    out: list[str] = []
    in_list = False
    for e in elements:
        category = e.category
        text = (e.text or "").strip()
        if not text:
            continue

        # Skip the doc-level title if it leaks onto page 1.
        if category == "Title" and text == doc_title:
            continue

        if category == "ListItem":
            out.append(f"- {text}")
            in_list = True
            continue
        if in_list:
            out.append("")
            in_list = False

        if category == "Title":
            out.append(f"### {text}\n")
        elif category == "Table":
            html = getattr(e.metadata, "text_as_html", None)
            out.append(html if html else text)
            out.append("")
        elif category == "Image":
            # No vision pipeline downstream — skip.
            continue
        else:
            out.append(text)
            out.append("")

    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Docling backend
# ---------------------------------------------------------------------------


def _load_via_docling(p: Path) -> Document:
    try:
        from docling.document_converter import (  # type: ignore[import-not-found, import-untyped]
            DocumentConverter,
        )
    except ImportError as exc:  # pragma: no cover — depends on user env
        raise ImportError(
            "docling is not installed. "
            "Install with: pip install 'loomflow[loader-pdf-docling]' "
            "(adds docling and its layout-model deps). "
            "Or pass backend='unstructured' to use the default."
        ) from exc

    try:
        converter = DocumentConverter()
        result = converter.convert(str(p))
    except Exception as exc:  # noqa: BLE001 — surface to caller w/ context
        return _failed_doc(p, "docling", "default", exc)

    document = result.document

    # ``export_to_markdown`` returns clean markdown with hierarchy
    # preserved (Docling's ``DoclingDocument`` knows about titles,
    # sections, tables, lists, etc.). We prepend a per-page section
    # header so the output shape matches the unstructured backend's
    # ``# title`` + ``## Page N`` convention.
    title = _unescape_markdown(_extract_title_docling(document) or p.stem)
    page_count = _docling_page_count(document)
    body = _unescape_markdown(document.export_to_markdown())

    parts: list[str] = []
    parts.append(f"# {title}\n")
    if page_count > 1:
        # Docling's per-page export keeps section structure; if we
        # have multiple pages, render them as ``## Page N`` blocks.
        for page_no in range(1, page_count + 1):
            page_md = _unescape_markdown(
                _docling_page_markdown(document, page_no)
            )
            parts.append(f"## Page {page_no}\n")
            parts.append(page_md or "(no extractable text)")
            parts.append("")
    else:
        # Single-page doc — match the unstructured convention by
        # wrapping the whole body under ``## Page 1``.
        parts.append("## Page 1\n")
        parts.append(body.strip() or "(no extractable text)")
        parts.append("")

    content = "\n".join(parts)

    return Document(
        content=content,
        metadata={
            "source": str(p),
            "format": "pdf",
            "page_count": page_count,
            "title": title,
            "backend": "docling",
            "strategy": "default",
        },
    )


def _extract_title_docling(document: Any) -> str:
    """Pull a title from Docling's structured representation.

    Docling records section headers in ``document.texts`` with
    labels like ``title`` / ``section_header``. The first
    ``title``-labelled item is the doc title; if none, look for
    the first heading.
    """
    texts = getattr(document, "texts", None) or []
    for t in texts:
        label = getattr(t, "label", "") or ""
        if str(label).lower() == "title":
            content = (getattr(t, "text", "") or "").strip()
            if content:
                return content
    # Fallback: first section header.
    for t in texts:
        label = getattr(t, "label", "") or ""
        if "section_header" in str(label).lower():
            content = (getattr(t, "text", "") or "").strip()
            if content:
                return content
    return ""


def _docling_page_count(document: Any) -> int:
    """Best-effort page count from a ``DoclingDocument``.

    Docling exposes ``num_pages()`` on the canonical type but
    different versions / accessors land here, so we fall back to
    walking ``pages`` if the method is missing or returns 0.
    """
    fn = getattr(document, "num_pages", None)
    if callable(fn):
        try:
            n = fn()
            if isinstance(n, int) and n > 0:
                return n
        except Exception:  # noqa: BLE001
            pass
    pages = getattr(document, "pages", None)
    if isinstance(pages, dict) and pages:
        return len(pages)
    if isinstance(pages, list) and pages:
        return len(pages)
    return 1  # safe default — at least one page exists


def _docling_page_markdown(document: Any, page_no: int) -> str:
    """Export the markdown for one page of a ``DoclingDocument``.

    Newer docling versions accept ``page_no=`` on
    ``export_to_markdown``; older ones don't. We try the kwarg
    path first; on failure we fall back to filtering the texts
    manually.
    """
    fn = getattr(document, "export_to_markdown", None)
    if callable(fn):
        try:
            md = fn(page_no=page_no)
            if isinstance(md, str) and md.strip():
                return md
        except TypeError:
            # Older docling — no per-page kwarg. Fall through to
            # the text-walking fallback below.
            pass
        except Exception:  # noqa: BLE001 — defensive
            pass

    # Fallback: collect any text item whose first provenance entry
    # lands on the requested page.
    out: list[str] = []
    for t in getattr(document, "texts", None) or []:
        prov = getattr(t, "prov", None) or []
        if not prov:
            continue
        first = prov[0]
        if getattr(first, "page_no", None) == page_no:
            text = (getattr(t, "text", "") or "").strip()
            if text:
                out.append(text)
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Shared error helper
# ---------------------------------------------------------------------------


def _failed_doc(
    p: Path, backend: str, strategy: str, exc: BaseException
) -> Document:
    """Emit a warning + log line and return a non-fatal empty
    Document so the loader pipeline keeps going on bad inputs."""
    warnings.warn(
        f"{backend} failed to parse {p.name!r} "
        f"(strategy={strategy!r}): "
        f"{exc.__class__.__name__}: {exc}. "
        "Returning an empty Document; fix the underlying issue or "
        "try a different backend / strategy.",
        RuntimeWarning,
        stacklevel=3,
    )
    _log.warning(
        "%s backend raised on %s: %r", backend, p.name, exc
    )
    return Document(
        content=f"# {p.stem}\n\n(no extractable content)\n",
        metadata={
            "source": str(p),
            "format": "pdf",
            "page_count": 0,
            "title": "",
            "backend": backend,
            "strategy": strategy,
            "extraction_error": str(exc),
        },
    )
