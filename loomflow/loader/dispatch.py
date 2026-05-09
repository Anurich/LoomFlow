"""Auto-detect the right loader from a file's extension.

::

    from loomflow.loader import load
    doc = load("report.pdf")        # → load_pdf
    doc = load("data.xlsx")         # → load_excel
    doc = load("research.md")       # → load_markdown
"""

from __future__ import annotations

from pathlib import Path

from .base import Document
from .csv import load_csv, load_tsv
from .docx import load_docx
from .excel import load_excel
from .html import load_html
from .pdf import load_pdf
from .text import load_markdown, load_text

_DISPATCH = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xlsx": load_excel,
    ".xlsm": load_excel,
    ".csv": load_csv,
    ".tsv": load_tsv,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".html": load_html,
    ".htm": load_html,
}


def load(path: str | Path) -> Document:
    """Load a document by auto-detecting its format from the file
    extension. Supported: ``.pdf``, ``.docx``, ``.xlsx``, ``.xlsm``,
    ``.csv``, ``.tsv``, ``.md``, ``.markdown``, ``.txt``, ``.html``,
    ``.htm``.

    Raises :class:`ValueError` for unknown extensions.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    loader = _DISPATCH.get(suffix)
    if loader is None:
        supported = ", ".join(sorted(_DISPATCH))
        raise ValueError(
            f"unsupported extension {suffix!r}; supported: {supported}"
        )
    return loader(p)
