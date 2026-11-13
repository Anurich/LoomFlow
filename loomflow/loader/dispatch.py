"""Auto-detect the right loader from a file's extension.

::

    from loomflow.loader import load
    doc = load("report.pdf")        # → load_pdf
    doc = load("data.xlsx")         # → load_excel
    doc = load("research.md")       # → load_markdown
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


# Loaders that decode text themselves and so take ``encoding=`` /
# ``errors=``.
_TEXT_LOADERS = frozenset(
    {load_csv, load_tsv, load_markdown, load_text, load_html}
)


def load(
    path: str | Path,
    *,
    encoding: str | None = None,
    errors: str | None = None,
) -> Document:
    """Load a document by auto-detecting its format from the file
    extension. Supported: ``.pdf``, ``.docx``, ``.xlsx``, ``.xlsm``,
    ``.csv``, ``.tsv``, ``.md``, ``.markdown``, ``.txt``, ``.html``,
    ``.htm``.

    ``encoding`` / ``errors`` apply to the text formats (``.txt``,
    ``.md``, ``.markdown``, ``.csv``, ``.tsv``, ``.html``, ``.htm``),
    which default to UTF-8 with a byte-order mark stripped; pass e.g.
    ``encoding="latin-1"`` for a file in another encoding. They are
    ignored for binary formats.

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
    if loader in _TEXT_LOADERS:
        options: dict[str, Any] = {}
        if encoding is not None:
            options["encoding"] = encoding
        if errors is not None:
            options["errors"] = errors
        return loader(p, **options)
    return loader(p)
