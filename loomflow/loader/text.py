"""Plain-text and markdown loaders.

These are the simplest cases — markdown / text files are already
text; we just decode and tag. Files are read as UTF-8 by default,
with a leading byte-order mark dropped; pass ``encoding=`` /
``errors=`` for anything else.
"""

from __future__ import annotations

from pathlib import Path

from .base import DEFAULT_ENCODING, Document, read_text_file


def load_markdown(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
) -> Document:
    """Load a markdown file.

    ``encoding`` defaults to ``"utf-8-sig"`` (UTF-8, BOM stripped);
    ``errors`` is the codec error handler (``"strict"`` raises on
    undecodable bytes, ``"replace"`` substitutes them).
    """
    p = Path(path)
    text = read_text_file(p, encoding=encoding, errors=errors)
    return Document(
        content=text,
        metadata={
            "source": str(p),
            "format": "md",
            "byte_count": len(text.encode("utf-8")),
            "line_count": text.count("\n") + (1 if text else 0),
        },
    )


def load_text(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
) -> Document:
    """Load a plain-text file. Wraps content in markdown by
    adding a ``# {filename}`` heading so downstream chunkers /
    consumers see consistent markdown.

    ``encoding`` / ``errors`` as for :func:`load_markdown`.
    """
    p = Path(path)
    text = read_text_file(p, encoding=encoding, errors=errors)
    return Document(
        content=f"# {p.name}\n\n{text}",
        metadata={
            "source": str(p),
            "format": "txt",
            "byte_count": len(text.encode("utf-8")),
            "line_count": text.count("\n") + (1 if text else 0),
        },
    )
