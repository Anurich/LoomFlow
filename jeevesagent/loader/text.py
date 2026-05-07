"""Plain-text and markdown loaders.

These are the simplest cases — markdown / text files are already
UTF-8 text; we just read and tag.
"""

from __future__ import annotations

from pathlib import Path

from .base import Document


def load_markdown(path: str | Path) -> Document:
    """Load a markdown file. Just reads UTF-8 text."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return Document(
        content=text,
        metadata={
            "source": str(p),
            "format": "md",
            "byte_count": len(text.encode("utf-8")),
            "line_count": text.count("\n") + (1 if text else 0),
        },
    )


def load_text(path: str | Path) -> Document:
    """Load a plain-text file. Wraps content in markdown by
    adding a ``# {filename}`` heading so downstream chunkers /
    consumers see consistent markdown."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return Document(
        content=f"# {p.name}\n\n{text}",
        metadata={
            "source": str(p),
            "format": "txt",
            "byte_count": len(text.encode("utf-8")),
            "line_count": text.count("\n") + (1 if text else 0),
        },
    )
