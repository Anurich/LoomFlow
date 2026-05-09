"""Core types for the loader: :class:`Document` and :class:`Chunk`.

Every loader normalizes its source format to a :class:`Document`
whose ``content`` is markdown text and whose ``metadata`` carries
provenance (source path, MIME type, page / sheet count, etc.).
The chunkers in :mod:`loomflow.loader.chunking` consume the
``content`` and produce :class:`Chunk` objects with their own
metadata pointing back at the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A loaded document, normalized to markdown.

    ``content``
        The full markdown text. Loaders produce reasonable
        markdown: PDF / DOCX preserve headings + paragraphs; Excel
        / CSV become markdown tables; HTML preserves heading +
        paragraph + list structure.

    ``metadata``
        Free-form dict with at least:

        * ``source`` — the source file path (str)
        * ``format`` — the source format (``"pdf"``, ``"docx"``,
            ``"xlsx"``, ``"csv"``, ``"tsv"``, ``"md"``, ``"txt"``,
            ``"html"``)

        Format-specific keys may be present (``"page_count"`` for
        PDFs, ``"sheet_names"`` for Excel, ``"row_count"`` for CSV,
        etc.).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


@dataclass
class Chunk:
    """One piece of a chunked document.

    ``content`` is a substring of the source document's content
    (with possible cleanup — trimmed whitespace, etc.).
    ``metadata`` carries:

    * ``source`` — pass-through from the parent :class:`Document`
    * ``index`` — zero-based chunk index in the source
    * ``chunk_size`` — actual length of ``content`` (chars)
    * Strategy-specific keys (e.g. ``headers`` from
      :class:`MarkdownChunker`, ``token_count`` from
      :class:`TokenChunker`).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)
