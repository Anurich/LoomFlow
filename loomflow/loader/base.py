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
from pathlib import Path
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


#: Default text encoding for the text-based loaders: UTF-8, with a
#: leading byte-order mark (as Excel and Notepad write) stripped.
DEFAULT_ENCODING = "utf-8-sig"


def read_text_file(
    path: Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
    universal_newlines: bool = True,
) -> str:
    """Read a text file for a loader.

    The default ``utf-8-sig`` reads plain UTF-8 and drops a BOM if
    there is one. A file in another encoding raises
    :class:`UnicodeDecodeError` whose message names the way out:
    pass ``encoding=`` (``"latin-1"``, ``"cp1252"``, ``"utf-16"`` …)
    or ``errors="replace"`` to the loader.

    ``universal_newlines=False`` keeps CRLF / CR line endings as
    they are (the csv module does its own line handling).
    """
    try:
        text = path.read_bytes().decode(encoding, errors)
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"{exc.reason} — {path} is not valid {encoding!r} text. Pass "
            "encoding= with the file's real encoding (e.g. "
            "encoding='latin-1' or 'cp1252'), or errors='replace' to "
            "substitute the undecodable bytes",
        ) from exc
    # ``utf-8-sig`` strips the BOM itself; cover an explicit "utf-8".
    text = text.removeprefix("\ufeff")
    if universal_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text
