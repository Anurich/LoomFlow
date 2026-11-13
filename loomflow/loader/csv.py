"""CSV / TSV loaders → markdown table.

Uses the stdlib ``csv`` module. The first row is treated as the
header. Each row becomes a markdown table row. For very large CSVs
(hundreds of thousands of rows), prefer pandas + manual chunking;
this loader is intended for documents an LLM will read end-to-end.

Files are read as UTF-8 by default with a leading byte-order mark
dropped (Excel writes one, and it used to end up inside the first
header cell); pass ``encoding=`` / ``errors=`` for other encodings.
"""

from __future__ import annotations

import csv as _stdlib_csv
import io
from pathlib import Path

from .base import DEFAULT_ENCODING, Document, read_text_file


def _escape_pipe(s: str) -> str:
    """Markdown tables can't contain unescaped ``|``."""
    return s.replace("|", "\\|").replace("\n", " ")


def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return "(empty)"
    header = rows[0]
    body = rows[1:]
    out = [
        "| " + " | ".join(_escape_pipe(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        # Pad short rows so the table stays valid.
        padded = row + [""] * (len(header) - len(row))
        out.append(
            "| " + " | ".join(_escape_pipe(c) for c in padded) + " |"
        )
    return "\n".join(out)


def _load_delimited(
    path: Path,
    delimiter: str,
    format_name: str,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
) -> Document:
    # Decode the bytes ourselves (no newline translation, as the csv
    # module wants) so a decode failure names the ``encoding=`` option.
    raw = read_text_file(
        path, encoding=encoding, errors=errors, universal_newlines=False
    )
    reader = _stdlib_csv.reader(io.StringIO(raw, newline=""), delimiter=delimiter)
    rows = list(reader)

    table = _rows_to_markdown_table(rows)
    content = f"# {path.name}\n\n{table}\n"
    return Document(
        content=content,
        metadata={
            "source": str(path),
            "format": format_name,
            "row_count": max(len(rows) - 1, 0),  # excluding header
            "column_count": len(rows[0]) if rows else 0,
        },
    )


def load_csv(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
) -> Document:
    """Load a comma-separated file → markdown table.

    ``encoding`` defaults to ``"utf-8-sig"`` (UTF-8, BOM stripped);
    ``errors`` is the codec error handler.
    """
    return _load_delimited(
        Path(path),
        delimiter=",",
        format_name="csv",
        encoding=encoding,
        errors=errors,
    )


def load_tsv(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
) -> Document:
    """Load a tab-separated file → markdown table.

    ``encoding`` / ``errors`` as for :func:`load_csv`.
    """
    return _load_delimited(
        Path(path),
        delimiter="\t",
        format_name="tsv",
        encoding=encoding,
        errors=errors,
    )
