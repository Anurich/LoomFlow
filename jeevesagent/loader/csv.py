"""CSV / TSV loaders → markdown table.

Uses the stdlib ``csv`` module. The first row is treated as the
header. Each row becomes a markdown table row. For very large CSVs
(hundreds of thousands of rows), prefer pandas + manual chunking;
this loader is intended for documents an LLM will read end-to-end.
"""

from __future__ import annotations

import csv as _stdlib_csv
from pathlib import Path

from .base import Document


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
    path: Path, delimiter: str, format_name: str
) -> Document:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = _stdlib_csv.reader(fh, delimiter=delimiter)
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


def load_csv(path: str | Path) -> Document:
    """Load a comma-separated file → markdown table."""
    return _load_delimited(Path(path), delimiter=",", format_name="csv")


def load_tsv(path: str | Path) -> Document:
    """Load a tab-separated file → markdown table."""
    return _load_delimited(Path(path), delimiter="\t", format_name="tsv")
