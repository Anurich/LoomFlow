"""Excel (.xlsx / .xlsm) loader → markdown.

Uses ``openpyxl`` (lazy import). Each sheet becomes a markdown
section with the sheet name as an ``#`` heading and the cell grid
as a markdown table. Empty rows / columns at the edges are
trimmed.

For very wide / tall sheets, the markdown output can balloon —
prefer pandas + tailored summarization for spreadsheets that
exceed an LLM's comfortable context. This loader is intended for
the common case (config sheets, lookup tables, structured records).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Document


def _stringify(cell: Any) -> str:
    """Render a cell value as a markdown-table-safe string."""
    if cell is None:
        return ""
    s = str(cell)
    return s.replace("|", "\\|").replace("\n", " ")


def _trim_empty_edges(
    rows: list[list[Any]],
) -> list[list[str]]:
    """Drop trailing all-empty rows and trailing empty columns."""
    string_rows = [
        [_stringify(c) for c in row] for row in rows
    ]
    # Trim trailing empty rows
    while string_rows and not any(
        c.strip() for c in string_rows[-1]
    ):
        string_rows.pop()
    if not string_rows:
        return []
    # Trim trailing empty columns
    max_meaningful = 0
    for row in string_rows:
        for i in range(len(row) - 1, -1, -1):
            if row[i].strip():
                max_meaningful = max(max_meaningful, i + 1)
                break
    if max_meaningful:
        string_rows = [row[:max_meaningful] for row in string_rows]
    else:
        string_rows = []
    return string_rows


def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return "(empty)"
    header = rows[0]
    body = rows[1:]
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        out.append("| " + " | ".join(padded) + " |")
    return "\n".join(out)


def load_excel(path: str | Path) -> Document:
    """Load an Excel workbook → markdown.

    Each sheet becomes ``## {sheet_name}`` with the cell grid as a
    markdown table. Formula cells return their cached values
    (data_only=True).

    Requires ``openpyxl``:
    ``pip install 'jeevesagent[loader-excel]'``.
    """
    try:
        import openpyxl  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openpyxl is not installed. "
            "Install with: pip install 'jeevesagent[loader-excel]'."
        ) from exc

    p = Path(path)
    wb = openpyxl.load_workbook(str(p), data_only=True, read_only=True)

    parts: list[str] = [f"# {p.stem}\n"]
    sheet_names: list[str] = []
    total_rows = 0
    for sheet in wb.worksheets:
        sheet_names.append(sheet.title)
        rows = list(sheet.iter_rows(values_only=True))
        trimmed = _trim_empty_edges([list(r) for r in rows])
        total_rows += max(len(trimmed) - 1, 0)
        parts.append(f"## {sheet.title}\n")
        if not trimmed:
            parts.append("(empty)\n")
            continue
        parts.append(_rows_to_markdown_table(trimmed))
        parts.append("")

    wb.close()
    return Document(
        content="\n".join(parts).rstrip() + "\n",
        metadata={
            "source": str(p),
            "format": "xlsx",
            "sheet_count": len(sheet_names),
            "sheet_names": sheet_names,
            "row_count": total_rows,
        },
    )
