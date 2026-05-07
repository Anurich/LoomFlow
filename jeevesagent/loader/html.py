"""HTML loader → markdown.

Uses ``beautifulsoup4`` (lazy import) to walk the DOM and emit
markdown that preserves heading + paragraph + list structure.
Strips ``<script>`` / ``<style>`` content. Drops most attributes;
the goal is to keep the textual structure, not pixel-perfect
rendering.
"""

from __future__ import annotations

from pathlib import Path

from .base import Document

_BLOCK_TAGS_TO_HEADINGS: dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}


def _walk(node: object, parts: list[str]) -> None:
    """Recursively walk a BeautifulSoup node and append markdown
    fragments to ``parts``."""
    name = getattr(node, "name", None)

    # Text node
    if name is None:
        text = str(node).strip()
        if text:
            parts.append(text)
        return

    if name in ("script", "style"):
        return

    if name in _BLOCK_TAGS_TO_HEADINGS:
        level = _BLOCK_TAGS_TO_HEADINGS[name]
        text = node.get_text(separator=" ", strip=True)  # type: ignore[attr-defined]
        if text:
            parts.append(f"\n\n{'#' * level} {text}\n")
        return

    if name in ("p", "div", "section", "article", "main"):
        text = node.get_text(separator=" ", strip=True)  # type: ignore[attr-defined]
        if text:
            parts.append(f"\n\n{text}\n")
        return

    if name in ("ul", "ol"):
        ordered = name == "ol"
        for i, li in enumerate(
            node.find_all("li", recursive=False),  # type: ignore[attr-defined]
            start=1,
        ):
            text = li.get_text(separator=" ", strip=True)
            if text:
                marker = f"{i}." if ordered else "-"
                parts.append(f"{marker} {text}\n")
        parts.append("\n")
        return

    if name in ("table",):
        rows: list[list[str]] = []
        for tr in node.find_all("tr"):  # type: ignore[attr-defined]
            cells = [
                td.get_text(separator=" ", strip=True).replace(
                    "|", "\\|"
                )
                for td in tr.find_all(["th", "td"])
            ]
            if cells:
                rows.append(cells)
        if rows:
            header = rows[0]
            parts.append(
                "\n\n| "
                + " | ".join(header)
                + " |\n| "
                + " | ".join("---" for _ in header)
                + " |\n"
            )
            for row in rows[1:]:
                padded = row + [""] * (len(header) - len(row))
                parts.append("| " + " | ".join(padded) + " |\n")
        return

    if name in ("br",):
        parts.append("\n")
        return

    # Generic container — descend
    for child in getattr(node, "children", []):
        _walk(child, parts)


def load_html(path: str | Path) -> Document:
    """Load an HTML file → markdown.

    Requires ``beautifulsoup4``:
    ``pip install 'jeevesagent[loader-html]'``.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "beautifulsoup4 is not installed. "
            "Install with: pip install 'jeevesagent[loader-html]'."
        ) from exc

    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    title_tag = soup.find("title")
    title = (
        title_tag.get_text(strip=True) if title_tag else p.stem
    )

    body = soup.find("body") or soup
    parts: list[str] = [f"# {title}\n"]
    _walk(body, parts)

    # Collapse runs of blank lines and strip trailing whitespace.
    text = "".join(parts)
    out_lines = []
    prev_blank = False
    for line in text.splitlines():
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        out_lines.append(line)
        prev_blank = is_blank
    content = "\n".join(out_lines).strip() + "\n"

    return Document(
        content=content,
        metadata={
            "source": str(p),
            "format": "html",
            "title": title,
        },
    )
