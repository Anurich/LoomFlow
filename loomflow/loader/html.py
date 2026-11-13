"""HTML loader → markdown.

Uses ``beautifulsoup4`` (lazy import) to walk the DOM and emit
markdown that preserves heading + paragraph + list + table structure,
however deeply the page nests them in ``<div>`` / ``<section>`` /
``<article>`` / ``<main>`` wrappers. Strips ``<script>`` / ``<style>``
content and comments. Drops most attributes; the goal is to keep the
textual structure, not pixel-perfect rendering.
"""

from __future__ import annotations

from pathlib import Path

from .base import DEFAULT_ENCODING, Document, read_text_file

_BLOCK_TAGS_TO_HEADINGS: dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}

# Never rendered: code, styling, and the document head (the title is
# read separately).
_SKIP_TAGS = frozenset({"script", "style", "template", "head"})

# Elements that start a new block. A container holding any of these
# is walked child by child; one holding only inline content is a
# paragraph.
_BLOCK_TAGS = frozenset(
    {
        *_BLOCK_TAGS_TO_HEADINGS,
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "nav",
        "aside",
        "ul",
        "ol",
        "li",
        "table",
        "pre",
        "blockquote",
        "figure",
        "figcaption",
        "form",
        "dl",
        "dt",
        "dd",
        "details",
        "summary",
        "hr",
    }
)

# Text-node subclasses that are markup, not content.
_NON_CONTENT_NODES = frozenset(
    {"Comment", "Doctype", "Declaration", "ProcessingInstruction"}
)


def _is_content_text(node: object) -> bool:
    return type(node).__name__ not in _NON_CONTENT_NODES


def _inline_text(node: object) -> str:
    return str(
        node.get_text(separator=" ", strip=True)  # type: ignore[attr-defined]
    )


def _has_block_descendant(node: object) -> bool:
    find = getattr(node, "find", None)
    return find is not None and find(list(_BLOCK_TAGS)) is not None


def _walk(node: object, parts: list[str]) -> None:
    """Recursively walk a BeautifulSoup node and append markdown
    fragments to ``parts``."""
    name = getattr(node, "name", None)

    # Text node
    if name is None:
        text = str(node).strip() if _is_content_text(node) else ""
        if text:
            parts.append(text)
        return

    if name in _SKIP_TAGS:
        return

    if name in _BLOCK_TAGS_TO_HEADINGS:
        level = _BLOCK_TAGS_TO_HEADINGS[name]
        text = _inline_text(node)
        if text:
            parts.append(f"\n\n{'#' * level} {text}\n")
        return

    if name == "p":
        text = _inline_text(node)
        if text:
            parts.append(f"\n\n{text}\n")
        return

    if name in ("ul", "ol"):
        parts.append("\n")  # a blank line before the list
        _walk_list(node, parts, depth=0)
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

    _walk_container(node, parts)


def _walk_container(node: object, parts: list[str]) -> None:
    """A ``<div>`` / ``<section>`` / ``<article>`` / ``<main>`` /
    ``<body>`` or any other wrapper.

    With only inline content it is one paragraph. Otherwise its
    children are walked in order, so the headings, lists and tables
    inside keep their structure (flattening the whole container to
    text used to lose them); inline content between blocks is
    gathered into paragraphs.
    """
    if not _has_block_descendant(node):
        text = _inline_text(node)
        if text:
            parts.append(f"\n\n{text}\n")
        return

    run: list[str] = []  # consecutive inline pieces → one paragraph

    def flush() -> None:
        if run:
            parts.append("\n\n" + " ".join(run) + "\n")
            run.clear()

    for child in getattr(node, "children", []):
        child_name = getattr(child, "name", None)
        if child_name is None:
            text = str(child).strip() if _is_content_text(child) else ""
            if text:
                run.append(text)
        elif child_name in _SKIP_TAGS or child_name == "br":
            continue
        elif child_name in _BLOCK_TAGS or _has_block_descendant(child):
            flush()
            _walk(child, parts)
        else:  # inline element (<a>, <span>, <b>, …)
            text = _inline_text(child)
            if text:
                run.append(text)
    flush()


def _walk_list(node: object, parts: list[str], depth: int) -> None:
    """Emit a ``<ul>`` / ``<ol>`` as markdown items; nested lists
    become indented sub-items instead of being folded into their
    parent's text."""
    ordered = getattr(node, "name", None) == "ol"
    indent = "  " * depth
    for i, li in enumerate(
        node.find_all("li", recursive=False),  # type: ignore[attr-defined]
        start=1,
    ):
        own: list[str] = []
        nested: list[object] = []
        for child in li.children:
            child_name = getattr(child, "name", None)
            if child_name in ("ul", "ol"):
                nested.append(child)
            elif child_name is None:
                text = str(child).strip() if _is_content_text(child) else ""
                if text:
                    own.append(text)
            elif child_name not in _SKIP_TAGS:
                text = _inline_text(child)
                if text:
                    own.append(text)
        text = " ".join(own)
        if text:
            marker = f"{i}." if ordered else "-"
            parts.append(f"{indent}{marker} {text}\n")
        for sub in nested:
            _walk_list(sub, parts, depth + 1)


def load_html(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "replace",
) -> Document:
    """Load an HTML file → markdown.

    ``encoding`` defaults to UTF-8 (byte-order mark stripped) and
    undecodable bytes are replaced rather than raising (``errors=``),
    as before.

    Requires ``beautifulsoup4``:
    ``pip install 'loomflow[loader-html]'``.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "beautifulsoup4 is not installed. "
            "Install with: pip install 'loomflow[loader-html]'."
        ) from exc

    p = Path(path)
    raw = read_text_file(p, encoding=encoding, errors=errors)
    soup = BeautifulSoup(raw, "html.parser")

    title_tag = soup.find("title")
    title = (
        title_tag.get_text(strip=True) if title_tag else p.stem
    )

    # Scripts and styles are never content, wherever they sit.
    for tag in soup(["script", "style", "template"]):
        tag.decompose()

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
