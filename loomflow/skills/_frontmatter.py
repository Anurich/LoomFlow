"""Minimal YAML frontmatter parser for SKILL.md files.

Why hand-rolled?
----------------

We want zero new dependencies. ``pyyaml`` is heavy and pulls C
extensions; ``ruamel`` is heavier. SKILL.md frontmatter only needs
a small YAML subset:

* Top-level scalar fields: ``name``, ``description``, ``license``,
  ``compatibility``
* One-level nested dict: ``metadata: {author, version, ...}``
* Inline or block lists: ``allowed_tools: [bash, read]`` /
  ``allowed_tools:\n  - bash\n  - read``
* Multi-line strings via ``|`` (literal) or ``>`` (folded)

Anything more exotic (anchors, aliases, complex flow-style nesting)
the parser refuses with a clear error. SKILL.md doesn't need it
and silent surprises are worse than a strict failure.

Public surface: :func:`parse_frontmatter` returns
``(metadata: dict, body: str)``. Raises :class:`FrontmatterError`
on malformed input.
"""

from __future__ import annotations

import re
from typing import Any


class FrontmatterError(ValueError):
    """Raised on malformed SKILL.md frontmatter."""


_FENCE_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into ``(frontmatter_dict, body_text)``.

    The frontmatter is everything between two ``---`` lines at the
    start of the file. Body is the rest. If no frontmatter fence is
    found, raises :class:`FrontmatterError` — SKILL.md without
    frontmatter is almost always a mistake, so we fail loudly.
    """
    match = _FENCE_RE.match(text)
    if match is None:
        raise FrontmatterError(
            "SKILL.md must start with a YAML frontmatter block "
            "delimited by '---' lines. Got: "
            f"{text[:80]!r}..."
        )
    yaml_block = match.group(1)
    body = text[match.end() :]
    metadata = _parse_yaml(yaml_block)
    if not isinstance(metadata, dict):
        raise FrontmatterError(
            "Frontmatter must be a mapping (key: value pairs); "
            f"got {type(metadata).__name__}"
        )
    return metadata, body


# ---------------------------------------------------------------------------
# Tiny YAML subset parser
# ---------------------------------------------------------------------------


def _parse_yaml(text: str) -> Any:
    """Parse a YAML subset into Python objects."""
    lines = text.splitlines()
    parser = _Parser(lines)
    return parser.parse_block(indent=0)


class _Parser:
    def __init__(self, lines: list[str]) -> None:
        # Strip trailing whitespace; preserve leading (used for
        # indentation detection).
        self._lines = [line.rstrip() for line in lines]
        self._i = 0

    def _peek(self) -> str | None:
        # Skip blank + comment lines.
        while self._i < len(self._lines):
            line = self._lines[self._i]
            stripped = line.lstrip()
            if stripped == "" or stripped.startswith("#"):
                self._i += 1
                continue
            return line
        return None

    def _consume(self) -> str:
        line = self._lines[self._i]
        self._i += 1
        return line

    def parse_block(self, indent: int) -> Any:
        """Parse a mapping block at the given indent level."""
        result: dict[str, Any] = {}
        while True:
            line = self._peek()
            if line is None:
                break
            line_indent = len(line) - len(line.lstrip())
            if line_indent < indent:
                break
            if line_indent > indent:
                raise FrontmatterError(
                    f"Unexpected indentation at line {self._i + 1}: "
                    f"{line!r}"
                )
            stripped = line.lstrip()
            if ":" not in stripped:
                raise FrontmatterError(
                    f"Line {self._i + 1} is not a mapping entry: "
                    f"{stripped!r}"
                )
            key, _, rest = stripped.partition(":")
            key = key.strip()
            rest = rest.strip()
            self._consume()

            if rest == "" or rest == "|" or rest == ">":
                # Either a nested mapping, a block list, or a
                # multi-line string.
                next_line = self._peek()
                if next_line is None:
                    result[key] = "" if rest in {"|", ">"} else None
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    result[key] = "" if rest in {"|", ">"} else None
                    continue
                if rest == "|":
                    result[key] = self._parse_literal_string(next_indent)
                elif rest == ">":
                    result[key] = self._parse_folded_string(next_indent)
                elif next_line.lstrip().startswith("- "):
                    result[key] = self._parse_block_list(next_indent)
                else:
                    result[key] = self.parse_block(next_indent)
            elif rest.startswith("[") and rest.endswith("]"):
                result[key] = _parse_inline_list(rest)
            elif rest.startswith("{") and rest.endswith("}"):
                raise FrontmatterError(
                    "Inline flow-style mappings ({a: 1, b: 2}) "
                    "aren't supported. Use block style:\n"
                    "  key:\n    a: 1\n    b: 2"
                )
            else:
                result[key] = _parse_scalar(rest)
        return result

    def _parse_block_list(self, indent: int) -> list[Any]:
        items: list[Any] = []
        while True:
            line = self._peek()
            if line is None:
                break
            line_indent = len(line) - len(line.lstrip())
            if line_indent != indent or not line.lstrip().startswith("- "):
                break
            self._consume()
            value = line.lstrip()[2:].strip()
            items.append(_parse_scalar(value))
        return items

    def _parse_literal_string(self, indent: int) -> str:
        # ``|`` keeps newlines verbatim.
        parts: list[str] = []
        while True:
            line = self._peek()
            if line is None:
                break
            line_indent = len(line) - len(line.lstrip())
            if line_indent < indent:
                break
            self._consume()
            parts.append(line[indent:])
        return "\n".join(parts).rstrip()

    def _parse_folded_string(self, indent: int) -> str:
        # ``>`` joins lines with spaces; blank lines become newlines.
        parts: list[str] = []
        while True:
            line = self._peek()
            if line is None:
                break
            line_indent = len(line) - len(line.lstrip())
            if line_indent < indent and line.strip() != "":
                break
            self._consume()
            parts.append(line[indent:] if len(line) >= indent else "")
        out: list[str] = []
        buf: list[str] = []
        for part in parts:
            if part == "":
                if buf:
                    out.append(" ".join(buf))
                    buf = []
                out.append("")
            else:
                buf.append(part.strip())
        if buf:
            out.append(" ".join(buf))
        return "\n".join(out).strip()


def _parse_inline_list(text: str) -> list[Any]:
    inner = text[1:-1].strip()
    if inner == "":
        return []
    return [_parse_scalar(item.strip()) for item in inner.split(",")]


def _parse_scalar(text: str) -> Any:
    """Parse a single scalar value: string, int, float, bool, null."""
    if text == "" or text == "~" or text.lower() == "null":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    # Quoted string — strip quotes verbatim.
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    # Integer.
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    # Float.
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    # Bare string.
    return text
