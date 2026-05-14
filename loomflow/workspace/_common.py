"""Helpers shared between :class:`LocalDiskWorkspace` and
:class:`InMemoryWorkspace`: slugification, frontmatter parsing /
rendering, and the canonical index renderer.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .types import Note, NoteKind, NoteMatch, NoteSummary

# ---------------------------------------------------------------------------
# Slug + lede
# ---------------------------------------------------------------------------

_NON_WORD = re.compile(r"[^\w]+")
_LEADING_DASH = re.compile(r"^-+|-+$")


def slugify_title(title: str, *, max_len: int = 60) -> str:
    """Turn a free-form title into a slug fragment.

    ``"Population trends 2026"`` -> ``"population-trends-2026"``.

    The disk backend prefixes this with ``"NNN-"`` per author to
    keep slugs deterministic + sortable; the slug fragment alone
    isn't unique across authors.
    """
    s = _NON_WORD.sub("-", title.lower())
    s = _LEADING_DASH.sub("", s)
    if not s:
        s = "untitled"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


def extract_lede(body: str, *, max_len: int = 120) -> str:
    """First non-empty, non-heading line of ``body``, truncated.

    Used as the index entry's one-liner. Skips markdown headings
    (``#``) and code-fence delimiters so the lede is prose, not
    a section header.
    """
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("```"):
            continue
        if len(stripped) > max_len:
            return stripped[:max_len].rstrip() + "…"
        return stripped
    return ""


_QUERY_SPLIT = re.compile(r"[^\w]+")


def tokenize_query(q: str) -> list[str]:
    """Split a search query into lowercased terms.

    Punctuation and whitespace are separators; empty terms drop
    out. A single-word query yields exactly one term, so
    single-word search behaves identically to the pre-tokenization
    substring match.
    """
    return [t for t in _QUERY_SPLIT.split(q.lower()) if t]


def score_bm25(q: str, notes: list[Note], limit: int) -> list[NoteMatch]:
    """Tokenized substring scoring — shared by both workspace backends.

    The query is split into terms; each term is scored independently
    against every note at three tiers (title 1.0 > tag 0.7 > body
    0.5), and a note's score is those per-term tiers AVERAGED over
    the query terms:

    * a note matching ALL terms in its title scores 1.0;
    * a note matching half the terms in its body scores 0.25;
    * a note matching NONE of the terms is dropped.

    This is OR-semantics ranked by coverage — what an agent expects
    from ``search_notes("conda env conflict")``. The earlier
    implementation tested the whole query as ONE substring, so any
    multi-word query returned nothing unless that exact phrase
    appeared contiguously. A single-word query still collapses to
    the old substring-tier match, so existing callers are
    unaffected.
    """
    terms = tokenize_query(q)
    if not terms:
        return []
    scored: list[tuple[float, str, Note]] = []
    for note in notes:
        title_l = note.title.lower()
        body_l = note.body.lower()
        tags_l = [t.lower() for t in note.tags]
        total = 0.0
        snippet = ""
        for term in terms:
            if term in title_l:
                total += 1.0
            elif any(term in t for t in tags_l):
                total += 0.7
            elif term in body_l:
                total += 0.5
                # Snippet comes from the FIRST body-matched term,
                # with ±context; title-/tag-only hits fall back to
                # the title below.
                if not snippet:
                    idx = body_l.find(term)
                    start = max(0, idx - 40)
                    end = min(len(note.body), idx + len(term) + 60)
                    snip = (
                        note.body[start:end].replace("\n", " ").strip()
                    )
                    if start > 0:
                        snip = "…" + snip
                    if end < len(note.body):
                        snip = snip + "…"
                    snippet = snip
        if total <= 0.0:
            continue
        scored.append((total / len(terms), snippet or note.title, note))
    scored.sort(key=lambda t: (t[0], t[2].updated_at), reverse=True)
    out: list[NoteMatch] = []
    for score, snippet, note in scored[:limit]:
        out.append(
            NoteMatch(
                summary=summary_from_note(note),
                score=score,
                snippet=snippet or extract_lede(note.body),
            )
        )
    return out


def summary_from_note(note: Note) -> NoteSummary:
    return NoteSummary(
        slug=note.slug,
        author=note.author,
        title=note.title,
        kind=note.kind,
        tags=list(note.tags),
        created_at=note.created_at,
        updated_at=note.updated_at,
        lede=extract_lede(note.body),
        namespace=note.namespace,
        archived_at=note.archived_at,
        cited_count=note.cited_count,
        success_count=note.success_count,
        last_cited_at=note.last_cited_at,
    )


# ---------------------------------------------------------------------------
# YAML frontmatter (small, hand-rolled — no pyyaml dep)
# ---------------------------------------------------------------------------

_FRONTMATTER_FENCE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def render_note_file(note: Note) -> str:
    """Render a :class:`Note` as a markdown file with YAML
    frontmatter. The body lands verbatim under the fence so it
    round-trips through :func:`parse_note_file` byte-stable.

    Optional fields (``user_id``, ``run_id``, ``namespace``,
    ``archived_at``, ``answered``, ``answered_by``, ``parent_slug``)
    are emitted ONLY when set, keeping legacy notes' frontmatter
    minimal and the diff against the v0.9.x file shape small.
    """
    tags_inline = ", ".join(note.tags) if note.tags else ""
    fm_lines = [
        "---",
        f"slug: {note.slug}",
        f"title: {_yaml_quote(note.title)}",
        f"author: {note.author}",
        f"kind: {note.kind}",
        f"tags: [{tags_inline}]",
        f"created_at: {note.created_at.isoformat()}",
        f"updated_at: {note.updated_at.isoformat()}",
    ]
    if note.user_id is not None:
        fm_lines.append(f"user_id: {_yaml_quote(note.user_id)}")
    if note.run_id is not None:
        fm_lines.append(f"run_id: {_yaml_quote(note.run_id)}")
    if note.namespace is not None:
        fm_lines.append(f"namespace: {_yaml_quote(note.namespace)}")
    if note.archived_at is not None:
        fm_lines.append(f"archived_at: {note.archived_at.isoformat()}")
    if note.answered is not None:
        fm_lines.append(f"answered: {'true' if note.answered else 'false'}")
    if note.answered_by is not None:
        fm_lines.append(f"answered_by: {_yaml_quote(note.answered_by)}")
    if note.parent_slug is not None:
        fm_lines.append(f"parent_slug: {_yaml_quote(note.parent_slug)}")
    # Citation fields — emit only when non-default so legacy notes
    # stay lean. Zero counts and None timestamps are absent.
    if note.cited_count:
        fm_lines.append(f"cited_count: {note.cited_count}")
    if note.success_count:
        fm_lines.append(f"success_count: {note.success_count}")
    if note.last_cited_at is not None:
        fm_lines.append(
            f"last_cited_at: {note.last_cited_at.isoformat()}"
        )
    fm_lines.append("---")
    return "\n".join(fm_lines) + "\n\n" + note.body.rstrip() + "\n"


def parse_note_file(text: str) -> tuple[dict[str, Any], str]:
    """Parse a workspace note file into ``(frontmatter, body)``.

    The frontmatter parser is deliberately small (matching the
    skills frontmatter parser's restraint): top-level scalars,
    inline ``[a, b]`` lists, ISO datetimes, quoted strings.
    Anything more exotic gets stored as a raw string and the
    caller can re-validate.
    """
    match = _FRONTMATTER_FENCE.match(text)
    if match is None:
        return {}, text
    body = text[match.end():]
    fm_text = match.group(1)
    fm: dict[str, Any] = {}
    for raw_line in fm_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = _parse_yaml_scalar(value.strip())
    return fm, body


def _yaml_quote(s: str) -> str:
    """Quote a string for YAML inline output. Fast path for safe
    bareword strings; double-quotes for anything with special
    characters."""
    if not s:
        return '""'
    if re.fullmatch(r"[\w\- ./]+", s) and ":" not in s:
        return s
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _parse_yaml_scalar(text: str) -> Any:
    if text == "" or text.lower() == "null" or text == "~":
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",") if item.strip()]
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def note_from_frontmatter(fm: dict[str, Any], body: str) -> Note:
    """Reconstruct a :class:`Note` from parsed frontmatter + body.

    All v0.10.x-and-later fields (``namespace``, ``archived_at``,
    ``answered``, ``answered_by``, ``parent_slug``) default to
    ``None`` when absent — legacy v0.9.x notes load cleanly with
    the new fields just empty.
    """
    kind: NoteKind = fm.get("kind") or "note"  # type: ignore[assignment]
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return Note(
        slug=str(fm["slug"]),
        author=str(fm["author"]),
        title=str(fm["title"]),
        body=body.rstrip(),
        kind=kind,
        tags=list(tags),
        created_at=_parse_dt(fm.get("created_at")),
        updated_at=_parse_dt(fm.get("updated_at")),
        user_id=fm.get("user_id"),
        run_id=fm.get("run_id"),
        namespace=fm.get("namespace"),
        archived_at=_parse_dt_optional(fm.get("archived_at")),
        answered=_parse_bool_optional(fm.get("answered")),
        answered_by=fm.get("answered_by"),
        parent_slug=fm.get("parent_slug"),
        cited_count=_parse_int(fm.get("cited_count")),
        success_count=_parse_int(fm.get("success_count")),
        last_cited_at=_parse_dt_optional(fm.get("last_cited_at")),
    )


def _parse_int(value: Any) -> int:
    """Defensive int parse — legacy notes have no cited_count field
    (returns 0) and human-edited values may be strings."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    from datetime import UTC
    return datetime.now(UTC)


def _parse_dt_optional(value: Any) -> datetime | None:
    """Like :func:`_parse_dt` but returns ``None`` for absent /
    null values instead of "now"."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None


def _parse_bool_optional(value: Any) -> bool | None:
    """Tri-state parse — ``None`` stays ``None``; truthy strings
    (``"true"``, ``"yes"``, ``"1"``) map to True; falsy strings
    (``"false"``, ``"no"``, ``"0"``) map to False."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


# ---------------------------------------------------------------------------
# Index rendering — the canonical WORKSPACE.md
# ---------------------------------------------------------------------------


def render_workspace_index(summaries: list[NoteSummary]) -> str:
    """Render every note in the workspace as a single markdown
    document.

    Sections:

    1. Header + count + latest-activity timestamp.
    2. Per-author table of contents — counts + last write per
       author so a teammate sees who's contributing what.
    3. Chronological timeline (newest first) — slug, title, kind,
       lede.
    4. Open questions section (kind=="question") promoted to the
       top so they're visible.

    This gets written to ``WORKSPACE.md`` after every note write
    on the disk backend; agents read it via ``list_notes()`` or
    by directly reading the file with a normal ``read_tool``.
    """
    if not summaries:
        return _EMPTY_INDEX

    latest = max(s.updated_at for s in summaries).isoformat()
    authors: dict[str, list[NoteSummary]] = {}
    for s in summaries:
        authors.setdefault(s.author, []).append(s)

    parts: list[str] = []
    parts.append("# Workspace notebook\n")
    parts.append(
        f"{len(summaries)} note(s) from {len(authors)} agent(s). "
        f"Latest activity: {latest}.\n"
    )

    # Open questions first — surface things that need attention.
    open_qs = [s for s in summaries if s.kind == "question"]
    if open_qs:
        parts.append("## Open questions\n")
        for q in sorted(open_qs, key=lambda s: s.updated_at, reverse=True):
            parts.append(f"- **`{q.slug}`** [{q.author}] — {q.title}")
            if q.lede:
                parts.append(f"  > {q.lede}")
        parts.append("")

    # Per-author summary.
    parts.append("## Contributors\n")
    for author in sorted(authors):
        author_notes = authors[author]
        last = max(n.updated_at for n in author_notes).isoformat()
        kinds = sorted({n.kind for n in author_notes})
        parts.append(
            f"- **{author}** — {len(author_notes)} note(s), kinds: "
            f"{', '.join(kinds)}, last: {last}"
        )
    parts.append("")

    # Chronological timeline.
    parts.append("## All notes (newest first)\n")
    chrono = sorted(summaries, key=lambda s: s.updated_at, reverse=True)
    for s in chrono:
        tag_suffix = f" `[{', '.join(s.tags)}]`" if s.tags else ""
        parts.append(
            f"- **`{s.slug}`** [{s.author} · {s.kind}]{tag_suffix} "
            f"— {s.title}"
        )
        if s.lede:
            parts.append(f"  > {s.lede}")

    parts.append("")
    return "\n".join(parts)


_EMPTY_INDEX = (
    "# Workspace notebook\n\n"
    "Empty. Be the first to add a note — call `note(title, content)` to "
    "share findings with your teammates.\n"
)
