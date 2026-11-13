"""Chunking strategies for splitting documents into LLM-friendly pieces.

Four strategies, picked by what your downstream RAG / context
window needs:

* :class:`RecursiveChunker` — the production default. Splits on a
  hierarchy of separators (paragraph → line → sentence → word) so
  semantic boundaries survive when possible. The same algorithm
  LangChain's ``RecursiveCharacterTextSplitter`` uses; widely
  recommended in Anthropic's RAG cookbook.
* :class:`MarkdownChunker` — splits on heading boundaries (``#``,
  ``##``, ``###``, …). Each chunk's metadata records the trail of
  parent headers, so retrieval surfaces section context. Use this
  for the markdown produced by the PDF / DOCX / Excel loaders.
* :class:`SentenceChunker` — sentence-boundary chunks. Use for
  QA-style RAG where each chunk should answer one short question.
* :class:`TokenChunker` — chunk by token count via ``tiktoken``
  (lazy import). Use when you need tight control over context-
  window fit.

Defaults
--------

All chunkers default to ``chunk_size=800`` characters with
``chunk_overlap=100`` (12.5% overlap) — the values Anthropic
recommends in their RAG documentation. Override per-chunker as
needed.

``chunk_size`` is a hard upper bound for the character chunkers: no
chunk is ever longer. ``chunk_overlap`` is an upper bound too, and
overlap is made of *whole pieces* (words, sentences, paragraphs —
whatever the text was split on), never a cut through the middle of
one. Consecutive chunks therefore share up to ``chunk_overlap``
characters, and share nothing when the piece at the boundary is
itself longer than ``chunk_overlap`` (e.g. 300-character paragraphs
with ``chunk_overlap=100``). Only :class:`TokenChunker` overlaps by
an exact amount.

Convenience factory: :func:`chunk` picks a strategy by name::

    from loomflow.loader import chunk

    pieces = chunk(text, strategy="recursive", chunk_size=800)
    pieces = chunk(text, strategy="markdown")
    pieces = chunk(text, strategy="sentence", chunk_size=400)
    pieces = chunk(text, strategy="token", chunk_size=512)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .base import Chunk

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Anthropic's RAG cookbook recommends 200-1000 tokens per chunk with
# 10-20% overlap. With ~4 chars per token average, 800 chars ≈ 200
# tokens — solid default for both Claude and OpenAI embedding models.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Chunker(Protocol):
    """Anything with a ``split(text) -> list[Chunk]`` method."""

    def split(self, text: str, *, source: str = "") -> list[Chunk]:
        ...


# ---------------------------------------------------------------------------
# RecursiveChunker — production default
# ---------------------------------------------------------------------------

# Hierarchy of separators, ordered most-semantic-first. The
# splitter prefers paragraph breaks; falls back to line breaks,
# then sentence terminators, then whitespace. Same hierarchy
# LangChain's RecursiveCharacterTextSplitter uses.
_DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",   # paragraph
    "\n",     # line
    ". ",     # sentence (period + space)
    "! ",
    "? ",
    "; ",
    ", ",
    " ",      # word
    "",       # character (last resort)
)


class RecursiveChunker:
    """Recursive character splitter — the production workhorse.

    Produces chunks of at most ``chunk_size`` characters. Splits on
    a hierarchy of separators (paragraph → line → sentence → word →
    char), trying to preserve semantic boundaries.

    ``chunk_overlap`` is the most a chunk may repeat from the end of
    the previous one. The repeated text is always whole pieces at the
    level the text was split on, so the actual overlap is anywhere
    from 0 to ``chunk_overlap`` characters: text split into words
    overlaps by close to the full amount, while paragraphs longer
    than ``chunk_overlap`` do not overlap at all.

    ``separators`` may be customised. If none of them can break a
    run of text down to ``chunk_size`` (say ``["\\n\\n", " "]`` and a
    500-character token), the run is cut every ``chunk_size``
    characters, so the size bound holds whatever the separators.

    This is the algorithm LangChain calls
    ``RecursiveCharacterTextSplitter`` and the one most production
    RAG pipelines default to. Anthropic's cookbook specifically
    recommends it for general text.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Sequence[str] = _DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators)

    def split(self, text: str, *, source: str = "") -> list[Chunk]:
        if not text:
            return []
        # Chunks are stripped of surrounding whitespace (as LangChain
        # does), and whitespace-only ones dropped.
        merged = [c.strip() for c in self._recursive_split(text, self.separators)]
        merged = [c for c in merged if c]
        return [
            Chunk(
                content=c,
                metadata={
                    "source": source,
                    "index": i,
                    "chunk_size": len(c),
                    "strategy": "recursive",
                },
            )
            for i, c in enumerate(merged)
        ]

    def _recursive_split(
        self, text: str, separators: Sequence[str]
    ) -> list[str]:
        """Split ``text`` into chunks of at most ``chunk_size``.

        LangChain's ``RecursiveCharacterTextSplitter`` algorithm: split
        on the first separator present, merge consecutive small pieces
        back up to ``chunk_size`` (with overlap made of whole pieces),
        and recurse into any piece that is still too long using the
        finer separators — or, when none is left, cut it every
        ``chunk_size`` characters. Pieces keep their trailing
        separator, so joining a chunk's pieces reproduces the text
        exactly.
        """
        separator = separators[-1] if separators else ""
        finer: Sequence[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator, finer = sep, separators[i + 1 :]
                break
        if separator:
            parts = text.split(separator)
            pieces = [p + separator for p in parts[:-1]] + [parts[-1]]
        else:
            pieces = list(text)  # characters, the last resort
        chunks: list[str] = []
        small: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                small.append(piece)
                continue
            chunks.extend(self._merge_pieces(small))
            small = []
            if finer:
                chunks.extend(self._recursive_split(piece, finer))
            else:
                # No separator left can shorten it (custom separators
                # without ""): cut it hard, so ``chunk_size`` stays a
                # real upper bound.
                chunks.extend(
                    piece[start : start + self.chunk_size]
                    for start in range(0, len(piece), self.chunk_size)
                )
        chunks.extend(self._merge_pieces(small))
        return chunks

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Merge pieces into chunks of at most ``chunk_size``; each new
        chunk starts with the previous one's last pieces, up to
        ``chunk_overlap`` characters (whole pieces, never mid-word —
        so no overlap at all when the last piece alone is longer than
        ``chunk_overlap``)."""
        chunks: list[str] = []
        current: list[str] = []
        total = 0
        for piece in pieces:
            if current and total + len(piece) > self.chunk_size:
                chunks.append("".join(current))
                # Drop pieces from the front until what remains fits
                # as overlap and leaves room for the next piece.
                while current and (
                    total > self.chunk_overlap or total + len(piece) > self.chunk_size
                ):
                    total -= len(current.pop(0))
            current.append(piece)
            total += len(piece)
        if current:
            chunks.append("".join(current))
        return chunks


# ---------------------------------------------------------------------------
# MarkdownChunker — heading-aware
# ---------------------------------------------------------------------------


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})", re.MULTILINE)


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of fenced code blocks (CommonMark: a
    fence closes on a line of the same character, at least as long;
    an unclosed fence runs to the end)."""
    ranges: list[tuple[int, int]] = []
    open_at: int | None = None
    fence = ""
    for m in _FENCE_RE.finditer(text):
        marker = m.group(1)
        if open_at is None:
            open_at, fence = m.start(), marker
        elif marker[0] == fence[0] and len(marker) >= len(fence):
            ranges.append((open_at, m.end()))
            open_at = None
    if open_at is not None:
        ranges.append((open_at, len(text)))
    return ranges


class MarkdownChunker:
    """Split markdown on heading boundaries.

    Each chunk corresponds to one section: the heading line plus
    its content up to (but not including) the next heading at the
    same OR shallower depth. Long sections are further split via
    :class:`RecursiveChunker` so no chunk exceeds ``chunk_size``.

    A heading with nothing under it (a title followed directly by a
    sub-heading, say) is not emitted as a chunk of its own — a bare
    heading retrieves well and answers nothing. It is prepended to
    the next section's chunk instead, so the content carries its
    heading trail in the text as well as in the metadata.

    Each chunk's metadata records the trail of parent headers
    (the path from the document root to this section), letting the
    retriever show users where each chunk came from.

    Use this for markdown produced by the PDF / DOCX / Excel
    loaders — it preserves the document's hierarchy.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback = RecursiveChunker(chunk_size, chunk_overlap)

    def split(self, text: str, *, source: str = "") -> list[Chunk]:
        if not text:
            return []
        sections = self._merge_heading_only(self._split_into_sections(text))
        chunks: list[Chunk] = []
        chunk_index = 0
        for headers, content in sections:
            content_stripped = content.strip()
            if not content_stripped:
                continue
            if len(content) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        content=content,
                        metadata={
                            "source": source,
                            "index": chunk_index,
                            "chunk_size": len(content),
                            "headers": list(headers),
                            "strategy": "markdown",
                        },
                    )
                )
                chunk_index += 1
            else:
                # Section too big — fall back to recursive splitting,
                # carrying the headers into each sub-chunk's metadata.
                sub = self._fallback.split(content, source=source)
                for s in sub:
                    chunks.append(
                        Chunk(
                            content=s.content,
                            metadata={
                                "source": source,
                                "index": chunk_index,
                                "chunk_size": len(s.content),
                                "headers": list(headers),
                                "strategy": "markdown+recursive",
                            },
                        )
                    )
                    chunk_index += 1
        return chunks

    @staticmethod
    def _merge_heading_only(
        sections: list[tuple[list[str], str]],
    ) -> list[tuple[list[str], str]]:
        """Fold sections that are nothing but a heading line into the
        section that follows (which keeps its own header trail). A
        trailing heading with nothing after it stays as it is, so no
        text is ever dropped."""
        merged: list[tuple[list[str], str]] = []
        pending = ""
        pending_headers: list[str] = []
        for headers, content in sections:
            body = _HEADING_RE.sub("", content, count=1) if headers else content
            if headers and not body.strip():
                # Heading only: carry it forward.
                pending += content if content.endswith("\n") else content + "\n"
                pending_headers = headers
                continue
            merged.append((headers, pending + content))
            pending = ""
        if pending:
            merged.append((pending_headers, pending))
        return merged

    def _split_into_sections(
        self, text: str
    ) -> list[tuple[list[str], str]]:
        """Walk the markdown and split on heading boundaries.

        Returns a list of ``(headers, content)`` tuples where
        ``headers`` is the path of header strings from the root to
        this section.
        """
        # Find all heading positions, ignoring ``#`` lines inside
        # fenced code blocks (shell comments, Python comments, ...).
        fenced = _fenced_ranges(text)
        heading_matches = [
            m
            for m in _HEADING_RE.finditer(text)
            if not any(lo <= m.start() < hi for lo, hi in fenced)
        ]
        if not heading_matches:
            return [([], text)]

        sections: list[tuple[list[str], str]] = []
        # Anything before the first heading
        if heading_matches[0].start() > 0:
            preface = text[: heading_matches[0].start()]
            if preface.strip():
                sections.append(([], preface))

        # Maintain a stack of (level, title) so we can compute the
        # current header path for each section.
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(heading_matches):
            level = len(m.group(1))
            title = m.group(2)
            # Pop deeper-or-equal levels off the stack.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            headers = [t for _, t in stack]

            start = m.start()
            end = (
                heading_matches[i + 1].start()
                if i + 1 < len(heading_matches)
                else len(text)
            )
            sections.append((headers, text[start:end]))
        return sections


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------


# A possible sentence boundary: terminator(s), any closing quotes /
# brackets, then whitespace — or a CJK full stop, which ends a
# sentence with or without a following space.
_BOUNDARY_RE = re.compile(
    r"(?P<end>[.!?…]+[\"'”’»)\]]*)(?P<gap>\s+)"
    r"|(?P<cjk>[。！？]+[”’」』）]*)(?P<cjk_gap>\s*)"
)

# Abbreviations whose trailing period does not end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev",
        "gen", "col", "lt", "sgt", "capt", "gov", "sen", "rep", "hon",
        "vs", "cf", "al", "ca", "approx", "est", "viz",
        "e.g", "i.e", "eg", "ie", "a.m", "p.m",
        "fig", "figs", "eq", "eqs", "sec", "ch", "vol", "pp", "ed", "eds",
        "inc", "ltd", "co", "corp", "dept", "univ", "no", "nos",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    }
)
# The word a period closes (back to whitespace / an opening bracket).
_LAST_WORD_RE = re.compile(r"[^\s(\[\"'“‘«]+\Z")
# Dotted acronyms: "U.S", "e.g", "Ph.D" (the final period is the
# candidate boundary itself).
_DOTTED_ACRONYM_RE = re.compile(r"(?:[^\W\d_]{1,2}\.)+[^\W\d_]{1,2}\Z")
# These end a sentence when the next one starts with a capital
# ("…apples, pears, etc. The rest…") but not otherwise.
_SOFT_ABBREVIATIONS = frozenset({"etc", "no", "nos", "co", "inc", "ltd", "corp", "est"})


def _is_sentence_break(text: str, match: re.Match[str]) -> bool:
    """Decide whether a boundary candidate really ends a sentence."""
    if match.group("cjk") is not None:
        return True
    end = match.group("end")
    following = text[match.end() : match.end() + 1]
    if not following:
        return False  # trailing whitespace, nothing follows
    if not end.startswith(".") or len(end.rstrip("\"'”’»)]")) > 1:
        return True  # "!", "?", "…", "?!", "..." always end one
    # A single period: look at the word it closes.
    word_match = _LAST_WORD_RE.search(
        text, max(0, match.start() - 32), match.start()
    )
    word = word_match.group(0) if word_match else ""
    lowered = word.lower()
    if lowered in _SOFT_ABBREVIATIONS:
        return following.isupper()
    if lowered in _ABBREVIATIONS or _DOTTED_ACRONYM_RE.match(word):
        return False
    # An initial ("J. K. Rowling", "U. S."): one capital letter.
    if len(word) == 1 and word.isalpha() and word.isupper():
        return False
    # A list number at the start of a line ("1. First item").
    if word.isdigit() and len(word) <= 3 and word_match is not None:
        preceding = text[: word_match.start()].rstrip(" \t")
        if not preceding or preceding.endswith("\n"):
            return False
    return True


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences.

    A sentence ends at ``.``, ``!``, ``?`` or ``…`` (plus any closing
    quote / bracket) followed by whitespace, or at a CJK full stop
    (``。！？``) with or without whitespace. The next sentence may
    start with anything — an uppercase or lowercase letter in any
    script, a digit, a quote or a bracket — so text that isn't
    capitalised English prose still splits. Periods after common
    abbreviations (``Dr.``, ``e.g.``, ``Fig.``) and single-letter
    initials don't split.
    """
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        if not _is_sentence_break(text, match):
            continue
        boundary = match.end("end") if match.group("end") is not None else match.end("cjk")
        sentence = text[start:boundary].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


class SentenceChunker:
    """Sentence-boundary chunks.

    Splits into sentences with :func:`split_sentences` — terminators
    (``.``, ``!``, ``?``, ``…``, CJK ``。！？``) followed by whitespace,
    whatever the next sentence starts with (any script, any case,
    digits, quotes, brackets), skipping common abbreviations and
    initials. Greedily packs sentences up to ``chunk_size``
    characters. Each new chunk starts with the previous chunk's last
    whole sentences, as many as fit in ``chunk_overlap`` characters —
    possibly none, when the last sentence is longer than that.

    Best for QA-style RAG where each chunk should answer one short
    question.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, *, source: str = "") -> list[Chunk]:
        if not text:
            return []
        sentences = split_sentences(text)
        if not sentences:
            return []
        # A sentence longer than a chunk is split at word (then
        # character) boundaries instead of becoming an oversized chunk.
        splitter = RecursiveChunker(self.chunk_size, 0, (" ", ""))
        sentences = [
            piece
            for sentence in sentences
            for piece in (
                [sentence]
                if len(sentence) <= self.chunk_size
                else [c.content for c in splitter.split(sentence)]
            )
        ]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for joining space
            if current_len + len(sentence) > self.chunk_size and current:
                content = " ".join(current).strip()
                chunks.append(
                    Chunk(
                        content=content,
                        metadata={
                            "source": source,
                            "index": len(chunks),
                            "chunk_size": len(content),
                            "strategy": "sentence",
                        },
                    )
                )
                # Build overlap: keep tail sentences whose total
                # length is just under chunk_overlap.
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if (
                        overlap_len + len(s) + 1 > self.chunk_overlap
                        or overlap_len + len(s) + 1 + len(sentence) > self.chunk_size
                    ):
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                current = overlap_sentences
                current_len = overlap_len
            current.append(sentence)
            current_len += sentence_len

        if current:
            content = " ".join(current).strip()
            chunks.append(
                Chunk(
                    content=content,
                    metadata={
                        "source": source,
                        "index": len(chunks),
                        "chunk_size": len(content),
                        "strategy": "sentence",
                    },
                )
            )
        return chunks


# ---------------------------------------------------------------------------
# TokenChunker
# ---------------------------------------------------------------------------


class TokenChunker:
    """Chunk by exact token count using ``tiktoken``.

    Each chunk is at most ``chunk_size`` TOKENS (not characters)
    with ``chunk_overlap`` tokens of overlap. Use this when you
    need tight control over context-window fit (embedding models
    have hard token limits — text-embedding-3-large is 8191).

    Requires ``tiktoken``: ``pip install 'loomflow[loader]'``.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding: str = "cl100k_base",  # GPT-4 / 4o / 4.1 default
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding

    def split(self, text: str, *, source: str = "") -> list[Chunk]:
        if not text:
            return []
        try:
            import tiktoken  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "tiktoken is required for TokenChunker. "
                "Install with: pip install 'loomflow[loader]'"
            ) from exc

        enc = tiktoken.get_encoding(self.encoding_name)
        token_ids = enc.encode(text)
        if not token_ids:
            return []

        # Cut on character boundaries: decoding a token slice that
        # splits a multi-byte character produced U+FFFD. Each token's
        # character offset maps a token window to a slice of the text
        # (a character shared by two tokens goes to the later chunk
        # too, never half of it).
        decoded, offsets = enc.decode_with_offsets(token_ids)
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(token_ids), step):
            piece_ids = token_ids[i : i + self.chunk_size]
            end = i + self.chunk_size
            content = decoded[offsets[i] : offsets[end] if end < len(offsets) else len(decoded)]
            chunks.append(
                Chunk(
                    content=content,
                    metadata={
                        "source": source,
                        "index": len(chunks),
                        "chunk_size": len(content),
                        "token_count": len(piece_ids),
                        "encoding": self.encoding_name,
                        "strategy": "token",
                    },
                )
            )
            if i + self.chunk_size >= len(token_ids):
                break
        return chunks


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def chunk(
    text: str,
    *,
    strategy: str = "recursive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source: str = "",
    # Strategy-specific options (only used by their named strategy;
    # exposed explicitly so IDEs surface them in autocomplete instead
    # of hiding them behind ``**kwargs``).
    separators: Sequence[str] | None = None,  # recursive
    encoding: str = "cl100k_base",            # token
) -> list[Chunk]:
    """One-liner chunking: pick a strategy by name and split.

    * ``strategy="recursive"`` (default) — char-level recursive
      split. Honours ``separators`` (list of separator strings,
      tried in order). Default separators: paragraphs, sentences,
      words, characters.
    * ``strategy="markdown"`` — splits on heading boundaries;
      preserves the header trail in each chunk's metadata.
    * ``strategy="sentence"`` — splits on sentence boundaries.
    * ``strategy="token"`` — chunks by exact token count via
      ``tiktoken`` (requires the ``loader-token`` extra). Honours
      ``encoding`` (default ``"cl100k_base"`` for GPT-4 / 4o / 4.1).

    Strategy-specific kwargs are silently ignored when not
    applicable (e.g. ``encoding=`` is harmless if you use
    ``strategy="markdown"``).
    """
    chunker: Chunker
    if strategy == "recursive":
        chunker = RecursiveChunker(
            chunk_size,
            chunk_overlap,
            separators if separators is not None else _DEFAULT_SEPARATORS,
        )
    elif strategy == "markdown":
        chunker = MarkdownChunker(chunk_size, chunk_overlap)
    elif strategy == "sentence":
        chunker = SentenceChunker(chunk_size, chunk_overlap)
    elif strategy == "token":
        chunker = TokenChunker(chunk_size, chunk_overlap, encoding)
    else:
        raise ValueError(
            f"unknown strategy {strategy!r}; "
            "expected: recursive, markdown, sentence, token"
        )
    return chunker.split(text, source=source)
