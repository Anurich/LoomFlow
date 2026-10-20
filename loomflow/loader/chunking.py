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

    Aims for chunks of ``chunk_size`` characters with
    ``chunk_overlap`` chars of overlap. Splits on a hierarchy of
    separators (paragraph → line → sentence → word → char), trying
    to preserve semantic boundaries.

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
        finer separators. Pieces keep their trailing separator, so
        joining a chunk's pieces reproduces the text exactly.
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
            else:  # nothing finer to split on: keep it whole
                chunks.append(piece)
        chunks.extend(self._merge_pieces(small))
        return chunks

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Merge pieces into chunks of at most ``chunk_size``; each new
        chunk starts with the previous one's last pieces, up to
        ``chunk_overlap`` characters (whole pieces, never mid-word)."""
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
        sections = self._split_into_sections(text)
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


_SENTENCE_RE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z])")


class SentenceChunker:
    """Sentence-boundary chunks.

    Splits on sentence terminators (``.``, ``!``, ``?``) followed by
    whitespace and a capital letter. Greedily packs sentences up to
    ``chunk_size`` characters; adds ``chunk_overlap`` chars between
    chunks (rounded to the nearest sentence boundary).

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
        sentences = [
            s.strip() for s in _SENTENCE_RE.split(text) if s.strip()
        ]
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
