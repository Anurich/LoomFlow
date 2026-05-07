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

    from jeevesagent.loader import chunk

    pieces = chunk(text, strategy="recursive", chunk_size=800)
    pieces = chunk(text, strategy="markdown")
    pieces = chunk(text, strategy="sentence", chunk_size=400)
    pieces = chunk(text, strategy="token", chunk_size=512)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

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
        pieces = self._recursive_split(text, self.separators)
        merged = self._merge_pieces(pieces)
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
        """Split ``text`` using the first separator that fits.

        Returns a flat list of strings, each <= ``chunk_size`` (when
        possible). When even the finest-grained separator can't get
        a piece under the limit, that piece is hard-cut.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Find the first separator that actually appears in text.
        for i, sep in enumerate(separators):
            if sep == "":
                # Hard-cut on character boundaries, last resort.
                return [
                    text[j : j + self.chunk_size]
                    for j in range(0, len(text), self.chunk_size)
                ]
            if sep in text:
                # Split on this separator; recurse on each piece
                # using the remaining (finer-grained) separators.
                sub = text.split(sep)
                results: list[str] = []
                remaining = separators[i + 1 :]
                for j, part in enumerate(sub):
                    # Re-attach the separator to all but the last
                    # piece so the join is reversible.
                    fragment = part + (sep if j < len(sub) - 1 else "")
                    if len(fragment) <= self.chunk_size:
                        results.append(fragment)
                    else:
                        results.extend(
                            self._recursive_split(fragment, remaining)
                        )
                return results

        return [text]  # shouldn't reach here (sep="" fallback exists)

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Greedily merge adjacent pieces back up to ``chunk_size``,
        adding ``chunk_overlap`` chars of overlap between chunks."""
        if not pieces:
            return []
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)
                # Start the next chunk with the tail of the previous
                # one (overlap), then add the new piece.
                if self.chunk_overlap > 0 and chunks:
                    overlap = chunks[-1][-self.chunk_overlap :]
                    current = overlap + piece
                else:
                    current = piece
                # If a single piece is already over chunk_size,
                # just keep it as-is — recursive_split should have
                # prevented this.
                if len(current) > self.chunk_size and not chunks:
                    chunks.append(current)
                    current = ""
        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# MarkdownChunker — heading-aware
# ---------------------------------------------------------------------------


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


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
        # Find all heading positions
        heading_matches = list(_HEADING_RE.finditer(text))
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

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for joining space
            if current_len + sentence_len > self.chunk_size and current:
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
                    if overlap_len + len(s) + 1 > self.chunk_overlap:
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

    Requires ``tiktoken``: ``pip install 'jeevesagent[loader]'``.
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
                "Install with: pip install 'jeevesagent[loader]'"
            ) from exc

        enc = tiktoken.get_encoding(self.encoding_name)
        token_ids = enc.encode(text)
        if not token_ids:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(token_ids), step):
            piece_ids = token_ids[i : i + self.chunk_size]
            content = enc.decode(piece_ids)
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
    **kwargs: Any,
) -> list[Chunk]:
    """One-liner chunking: pick a strategy by name and split.

    ``strategy`` is one of ``"recursive"`` (default), ``"markdown"``,
    ``"sentence"``, ``"token"``. ``kwargs`` pass through to the
    chunker's constructor.
    """
    chunker: Chunker
    if strategy == "recursive":
        chunker = RecursiveChunker(chunk_size, chunk_overlap, **kwargs)
    elif strategy == "markdown":
        chunker = MarkdownChunker(chunk_size, chunk_overlap)
    elif strategy == "sentence":
        chunker = SentenceChunker(chunk_size, chunk_overlap)
    elif strategy == "token":
        chunker = TokenChunker(chunk_size, chunk_overlap, **kwargs)
    else:
        raise ValueError(
            f"unknown strategy {strategy!r}; "
            "expected: recursive, markdown, sentence, token"
        )
    return chunker.split(text, source=source)
