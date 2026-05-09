"""Tests for chunking strategies."""

from __future__ import annotations

import pytest

from loomflow.loader import (
    Chunk,
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
    chunk,
)

# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------


def test_recursive_chunker_short_text_one_chunk() -> None:
    text = "short text"
    out = RecursiveChunker(chunk_size=100, chunk_overlap=0).split(text)
    assert len(out) == 1
    assert out[0].content == text


def test_recursive_chunker_splits_by_paragraph_first() -> None:
    text = (
        "Paragraph one. " * 30
        + "\n\n"
        + "Paragraph two. " * 30
    )
    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=0)
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    # Each chunk fits the limit.
    for c in chunks:
        assert len(c.content) <= 400


def test_recursive_chunker_respects_chunk_size() -> None:
    text = "a" * 2000
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
    chunks = chunker.split(text)
    for c in chunks:
        assert len(c.content) <= 500


def test_recursive_chunker_overlap_applied() -> None:
    """Adjacent chunks should share ~chunk_overlap chars at the boundary."""
    text = "alpha bravo charlie delta echo foxtrot " * 50  # repeated
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.split(text)
    if len(chunks) >= 2:
        first_tail = chunks[0].content[-50:]
        second_head = chunks[1].content[:50]
        # Some overlap should be visible (might not be exactly 50 chars
        # depending on word boundaries)
        common = sum(
            1 for a, b in zip(first_tail, second_head, strict=False) if a == b
        )
        assert common > 0


def test_recursive_chunker_metadata() -> None:
    chunks = RecursiveChunker(chunk_size=100, chunk_overlap=0).split(
        "hello world", source="test.md"
    )
    assert chunks[0].metadata["source"] == "test.md"
    assert chunks[0].metadata["index"] == 0
    assert chunks[0].metadata["strategy"] == "recursive"


def test_recursive_chunker_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=200)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=-1)


def test_recursive_chunker_empty_text() -> None:
    assert RecursiveChunker().split("") == []


# ---------------------------------------------------------------------------
# MarkdownChunker
# ---------------------------------------------------------------------------


def test_markdown_chunker_splits_by_heading() -> None:
    text = (
        "# Title\n\nIntro paragraph.\n\n"
        "## Section A\n\nContent A.\n\n"
        "## Section B\n\nContent B.\n\n"
        "### Subsection B1\n\nContent B1.\n"
    )
    chunks = MarkdownChunker(chunk_size=2000).split(text)
    # One chunk per heading section.
    sections_started = [
        c for c in chunks if c.metadata.get("headers")
    ]
    titles = [c.metadata["headers"][0] for c in sections_started]
    assert "Title" in titles
    assert any("Section A" in str(c.metadata["headers"]) for c in chunks)
    assert any("Section B" in str(c.metadata["headers"]) for c in chunks)


def test_markdown_chunker_preserves_header_trail() -> None:
    text = (
        "# Doc\n\n"
        "## Chapter 1\n\n"
        "### Section 1.1\n\nContent here.\n"
    )
    chunks = MarkdownChunker(chunk_size=2000).split(text)
    last = chunks[-1]
    headers = last.metadata.get("headers", [])
    assert "Doc" in headers
    assert "Chapter 1" in headers
    assert "Section 1.1" in headers


def test_markdown_chunker_falls_back_for_long_sections() -> None:
    """A single section longer than chunk_size should be split via
    the recursive fallback, with header trail preserved on each piece."""
    long_para = "word " * 500
    text = f"# Title\n\n## Section A\n\n{long_para}"
    chunks = MarkdownChunker(chunk_size=400).split(text)
    section_a_chunks = [
        c for c in chunks
        if "Section A" in c.metadata.get("headers", [])
    ]
    assert len(section_a_chunks) > 1
    # All sub-chunks should still know their parent headers
    for c in section_a_chunks:
        assert "Section A" in c.metadata["headers"]


def test_markdown_chunker_no_headings_one_chunk() -> None:
    """Plain text with no headings should yield one chunk."""
    text = "Just some prose with no headings."
    chunks = MarkdownChunker(chunk_size=2000).split(text)
    assert len(chunks) == 1
    assert chunks[0].metadata.get("headers") == []


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------


def test_sentence_chunker_splits_by_sentence_boundary() -> None:
    text = (
        "First sentence. Second sentence. Third sentence. "
        "Fourth sentence. Fifth sentence. Sixth sentence."
    )
    chunks = SentenceChunker(chunk_size=40, chunk_overlap=0).split(text)
    # Each chunk should contain whole sentences.
    for c in chunks:
        assert len(c.content) <= 80  # sentence boundaries can overshoot a bit


def test_sentence_chunker_metadata() -> None:
    chunks = SentenceChunker(chunk_size=100, chunk_overlap=0).split(
        "Hello world. Foo bar.", source="x.md"
    )
    assert chunks[0].metadata["source"] == "x.md"
    assert chunks[0].metadata["strategy"] == "sentence"


# ---------------------------------------------------------------------------
# TokenChunker
# ---------------------------------------------------------------------------


def test_token_chunker_respects_token_count() -> None:
    text = "hello world " * 200
    chunks = TokenChunker(chunk_size=20, chunk_overlap=0).split(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["token_count"] <= 20


def test_token_chunker_empty_text() -> None:
    assert TokenChunker().split("") == []


def test_token_chunker_metadata_includes_encoding() -> None:
    chunks = TokenChunker(chunk_size=100).split("hello")
    if chunks:
        assert chunks[0].metadata["encoding"] == "cl100k_base"
        assert chunks[0].metadata["strategy"] == "token"


# ---------------------------------------------------------------------------
# chunk() factory
# ---------------------------------------------------------------------------


def test_chunk_factory_recursive() -> None:
    out = chunk("hello world " * 100, strategy="recursive", chunk_size=200)
    assert all(isinstance(c, Chunk) for c in out)
    assert all(c.metadata["strategy"] == "recursive" for c in out)


def test_chunk_factory_markdown() -> None:
    out = chunk("# A\n\nfoo\n\n## B\n\nbar", strategy="markdown")
    assert all(c.metadata["strategy"].startswith("markdown") for c in out)


def test_chunk_factory_sentence() -> None:
    out = chunk(
        "Hello. World. Foo. Bar.",
        strategy="sentence",
        chunk_size=40,
        chunk_overlap=0,
    )
    assert all(c.metadata["strategy"] == "sentence" for c in out)


def test_chunk_factory_token() -> None:
    out = chunk(
        "hello world " * 50,
        strategy="token",
        chunk_size=20,
        chunk_overlap=0,
    )
    assert all(c.metadata["strategy"] == "token" for c in out)


def test_chunk_factory_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk("text", strategy="bogus")


def test_chunk_factory_propagates_source() -> None:
    out = chunk("hello", strategy="recursive", source="test.md")
    assert out[0].metadata["source"] == "test.md"
