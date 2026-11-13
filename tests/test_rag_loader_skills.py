"""Regression tests for the loaders, chunkers and skill scripts:
HTML structure inside containers, unicode-aware sentence splitting,
heading-only markdown chunks, the chunk-size bound, text encodings,
and skill scripts that point outside their skill folder.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from loomflow.loader import load
from loomflow.loader.chunking import (
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    chunk,
    split_sentences,
)
from loomflow.loader.csv import load_csv, load_tsv
from loomflow.loader.text import load_markdown, load_text
from loomflow.skills import Skill
from loomflow.skills.skill import SkillError

# ---------------------------------------------------------------------------
# 8. HTML: containers keep their structure
# ---------------------------------------------------------------------------


def _html(tmp_path: Path, body: str, name: str = "p.html") -> str:
    pytest.importorskip("bs4")
    path = tmp_path / name
    path.write_text(
        f"<html><head><title>T</title></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    return load(path).content


def test_html_div_keeps_headings_lists_and_paragraphs(tmp_path: Path) -> None:
    content = _html(
        tmp_path,
        "<div id='root'><h1>Head</h1><script>var secret=1;</script>"
        "<style>.a{color:red}</style><ul><li>one</li><li>two</li></ul>"
        "<p>para</p></div>",
    )
    assert content == "# T\n\n# Head\n\n- one\n- two\n\npara\n"
    assert "secret" not in content and "color" not in content


def test_html_nested_containers_and_tables(tmp_path: Path) -> None:
    content = _html(
        tmp_path,
        "<main><article><section><h2>Prices</h2>"
        "<table><tr><th>Item</th><th>Cost</th></tr>"
        "<tr><td>Tea</td><td>3</td></tr></table>"
        "<div><ol><li>first</li><li>second</li></ol></div>"
        "</section></article></main>",
    )
    assert "## Prices" in content
    assert "| Item | Cost |\n| --- | --- |\n| Tea | 3 |" in content
    assert "1. first\n2. second" in content


def test_html_inline_text_between_blocks_becomes_paragraphs(tmp_path: Path) -> None:
    content = _html(
        tmp_path,
        "<div>Intro <b>bold</b> text<!-- hidden note --><h3>Sub</h3>"
        "tail <a href='#'>link</a></div>",
    )
    assert "Intro bold text" in content
    assert "### Sub" in content
    assert "tail link" in content
    assert "hidden note" not in content
    assert content.index("Intro") < content.index("### Sub") < content.index("tail")


def test_html_plain_text_container_is_unchanged(tmp_path: Path) -> None:
    # A container with only inline content is still one paragraph.
    assert _html(tmp_path, "<div>Just <i>some</i> text</div>") == (
        "# T\n\nJust some text\n"
    )


def test_html_nested_lists_are_indented(tmp_path: Path) -> None:
    content = _html(
        tmp_path,
        "<section><ul><li>fruit<ul><li>apple</li><li>pear</li></ul></li>"
        "<li>veg</li></ul></section>",
    )
    assert "- fruit\n  - apple\n  - pear\n- veg" in content


def test_html_strips_bom(tmp_path: Path) -> None:
    pytest.importorskip("bs4")
    path = tmp_path / "bom.html"
    path.write_bytes("\ufeff<html><body><p>hi</p></body></html>".encode())
    assert "\ufeff" not in load(path).content


# ---------------------------------------------------------------------------
# 9. Sentence splitting
# ---------------------------------------------------------------------------


def test_sentences_split_without_ascii_capitals() -> None:
    text = "это первое предложение. это второе предложение. это третье предложение."
    assert split_sentences(text) == [
        "это первое предложение.",
        "это второе предложение.",
        "это третье предложение.",
    ]


def test_sentences_start_with_digits_quotes_parens_and_accents() -> None:
    text = 'He paid $5. 3 more came. "Quoted start." (Paren start.) élan vital. Ünited.'
    assert split_sentences(text) == [
        "He paid $5.",
        "3 more came.",
        '"Quoted start."',
        "(Paren start.)",
        "élan vital.",
        "Ünited.",
    ]


def test_sentences_handle_cjk_and_mixed_terminators() -> None:
    assert split_sentences("这是第一句。这是第二句！好吗？") == [
        "这是第一句。",
        "这是第二句！",
        "好吗？",
    ]
    assert split_sentences("Really?! Yes... ok. Fine… done") == [
        "Really?!",
        "Yes...",
        "ok.",
        "Fine…",
        "done",
    ]


def test_sentences_do_not_split_on_abbreviations_or_decimals() -> None:
    text = (
        "Dr. Smith met J. K. Rowling at 5 p.m. in the U.S. capital, e.g. near "
        "Fig. 3. Pi is 3.14 exactly. Apples, pears, etc. The rest came later."
    )
    assert split_sentences(text) == [
        "Dr. Smith met J. K. Rowling at 5 p.m. in the U.S. capital, e.g. near Fig. 3.",
        "Pi is 3.14 exactly.",
        "Apples, pears, etc.",
        "The rest came later.",
    ]
    assert split_sentences("1. First item.\n2. Second item.") == [
        "1. First item.",
        "2. Second item.",
    ]


def test_sentence_chunker_packs_lowercase_sentences() -> None:
    text = "это первое предложение. это второе предложение. это третье предложение. " * 5
    chunks = SentenceChunker(chunk_size=60, chunk_overlap=0).split(text)
    # 15 sentences of ~23 chars: two per 60-char chunk, not one giant
    # "sentence" hard-cut at arbitrary words.
    assert len(chunks) == 8
    assert all(c.content.endswith(".") for c in chunks)
    assert all(len(c.content) <= 60 for c in chunks)


# ---------------------------------------------------------------------------
# 9. Markdown heading-only chunks
# ---------------------------------------------------------------------------


def test_markdown_heading_only_sections_merge_into_following_content() -> None:
    chunks = MarkdownChunker().split("# Title\n\n## A\n\n## B\ntext\n")
    assert [c.content for c in chunks] == ["# Title\n\n## A\n\n## B\ntext\n"]
    assert chunks[0].metadata["headers"] == ["Title", "B"]

    chunks = MarkdownChunker().split(
        "intro\n\n# Guide\n\n## Install\n\nrun pip\n\n## Use\n\ncall it\n"
    )
    assert [c.content for c in chunks] == [
        "intro\n\n",
        "# Guide\n\n## Install\n\nrun pip\n\n",
        "## Use\n\ncall it\n",
    ]
    assert [c.metadata["headers"] for c in chunks] == [
        [],
        ["Guide", "Install"],
        ["Guide", "Use"],
    ]
    for c in chunks:
        body = [ln for ln in c.content.splitlines() if ln and not ln.startswith("#")]
        assert body, f"heading-only chunk: {c.content!r}"


def test_markdown_trailing_heading_is_not_dropped() -> None:
    chunks = MarkdownChunker().split("# A\n\nbody\n\n## Appendix\n")
    assert "".join(c.content for c in chunks) == "# A\n\nbody\n\n## Appendix\n"


def test_markdown_merged_heading_respects_chunk_size() -> None:
    text = "# Title\n\n## Section\n\n" + "word " * 200
    chunks = MarkdownChunker(chunk_size=120, chunk_overlap=0).split(text)
    assert all(len(c.content) <= 120 for c in chunks)
    assert chunks[0].content.startswith("# Title")
    assert all(c.metadata["headers"] == ["Title", "Section"] for c in chunks)


# ---------------------------------------------------------------------------
# 9. chunk_size is a hard bound
# ---------------------------------------------------------------------------


def test_custom_separators_cannot_exceed_chunk_size() -> None:
    chunks = RecursiveChunker(50, 0, separators=["\n\n", " "]).split("a" * 500)
    assert [len(c.content) for c in chunks] == [50] * 10
    assert "".join(c.content for c in chunks) == "a" * 500

    mixed = "short words then " + "x" * 130 + " and more words"
    for piece in chunk(mixed, chunk_size=40, chunk_overlap=5, separators=[" "]):
        assert len(piece.content) <= 40

    # A separator that never occurs in the text.
    lone = RecursiveChunker(30, 0, separators=["|"]).split("b" * 95)
    assert [len(c.content) for c in lone] == [30, 30, 30, 5]


def test_chunkers_never_exceed_chunk_size_fuzz() -> None:
    rng = random.Random(1)
    words = ["alpha", "beta", "gamma", "délta", "日本語", "x" * 30, "e.", "Zed!", "q?", "\n\n", "\n"]
    separators = [None, ["\n\n", " "], ["\n"], [". ", " "]]
    for _ in range(150):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 300)))
        size = rng.randint(5, 200)
        overlap = rng.randint(0, size - 1)
        chunkers = [SentenceChunker(size, overlap), MarkdownChunker(size, overlap)]
        for seps in separators:
            chunkers.append(
                RecursiveChunker(size, overlap)
                if seps is None
                else RecursiveChunker(size, overlap, separators=seps)
            )
        for chunker in chunkers:
            for piece in chunker.split(text):
                assert len(piece.content) <= size, (type(chunker).__name__, size)


def test_overlap_is_whole_pieces_as_documented() -> None:
    # Words: overlap is close to chunk_overlap, never more.
    words = " ".join(f"w{i:03d}" for i in range(200))
    pieces = [c.content for c in RecursiveChunker(100, 20).split(words)]
    for previous, current in zip(pieces, pieces[1:], strict=False):
        shared = next(
            n for n in range(min(len(previous), len(current)), -1, -1)
            if previous.endswith(current[:n])
        )
        assert 0 < shared <= 20
    # Paragraphs longer than chunk_overlap: no overlap at all — which is
    # what the docstring now says.
    paragraphs = "\n\n".join(f"Sentence number {i} is here. " * 6 for i in range(8))
    pieces = [c.content for c in RecursiveChunker(400, 50).split(paragraphs)]
    assert len(pieces) > 1
    for previous, current in zip(pieces, pieces[1:], strict=False):
        assert not previous.endswith(current[:20])
    assert "whole pieces" in (RecursiveChunker.__doc__ or "")


# ---------------------------------------------------------------------------
# 11. Text / CSV encodings
# ---------------------------------------------------------------------------


def test_text_and_markdown_strip_the_bom(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_bytes("\ufeffcafé".encode())
    md = tmp_path / "a.md"
    md.write_bytes("\ufeff# Title\n".encode())
    assert load(txt).content == "# a.txt\n\ncafé"
    assert load(md).content == "# Title\n"
    assert load_text(txt, encoding="utf-8").content == "# a.txt\n\ncafé"


def test_csv_header_has_no_bom(tmp_path: Path) -> None:
    csv_path = tmp_path / "c.csv"
    csv_path.write_bytes("\ufeffname,v\r\nx,1\r\n".encode())
    doc = load(csv_path)
    assert doc.content.splitlines()[2] == "| name | v |"
    assert "\ufeff" not in doc.content
    tsv_path = tmp_path / "c.tsv"
    tsv_path.write_bytes("\ufeffname\tv\nx\t1\n".encode())
    assert load_tsv(tsv_path).content.splitlines()[2] == "| name | v |"


def test_csv_quoted_newlines_still_parse(tmp_path: Path) -> None:
    path = tmp_path / "q.csv"
    path.write_bytes(b'a,b\r\n"line one\r\nline two",2\r\n')
    doc = load_csv(path)
    assert doc.metadata["row_count"] == 1
    assert "line one" in doc.content and "line two" in doc.content


def test_decode_failure_names_the_encoding_option(tmp_path: Path) -> None:
    latin = "café".encode("latin-1")
    for name, loader in (
        ("b.txt", load_text),
        ("b.md", load_markdown),
        ("b.csv", load_csv),
        ("b.tsv", load_tsv),
    ):
        path = tmp_path / name
        path.write_bytes(latin)
        with pytest.raises(UnicodeDecodeError) as excinfo:
            loader(path)
        assert "encoding=" in str(excinfo.value)
        assert "errors='replace'" in str(excinfo.value)
        with pytest.raises(UnicodeDecodeError, match="encoding="):
            load(path)
        assert "café" in loader(path, encoding="latin-1").content
        assert "café" in load(path, encoding="latin-1").content
        assert "caf\ufffd" in loader(path, errors="replace").content
        assert "caf\ufffd" in load(path, errors="replace").content


def test_load_ignores_encoding_for_binary_formats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported extension"):
        load(tmp_path / "x.bin", encoding="latin-1")


# ---------------------------------------------------------------------------
# 13. Skill scripts must stay inside the skill folder
# ---------------------------------------------------------------------------


def _skill(tmp_path: Path, script: str, name: str = "evil") -> Path:
    root = tmp_path / "skills" / name
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "ok.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: test skill\n"
        "tools:\n"
        "  run:\n"
        "    description: runs\n"
        f"    script: {script}\n"
        "    args:\n"
        "      x:\n"
        "        type: string\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    "script",
    [
        "../../../../../../../../../../bin/echo",
        "../other/run.sh",
        "/bin/echo",
        "scripts/../../evil/../../outside.py",
        ".",
    ],
)
def test_skill_script_outside_the_skill_folder_is_rejected(
    tmp_path: Path, script: str
) -> None:
    root = _skill(tmp_path, script)
    with pytest.raises(SkillError, match="skill directory"):
        Skill(root)


def test_skill_script_symlink_out_of_the_folder_is_rejected(tmp_path: Path) -> None:
    root = _skill(tmp_path, "scripts/link.py")
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    try:
        (root / "scripts" / "link.py").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(SkillError, match="outside the skill directory"):
        Skill(root)


def test_skill_script_inside_the_folder_still_loads(tmp_path: Path) -> None:
    root = _skill(tmp_path, "scripts/ok.py", name="good")
    tools = Skill(root)._pending_tools
    assert [t.name for t in tools] == ["good__run"]
    assert Path(tools[0].fn.script) == (root / "scripts" / "ok.py").resolve()
