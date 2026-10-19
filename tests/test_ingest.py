from __future__ import annotations

from pathlib import Path

from src.rag.ingest import Chunk, chunk_text, load_documents


def test_heading_aware_chunks():
    text = (
        "# Plans\n\nSmart Light costs $19.\n\n## Add-ons\n\nExtra data is $8.\n\n"
        "## Support\n\nCall us Monday to Saturday.\n"
    )
    chunks = chunk_text(text, "plans.md")
    assert chunks, "expected chunks"
    headings = {c.heading for c in chunks}
    assert "Plans" in headings
    assert "Add-ons" in headings
    assert any("$8" in c.text for c in chunks)
    assert all(len(c.text) <= 700 for c in chunks)


def test_long_paragraph_is_split_on_boundaries():
    paragraph = ("word " * 260).strip()
    chunks = chunk_text(paragraph, "long.txt", max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    rebuilt = " ".join(c.text for c in chunks)
    assert rebuilt.split().count("word") == 260


def test_short_document_still_yields_chunk():
    chunks = chunk_text("Just one short line.", "tiny.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "tiny.txt"


def test_markdown_and_txt_loaded(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nHello A.\n")
    (docs / "b.txt").write_text("TITLE\n\nHello B.\n")
    (docs / "ignored.bin").write_bytes(b"\x00\x01")
    loaded = load_documents(docs)
    assert {d.source for d in loaded} == {"a.md", "b.txt"}


def test_chunk_ids_are_stable():
    one = Chunk.make("hello", "a.md", "H")
    two = Chunk.make("hello", "a.md", "H")
    assert one.chunk_id == two.chunk_id
    assert len(one.chunk_id) == 12
