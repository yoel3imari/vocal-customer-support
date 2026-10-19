"""Doc-folder ingestion: markdown / text / PDF → heading-aware chunks."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("voxdesk.ingest")

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
MAX_CHARS = 700


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    heading: str = ""
    chunk_id: str = ""

    @classmethod
    def make(cls, text: str, source: str, heading: str = "") -> "Chunk":
        digest = hashlib.sha1(f"{source}:{heading}:{text}".encode()).hexdigest()[:12]
        return cls(text=text, source=source, heading=heading, chunk_id=digest)


@dataclass
class Document:
    text: str
    source: str


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - corrupt page shouldn't kill the call
            log.warning("failed to extract %s page: %s", path.name, exc)
    return "\n\n".join(pages)


def load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.is_dir():
        return []
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", path, exc)
            continue
        if text.strip():
            docs.append(Document(text=text, source=str(path.relative_to(docs_dir))))
    return docs


def _split_long(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "), window.rfind(" "))
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_text(text: str, source: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Split a document by headings/paragraphs, packing text into ~700-char chunks."""
    chunks: list[Chunk] = []
    heading = ""
    section: list[str] = []

    def flush_section() -> None:
        body = "\n\n".join(p.strip() for p in section if p.strip()).strip()
        section.clear()
        if not body:
            return
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf = ""
        for para in paras:
            for piece in _split_long(para, max_chars):
                candidate = f"{buf}\n\n{piece}".strip() if buf else piece
                if len(candidate) <= max_chars:
                    buf = candidate
                else:
                    if buf:
                        chunks.append(Chunk.make(buf, source, heading))
                    buf = piece
        if buf:
            chunks.append(Chunk.make(buf, source, heading))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if re.match(r"^#{1,3}\s+", line):
            flush_section()
            heading = re.sub(r"^#{1,3}\s+", "", line).strip()
            continue
        if line.upper() == line and len(line) > 6 and not line.strip().endswith(".") and line.strip():
            # plain-text ALL-CAPS section label
            flush_section()
            heading = line.strip().title()
            continue
        section.append(line)
    flush_section()
    if not chunks and text.strip():
        chunks = [Chunk.make(text.strip()[:max_chars], source)]
    return chunks


def build_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(docs_dir):
        chunks.extend(chunk_text(doc.text, doc.source))
    # Heading-aware chunk text is what gets embedded; keep the source for citations.
    log.info("ingested %d chunks from %s", len(chunks), docs_dir)
    return chunks
