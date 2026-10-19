from __future__ import annotations

import io
from pathlib import Path

from src.rag.ingest import load_documents, read_pdf


def make_pdf(text: str) -> bytes:
    """Smallest possible one-page PDF with a text content stream (no external deps)."""
    content = f"BT /F1 14 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def test_pdf_text_is_extracted(tmp_path: Path):
    pdf = tmp_path / "billing.pdf"
    pdf.write_bytes(make_pdf("Late fee is 7 dollars."))
    assert "Late fee is 7 dollars." in read_pdf(pdf)


def test_pdf_documents_are_loaded_into_the_folder_scan(tmp_path: Path):
    pdf = tmp_path / "billing.pdf"
    pdf.write_bytes(make_pdf("Refunds take 5 to 10 business days."))
    docs = load_documents(tmp_path)
    assert [d.source for d in docs] == ["billing.pdf"]
    assert "Refunds take 5 to 10 business days." in docs[0].text
