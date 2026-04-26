"""Extract raw text from PDF or DOCX files."""

from __future__ import annotations

import os
from pathlib import Path

from pypdf import PdfReader
from docx import Document

PAGE_BREAK = "\n\f\n"


def extract_text(file_path: str) -> str:
    """Read a PDF or DOCX and return its text.

    PDF pages are joined with a form-feed marker so downstream cleaning can
    detect repeated headers/footers. DOCX paragraphs are joined with newlines.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    raise ValueError(f"Unsupported file extension: {ext!r} (expected .pdf or .docx)")


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return PAGE_BREAK.join(pages)


def _extract_docx(path: Path) -> str:
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)
