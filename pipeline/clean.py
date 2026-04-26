"""
Clean extracted text: normalize unicode, preserve structure,
remove artifacts, and prepare for chunking.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from pipeline.extract import PAGE_BREAK


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_INLINE_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?\s*$",
    re.IGNORECASE
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")


# -------------------------
# MAIN CLEAN FUNCTION
# -------------------------

def clean_text(raw: str) -> str:
    """Normalize and de-noise extracted text while preserving structure."""
    if not raw:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", raw)

    # 2. Remove control characters
    text = _CONTROL_CHARS.sub("", text)

    # 3. Split by pages (if exists)
    pages = text.split(PAGE_BREAK) if PAGE_BREAK in text else [text]

    # 4. Remove repeated headers/footers
    pages = _drop_repeated_headers_footers(pages)

    # 5. Clean each page
    pages = [_clean_page(p) for p in pages]

    # 6. Rebuild document
    text = "\n\n".join(p for p in pages if p.strip())

    # 7. Split into paragraphs (structure preservation step)
    paragraphs = _PARAGRAPH_SPLIT.split(text)

    cleaned_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # normalize internal whitespace WITHOUT destroying structure
        p = _normalize_whitespace_safe(p)

        cleaned_paragraphs.append(p)

    return "\n\n".join(cleaned_paragraphs).strip()


# -------------------------
# PAGE CLEANING
# -------------------------

def _clean_page(page: str) -> str:
    lines = []

    for line in page.splitlines():
        stripped = line.strip()

        if not stripped:
            lines.append("")
            continue

        # remove page numbers
        if _PAGE_NUMBER_LINE.match(stripped):
            continue

        lines.append(stripped)

    return "\n".join(lines)


# -------------------------
# HEADER / FOOTER REMOVAL
# -------------------------

def _drop_repeated_headers_footers(pages: list[str]) -> list[str]:
    """
    Remove first/last lines that appear on more than half of pages.
    Safe version: only removes short repeated metadata-like lines.
    """
    if len(pages) < 3:
        return pages

    firsts = Counter()
    lasts = Counter()

    parsed_pages = []

    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        parsed_pages.append(lines)

        if lines:
            firsts[lines[0]] += 1
            lasts[lines[-1]] += 1

    threshold = len(pages) // 2

    repeated = set()

    for ln, c in firsts.items():
        if c > threshold and len(ln) < 80:
            repeated.add(ln)

    for ln, c in lasts.items():
        if c > threshold and len(ln) < 80:
            repeated.add(ln)

    if not repeated:
        return pages

    cleaned_pages = []

    for lines in parsed_pages:
        filtered = [ln for ln in lines if ln not in repeated]
        cleaned_pages.append("\n".join(filtered))

    return cleaned_pages


# -------------------------
# SAFE WHITESPACE NORMALIZATION
# -------------------------

def _normalize_whitespace_safe(text: str) -> str:
    """
    Safer than collapsing everything blindly:
    - keeps line breaks
    - only normalizes inline spaces
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # normalize spaces inside line only
        line = _INLINE_WHITESPACE.sub(" ", line)
        cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)