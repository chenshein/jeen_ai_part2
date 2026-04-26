"""
Chunking module with per-section strategy classification.

Instead of applying one strategy to the entire document, this module:
1. Splits the document into logical sections (by headings or double-newlines).
2. Classifies each section independently based on content characteristics.
3. Applies the appropriate chunking strategy to each section.
4. Returns chunks with metadata (strategy used, section source, justification).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import nltk

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_NLTK_READY = False
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_TOKEN_RE = re.compile(r"\S+")
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+.+"               # Markdown headings
    r"|[A-Z][A-Za-z0-9 /&\-]{0,80}$"  # ALL-CAPS or Title-Case short lines
    r")",
    re.MULTILINE,
)
_QA_LINE_RE = re.compile(r"^\s*(?:Q\s*[:.\d]|[-•]\s*\w.*\?\s*$)", re.MULTILINE)
_QUESTION_RE = re.compile(r"\?")


def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
    _NLTK_READY = True


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk with metadata about how and why it was created."""
    text: str
    strategy: str          # "sentence" | "paragraph" | "fixed"
    justification: str     # human-readable reason for the strategy choice
    section_index: int     # which section of the document this came from
    chunk_index: int       # position within the section's chunks


@dataclass
class Section:
    """A logical section of the document, classified for chunking."""
    text: str
    index: int
    heading: str = ""
    strategy: str = ""
    justification: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    strategy: str = "auto",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Chunk a document using per-section classification.

    Parameters
    ----------
    text : str
        Full cleaned document text.
    strategy : str
        "auto" — classify each section independently (recommended).
        "fixed" / "sentence" / "paragraph" — force one strategy everywhere.
    chunk_size : int
        Maximum chunk size (characters for paragraph/sentence, tokens for fixed).
    overlap : int
        Overlap size for fixed-size strategy only.

    Returns
    -------
    list[Chunk]
        Chunks with metadata.  For backward compatibility, each Chunk's
        `text` attribute gives you the plain string.
    """
    if not text or not text.strip():
        return []

    # If a specific strategy is forced, apply it to the whole document
    if strategy in ("fixed", "sentence", "paragraph"):
        return _force_single_strategy(text, strategy, chunk_size, overlap)

    # --- Auto mode: per-section classification ---
    sections = _split_into_sections(text)
    all_chunks: list[Chunk] = []

    for section in sections:
        _classify_section(section)
        section_chunks = _chunk_section(section, chunk_size, overlap)
        all_chunks.extend(section_chunks)

    return all_chunks


def chunks_to_strings(chunks: list[Chunk]) -> list[str]:
    """Extract plain text from Chunk objects (backward-compat helper)."""
    return [c.text for c in chunks]


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> list[Section]:
    """
    Split a document into logical sections.

    Strategy:
    - First try heading-based splitting (markdown headings, title-case lines).
    - Fall back to double-newline paragraph groups if no headings found.
    - Merge very short consecutive blocks that belong together.
    """
    lines = text.split("\n")
    heading_indices = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Detect headings: markdown style, or short title-case/upper-case lines
        if _is_heading_line(stripped):
            heading_indices.append(i)

    if len(heading_indices) >= 2:
        return _split_by_headings(lines, heading_indices)

    # Fallback: split by paragraph groups (double newlines)
    return _split_by_paragraphs(text)


def _is_heading_line(line: str) -> bool:
    """Heuristic: is this line a heading?"""
    # Markdown heading
    if re.match(r"^#{1,6}\s+", line):
        return True
    # Short line, title-case or upper-case, no ending punctuation
    if (
        len(line) < 80
        and not line.endswith((".", "?", "!", ",", ";", ":"))
        and (line.istitle() or line.isupper())
        and len(line.split()) <= 10
    ):
        return True
    return False


def _split_by_headings(lines: list[str], heading_indices: list[int]) -> list[Section]:
    """Split document at heading boundaries."""
    sections = []

    # Content before first heading (if any)
    if heading_indices[0] > 0:
        pre = "\n".join(lines[: heading_indices[0]]).strip()
        if pre:
            sections.append(Section(text=pre, index=len(sections)))

    for i, h_idx in enumerate(heading_indices):
        end = heading_indices[i + 1] if i + 1 < len(heading_indices) else len(lines)
        heading = lines[h_idx].strip().lstrip("#").strip()
        body_lines = lines[h_idx + 1 : end]
        body = "\n".join(body_lines).strip()
        full_text = (heading + "\n" + body).strip() if body else heading
        sections.append(Section(text=full_text, index=len(sections), heading=heading))

    return sections


def _split_by_paragraphs(text: str) -> list[Section]:
    """Split by double-newline blocks, merging tiny adjacent blocks."""
    raw_blocks = [b.strip() for b in _PARAGRAPH_SPLIT.split(text) if b.strip()]

    # Before splitting: check if the WHOLE text is Q&A.
    # If all blocks are short Q&A pairs, keep them as one section so the
    # Q&A detector sees the full pattern and applies sentence-based strategy.
    all_lines = [l.strip() for l in text.splitlines() if l.strip()]
    if _section_is_qa(text, all_lines):
        return [Section(text=text.strip(), index=0)]

    # Merge blocks that are too small to stand alone (< 40 chars)
    merged: list[str] = []
    buf = ""
    for block in raw_blocks:
        if buf and len(block) < 40:
            buf = buf + "\n\n" + block
        elif buf and len(buf) < 40:
            buf = buf + "\n\n" + block
        else:
            if buf:
                merged.append(buf)
            buf = block
    if buf:
        merged.append(buf)

    return [Section(text=b, index=i) for i, b in enumerate(merged)]


# ---------------------------------------------------------------------------
# Section classification
# ---------------------------------------------------------------------------

def _classify_section(section: Section) -> None:
    """
    Classify a section and assign strategy + justification.

    Decision tree:
    1. High structure + independent statements (Q&A, definitions, lists)
       → sentence-based
    2. High structure + dependent sentences forming coherent blocks
       → paragraph-based
    3. Low structure (no clear sentences/paragraphs, OCR, wall-of-text)
       → fixed-size with overlap
    """
    text = section.text
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    word_count = len(text.split())

    # --- Check for Q&A / independent-statement pattern ---
    if _section_is_qa(text, lines):
        section.strategy = "sentence"
        section.justification = (
            "Section contains Q&A pairs or independent factual statements. "
            "Each statement is self-contained and independently retrievable, "
            "so sentence-based splitting preserves atomic meaning units."
        )
        return

    # --- Check for list of short independent items ---
    if _section_is_list(lines):
        section.strategy = "sentence"
        section.justification = (
            "Section contains a list of short independent items. "
            "Each item carries standalone meaning, making sentence-based "
            "splitting the best fit for retrieval precision."
        )
        return

    # --- Check for unstructured / wall-of-text ---
    if _section_is_unstructured(text, lines):
        section.strategy = "fixed"
        section.justification = (
            "Section lacks reliable structural markers (no clear paragraph "
            "breaks, inconsistent sentence boundaries). Fixed-size chunking "
            "with overlap is used as a safe fallback to prevent context loss "
            "at arbitrary boundaries."
        )
        return

    # --- Default: paragraph-based ---
    section.strategy = "paragraph"
    section.justification = (
        "Section contains well-structured prose where sentences build on "
        "each other to develop a coherent idea. Paragraph-based splitting "
        "preserves the author's logical grouping."
    )


def _section_is_qa(text: str, lines: list[str]) -> bool:
    """
    Detect Q&A or definition-style content with SHORT answers.

    Key distinction:
    - "What is X? It is Y." (1-2 sentence answers) → sentence-based (True)
    - "Q1: What is X?\nX is a framework that... [full paragraph]" → paragraph-based (False)

    The test is whether answers are atomic or developed. FAQ-style documents
    with multi-sentence paragraph answers should use paragraph-based splitting
    so each Q+answer block stays together as a coherent unit.
    """
    if not lines:
        return False

    question_lines = sum(1 for l in lines if "?" in l)
    qa_prefix_lines = sum(
        1 for l in lines if re.match(r"^\s*(?:Q|A)\s*[:.\d]", l, re.IGNORECASE)
    )

    # Q&A prefix pattern — but ONLY if answers are short.
    # If Q-prefix lines are a small fraction of total lines, the answers
    # are long paragraphs → paragraph-based, not sentence-based.
    if qa_prefix_lines >= 2:
        # Ratio of Q/A prefix lines to total lines.
        # High ratio (>40%) = short answers (Q&A style).
        # Low ratio (<40%) = long answers (FAQ with paragraphs).
        qa_ratio = qa_prefix_lines / len(lines)
        if qa_ratio >= 0.4:
            return True
        # Low ratio: these are paragraph-length answers, not atomic Q&A
        return False

    # At least 25% of lines are questions, and answers are short
    if len(lines) >= 2 and question_lines / len(lines) >= 0.25:
        non_q_lines = [l for l in lines if "?" not in l]
        if non_q_lines:
            avg_answer_len = sum(len(l.split()) for l in non_q_lines) / len(non_q_lines)
            if avg_answer_len < 30:
                return True

        # Handle Q&A on same line: "What is X? It is Y."
        # If most lines contain a ? AND also have text after it, it's Q&A
        if not non_q_lines and question_lines >= 2:
            qa_same_line = sum(
                1 for l in lines
                if re.search(r"\?\s+\S", l)  # has content after the ?
            )
            if qa_same_line / len(lines) >= 0.5:
                return True

    return False


def _section_is_list(lines: list[str]) -> bool:
    """
    Detect list-style content: bullet points, numbered items, definitions.
    """
    if len(lines) < 3:
        return False

    bullet_lines = sum(
        1
        for l in lines
        if re.match(r"^\s*(?:[-•*]|\d+[.)]\s)", l)
    )

    # More than 60% bullet/numbered lines → list
    if bullet_lines / len(lines) >= 0.6:
        return True

    # Short independent lines (avg < 15 words, each ending with period or standalone)
    avg_words = sum(len(l.split()) for l in lines) / len(lines)
    lines_with_period = sum(1 for l in lines if l.endswith("."))
    if avg_words < 15 and lines_with_period / len(lines) >= 0.5:
        return True

    return False


def _section_is_unstructured(text: str, lines: list[str]) -> bool:
    """
    Detect unstructured content: OCR output, raw logs, walls of text.

    Signals:
    - Very few paragraph breaks relative to text length
    - Very long lines with no sentence-ending punctuation
    - High ratio of non-alphabetic characters (OCR noise)
    """
    if not lines:
        return False

    # Single giant block with no paragraph breaks and very long
    if len(lines) <= 2 and len(text) > 1000:
        return True

    # Check sentence boundary reliability
    _ensure_nltk()
    sentences = nltk.sent_tokenize(text)
    avg_sent_len = sum(len(s) for s in sentences) / max(len(sentences), 1)

    # If average "sentence" is > 300 chars, boundaries are unreliable
    if avg_sent_len > 300:
        return True

    # High OCR noise: many non-alpha characters
    alpha_count = sum(1 for c in text if c.isalpha())
    if len(text) > 100 and alpha_count / len(text) < 0.5:
        return True

    return False


# ---------------------------------------------------------------------------
# Per-section chunking
# ---------------------------------------------------------------------------

def _chunk_section(section: Section, chunk_size: int, overlap: int) -> list[Chunk]:
    """Apply the classified strategy to a section and produce Chunk objects."""
    text = section.text

    if section.strategy == "sentence":
        raw = _do_sentence(text, chunk_size)
    elif section.strategy == "fixed":
        raw = _do_fixed(text, chunk_size, overlap)
    else:
        raw = _do_paragraph(text, chunk_size)

    return [
        Chunk(
            text=t,
            strategy=section.strategy,
            justification=section.justification,
            section_index=section.index,
            chunk_index=i,
        )
        for i, t in enumerate(raw)
        if t.strip()
    ]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _do_sentence(text: str, chunk_size: int) -> list[str]:
    """
    Sentence-based splitting.

    Groups sentences up to chunk_size, always breaking at sentence boundaries.
    For Q&A content, each question+answer pair becomes its own chunk
    to maximize retrieval precision.
    """
    _ensure_nltk()
    sentences = nltk.sent_tokenize(text)

    # Detect Q&A pattern and pair questions with their answers
    paired = _pair_qa_sentences(sentences)

    # If Q&A content, each pair is its own chunk (don't merge by size)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if _section_is_qa(text, lines):
        return [p for p in paired if p.strip()]

    # Non-Q&A: group sentences up to chunk_size
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in paired:
        unit_len = len(unit)
        combined_len = current_len + (1 if current else 0) + unit_len

        if combined_len <= chunk_size:
            current.append(unit)
            current_len = combined_len
        else:
            if current:
                chunks.append(" ".join(current))
            current = [unit]
            current_len = unit_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _pair_qa_sentences(sentences: list[str]) -> list[str]:
    """
    If sentences alternate question/answer, pair them so a question
    is never separated from its answer.
    """
    if len(sentences) < 2:
        return sentences

    paired: list[str] = []
    i = 0
    while i < len(sentences):
        if "?" in sentences[i] and i + 1 < len(sentences) and "?" not in sentences[i + 1]:
            # Pair question with its answer
            paired.append(sentences[i] + " " + sentences[i + 1])
            i += 2
        else:
            paired.append(sentences[i])
            i += 1

    return paired


def _do_paragraph(text: str, chunk_size: int) -> list[str]:
    """
    Paragraph-based splitting.

    Keeps paragraph boundaries. Merges short paragraphs up to chunk_size.
    Splits very long paragraphs (> chunk_size) using sentence fallback.

    For FAQ-style documents with Q1:/Q2:/... patterns, splits at question
    boundaries and unwraps soft line breaks from PDF extraction.
    """
    # First: try splitting by numbered question patterns (Q1:, Q2:, etc.)
    # These are structural boundaries even without double-newlines
    q_pattern = re.compile(r'\n(?=Q\d+\s*[:.]\s)', re.IGNORECASE)
    q_blocks = q_pattern.split(text)
    if len(q_blocks) >= 3:
        # Successfully split by Q-numbers — unwrap soft line breaks within each block
        paragraphs = [_unwrap_soft_linebreaks(b.strip()) for b in q_blocks if b.strip()]
    else:
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]

        # If no double-newline splits found, treat newlines as soft paragraph breaks
        if len(paragraphs) <= 1 and "\n" in text:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            # Re-merge lines that are continuations (short lines that aren't headings)
            paragraphs = _merge_continuation_lines(paragraphs)

    chunks: list[str] = []
    current = ""

    for p in paragraphs:
        # If a single paragraph exceeds chunk_size, fall back to sentence split
        if len(p) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            sub_chunks = _do_sentence(p, chunk_size)
            chunks.extend(sub_chunks)
            continue

        candidate = (current + "\n\n" + p).strip() if current else p

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


def _unwrap_soft_linebreaks(text: str) -> str:
    """Join soft-wrapped lines (from PDF extraction) into proper sentences."""
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if result:
                result.append("")  # preserve intentional blank lines
            continue
        # If previous line didn't end with sentence-ending punctuation,
        # this is a continuation — join with space instead of newline
        if result and result[-1] and not result[-1].endswith((".", "?", "!", ":")):
            result[-1] = result[-1] + " " + stripped
        else:
            result.append(stripped)
    return "\n".join(result)


def _merge_continuation_lines(lines: list[str]) -> list[str]:
    """Merge short continuation lines back into their parent paragraph."""
    if not lines:
        return lines

    merged: list[str] = []
    buf = lines[0]

    for line in lines[1:]:
        # If current buffer or new line is very short and not a heading,
        # they likely belong together
        if (
            not _is_heading_line(line)
            and not _is_heading_line(buf.split("\n")[0])
            and (len(line.split()) < 5 or len(buf.split()) < 5)
        ):
            buf = buf + " " + line
        else:
            merged.append(buf)
            buf = line

    merged.append(buf)
    return merged


def _do_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Fixed-size splitting with overlap.

    Operates on characters but breaks at word boundaries to avoid
    splitting words. chunk_size and overlap are in characters,
    consistent with the other strategies.
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []

    chunks: list[str] = []
    current_tokens: list[str] = []
    current_len = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]
        # +1 for the space separator (except for first token in chunk)
        added_len = len(token) + (1 if current_tokens else 0)

        if current_len + added_len <= chunk_size:
            current_tokens.append(token)
            current_len += added_len
            i += 1
        else:
            if current_tokens:
                chunks.append(" ".join(current_tokens))
                # Calculate how many tokens to keep for overlap
                overlap_tokens: list[str] = []
                overlap_len = 0
                for t in reversed(current_tokens):
                    new_len = overlap_len + len(t) + (1 if overlap_tokens else 0)
                    if new_len > overlap:
                        break
                    overlap_tokens.insert(0, t)
                    overlap_len = new_len
                current_tokens = overlap_tokens
                current_len = overlap_len
            else:
                # Single token exceeds chunk_size — take it alone
                chunks.append(token)
                i += 1
                current_tokens = []
                current_len = 0

    if current_tokens:
        chunks.append(" ".join(current_tokens))

    return chunks


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

def _force_single_strategy(
    text: str, strategy: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Apply one strategy to the entire document (no per-section classification)."""
    if strategy == "sentence":
        raw = _do_sentence(text, chunk_size)
    elif strategy == "fixed":
        raw = _do_fixed(text, chunk_size, overlap)
    else:
        raw = _do_paragraph(text, chunk_size)

    justification = {
        "sentence": "Forced sentence strategy — applied uniformly.",
        "fixed": "Forced fixed-size strategy — applied uniformly.",
        "paragraph": "Forced paragraph strategy — applied uniformly.",
    }[strategy]

    return [
        Chunk(text=t, strategy=strategy, justification=justification,
              section_index=0, chunk_index=i)
        for i, t in enumerate(raw)
        if t.strip()
    ]