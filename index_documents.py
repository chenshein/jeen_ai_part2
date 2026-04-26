#!/usr/bin/env python3
"""
Main pipeline: PDF/DOCX -> clean -> classify-and-chunk -> embed -> pgvector.

Each section of the document is independently classified and chunked
using the most appropriate strategy (sentence, paragraph, or fixed-size).

Usage:
    python index_documents.py path/to/file.pdf
    python index_documents.py path/to/file.docx --strategy auto
    python index_documents.py path/to/file.pdf  --strategy paragraph  # force one
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from pipeline.chunk import Chunk, chunk_text, chunks_to_strings
from pipeline.clean import clean_text
from pipeline.db import get_conn, init_db, store_chunks
from pipeline.embed import generate_embeddings
from pipeline.extract import extract_text

load_dotenv()

REQUIRED_ENV = ("GEMINI_API_KEY", "POSTGRES_URL")
DEFAULT_CHUNK_SIZE = 650
DEFAULT_OVERLAP_RATIO = 0.12  # 12% overlap for fixed-size strategy
DEFAULT_STRATEGY = "auto"     # per-section classification by default


def run_pipeline(
    file_path: str,
    strategy: str = DEFAULT_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> dict:
    """
    Run the full indexing pipeline on a single file.

    Parameters
    ----------
    file_path : str
        Path to a PDF or DOCX file.
    strategy : str
        "auto" (recommended) — classify each section independently.
        "fixed" / "sentence" / "paragraph" — force one strategy everywhere.
    chunk_size : int
        Max chunk size (chars for paragraph/sentence, tokens for fixed).
    overlap_ratio : float
        Overlap as fraction of chunk_size (used only by fixed-size strategy).

    Returns
    -------
    dict
        Summary with per-strategy breakdown.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

    overlap = max(1, int(round(chunk_size * overlap_ratio)))

    # --- Extract & clean ---
    raw_text = extract_text(file_path)
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        raise ValueError("No text extracted from document")

    # --- Chunk (with per-section classification when strategy="auto") ---
    chunks: list[Chunk] = chunk_text(
        cleaned_text,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        raise ValueError("Chunking produced zero chunks")

    # --- Embed ---
    plain_texts = chunks_to_strings(chunks)
    embeddings = generate_embeddings(plain_texts)
    filename = Path(file_path).name

    # --- Store ---
    # Build per-chunk metadata for the DB
    chunk_metadata = [
        {
            "strategy": c.strategy,
            "justification": c.justification,
            "section_index": c.section_index,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    with get_conn() as conn:
        init_db(conn)
        written = store_chunks(
            conn, plain_texts, embeddings, filename,
            strategy=strategy,
            chunk_metadata=chunk_metadata,
        )

    # --- Summary ---
    strategy_counts = Counter(c.strategy for c in chunks)

    return {
        "filename": filename,
        "mode": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(chunks),
        "rows_written": written,
        "strategy_breakdown": dict(strategy_counts),
        "sections_classified": max(c.section_index for c in chunks) + 1,
        "chunk_details": [
            {
                "chunk_index": i,
                "strategy": c.strategy,
                "justification": c.justification,
                "section": c.section_index,
                "length_chars": len(c.text),
                "preview": c.text[:80] + "..." if len(c.text) > 80 else c.text,
            }
            for i, c in enumerate(chunks)
        ],
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Index document into pgvector with intelligent chunking"
    )
    parser.add_argument("file", help="Path to PDF or DOCX")
    parser.add_argument(
        "--strategy",
        choices=["auto", "fixed", "sentence", "paragraph"],
        default=DEFAULT_STRATEGY,
        help=(
            "Chunking strategy. 'auto' (default) classifies each section "
            "independently. Others force a single strategy on the whole file."
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO,
        help="Overlap ratio for fixed-size strategy (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        result = run_pipeline(
            args.file,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap_ratio=args.overlap_ratio,
        )
        print(json.dumps(result, indent=2))
        return 0
    except FileNotFoundError:
        print(
            f"Error: file not found: {args.file}\n"
            "Please provide a valid .pdf or .docx path.",
            file=sys.stderr,
        )
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # Keep CLI output clean; avoid raw traceback noise.
        print(f"Error: unexpected failure: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())