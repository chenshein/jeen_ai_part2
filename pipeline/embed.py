"""Generate Gemini embeddings for chunks (batched, with retries)."""

from __future__ import annotations

import os
from typing import Iterable

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

EMBED_MODEL = "gemini-embedding-001"
OUTPUT_DIM = 768
BATCH_SIZE = 100

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _embed_batch(batch: list[str], task_type: str) -> list[list[float]]:
    client = _get_client()
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=OUTPUT_DIM,
    )
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=batch,
        config=config,
    )
    return [list(e.values) for e in result.embeddings]


def generate_embeddings(
    chunks: Iterable[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed each chunk; returns a list of OUTPUT_DIM-dim float vectors."""
    items = list(chunks)
    if not items:
        return []

    out: list[list[float]] = []
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        out.extend(_embed_batch(batch, task_type))
    return out


def embed_query(query: str) -> list[float]:
    """Embed a single query string with the RETRIEVAL_QUERY task type."""
    return _embed_batch([query], task_type="RETRIEVAL_QUERY")[0]
