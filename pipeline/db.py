"""PostgreSQL + pgvector connection, schema, and storage helpers.

Schema is fixed:
    id, chunk_text, embedding, filename, split_strategy, created_at

Each chunk stores its OWN split_strategy, since a single document may
use different strategies for different sections (sentence-based for Q&A,
paragraph-based for prose, fixed-size for unstructured content).
"""

from __future__ import annotations

import os
import uuid

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

EMBEDDING_DIM = 768

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY,
    filename        TEXT NOT NULL,
    split_strategy  TEXT NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       VECTOR({EMBEDDING_DIM}) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

STRATEGY_LABELS = {
    "fixed":     "fixed-size",
    "sentence":  "sentence-based",
    "paragraph": "paragraph-based",
}


def _label(strategy: str) -> str:
    """Map internal strategy id to the DB label, or pass it through unchanged."""
    return STRATEGY_LABELS.get(strategy, strategy)


def normalize_strategy(strategy: str | None) -> str | None:
    """Normalize strategy input to the DB labels."""
    if strategy is None:
        return None
    return _label(strategy)


def _make_chunk_id(filename: str, strategy_label: str, chunk_index: int) -> str:
    """Create a deterministic UUID per source file + strategy + chunk order."""
    seed = f"{filename}|{strategy_label}|{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def get_conn():
    """Open a Postgres connection using POSTGRES_URL and register pgvector."""
    url = os.environ.get("POSTGRES_URL")
    if not url:
        raise EnvironmentError("POSTGRES_URL is not set")
    try:
        conn = psycopg2.connect(url)
    except psycopg2.OperationalError as e:
        safe = url.split("@", 1)[-1] if "@" in url else url
        raise psycopg2.OperationalError(
            f"Could not connect to Postgres at {safe}. "
            f"Check that the docker container is running (docker compose ps), "
            f"that POSTGRES_URL in .env matches docker-compose.yml, and that "
            f"no other Postgres is bound to the same host port. Original error: {e}"
        ) from e
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def _table_needs_rebuild(conn) -> bool:
    """Return True when existing documents table does not match required schema."""
    required_columns = {"id", "filename", "split_strategy", "chunk_text", "embedding", "created_at"}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'documents'
            """
        )
        rows = cur.fetchall()
    if not rows:
        return False

    found_columns = {name for name, _ in rows}
    if found_columns != required_columns:
        return True

    id_type = next((dtype for name, dtype in rows if name == "id"), None)
    return id_type != "uuid"


def init_db(conn) -> None:
    """Create or migrate table/index to the required schema."""
    if _table_needs_rebuild(conn):
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()

    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()


def store_chunks(
    conn,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
    strategy: str = "auto",
    chunk_metadata: list[dict] | None = None,
) -> int:
    """
    Upsert chunks + embeddings; returns the number of rows written.

    Parameters
    ----------
    chunks : list[str]
        Plain text of each chunk.
    embeddings : list[list[float]]
        Embedding vector for each chunk.
    filename : str
        Source filename.
    strategy : str
        The mode used ("auto", "fixed", "sentence", "paragraph").
        When "auto", the actual per-chunk strategy comes from chunk_metadata.
        When forced ("fixed"/"sentence"/"paragraph"), all chunks get that label.
    chunk_metadata : list[dict] | None
        Per-chunk metadata from the classifier. Each dict has:
            - "strategy": "sentence" | "paragraph" | "fixed"
            - "justification": str
            - "section_index": int
            - "chunk_index": int
        Required when strategy="auto". Ignored when a single strategy is forced.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    if not chunks:
        return 0

    # Validate metadata for auto mode
    if strategy == "auto":
        if not chunk_metadata or len(chunk_metadata) != len(chunks):
            raise ValueError(
                "chunk_metadata must be provided for each chunk in auto mode"
            )

    rows = []
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        # Determine the strategy label for THIS specific chunk
        if chunk_metadata and idx < len(chunk_metadata):
            chunk_strategy = chunk_metadata[idx]["strategy"]
        else:
            chunk_strategy = strategy

        label = _label(chunk_strategy)
        chunk_id = _make_chunk_id(filename, label, idx)
        rows.append((chunk_id, filename, label, chunk_text, embedding))

    sql = """
        INSERT INTO documents (id, filename, split_strategy, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
        SET filename = EXCLUDED.filename,
            split_strategy = EXCLUDED.split_strategy,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            created_at = now();
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template="(%s::uuid, %s, %s, %s, %s)")
    conn.commit()
    return len(rows)


def clear_strategy(conn, filename: str, strategy: str) -> int:
    """Delete all rows for one (filename, split_strategy)."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM documents WHERE filename = %s AND split_strategy = %s",
            (filename, _label(strategy)),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted