
# Document Indexing Pipeline (RAG)

A document indexing pipeline that converts PDF/DOCX files into searchable vector embeddings stored in PostgreSQL with `pgvector`.


## How it works

1. **Extract** text from PDF / DOCX files
2. **Clean** and preprocess the text (Unicode normalization, header/footer removal, whitespace)
3. **Split** text into chunks — paragraph / sentence / fixed-size with overlap (chosen per section)
4. **Embed** each chunk using the Gemini API (`gemini-embedding-001`, 768-dim)
5. **Store** chunks + embeddings in PostgreSQL with metadata (source file, strategy, timestamps)
6. **Secure** API keys using environment variables in `.env`

---

## Installation

### 1) Start PostgreSQL with pgvector (Docker)

```bash
docker compose up -d
```

This launches the `pgvector/pgvector:pg16` image and binds host port **5433** to avoid colliding with a local Postgres install.

### 2) Install Python dependencies

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3) Configure environment

Copy `.env.example` to `.env` and fill in your Gemini API key:

```env
GEMINI_API_KEY=your_key_here
POSTGRES_URL=postgresql://vectoruser:vectorpass@localhost:5433/vectordb
```

---

## Usage

### Index a document

```bash
python index_documents.py file.docx
```

The script prints a JSON summary with total chunks, per-strategy breakdown, and chunk previews.

---

## Example

```bash
$ docker compose up -d
$ python index_documents.py file.docx
{
  "filename": "file.docx",
  "mode": "auto",
  "chunk_size": 650,
  "overlap": 78,
  "total_chunks": 4,
  "rows_written": 4,
  "strategy_breakdown": {
    "sentence": 4
  },
  "sections_classified": 1,
  "chunk_details": [
    {
      "chunk_index": 0,
      "strategy": "sentence",
      "justification": "Section contains Q&A pairs or independent factual statements. Each statement is self-contained and independently retrievable, so sentence-based splitting preserves atomic meaning units.",
      "section": 0,
      "length_chars": 54,
      "preview": "What is this system? It is a document indexing system."
    },
    {
      "chunk_index": 1,
      "strategy": "sentence",
      "justification": "Section contains Q&A pairs or independent factual statements. Each statement is self-contained and independently retrievable, so sentence-based splitting preserves atomic meaning units.",
      "section": 0,
      "length_chars": 57,
      "preview": "How does it work? It extracts text and builds embeddings."
    },
    {
      "chunk_index": 2,
      "strategy": "sentence",
      "justification": "Section contains Q&A pairs or independent factual statements. Each statement is self-contained and independently retrievable, so sentence-based splitting preserves atomic meaning units.",
      "section": 0,
      "length_chars": 60,
      "preview": "Why is it useful? It enables semantic search over documents."
    },
    {
      "chunk_index": 3,
      "strategy": "sentence",
      "justification": "Section contains Q&A pairs or independent factual statements. Each statement is self-contained and independently retrievable, so sentence-based splitting preserves atomic meaning units.",
      "section": 0,
      "length_chars": 46,
      "preview": "Can it scale? Yes, it supports large datasets."
    }
  ]
}
```

---

## PostgreSQL schema

A single `documents` table:

| column          | type           | description                                          |
|-----------------|----------------|------------------------------------------------------|
| `id`            | UUID           | deterministic per (filename, strategy, chunk_index)  |
| `filename`      | TEXT           | source file name                                     |
| `split_strategy`| TEXT           | `sentence-based` / `paragraph-based` / `fixed-size`  |
| `chunk_text`    | TEXT           | text content of the chunk                            |
| `embedding`     | VECTOR(768)    | Gemini embedding                                     |
| `created_at`    | TIMESTAMPTZ    | insertion timestamp                                  |


Each chunk stores its **own** `split_strategy` because one document can produce chunks with mixed strategies (per-section classification).

---

## Chunking decision tree

The pipeline splits each document into sections and classifies each section independently — one document can produce chunks with mixed strategies.

- **Heading-based section split** — splits at markdown headings (`#`) or short title-case / ALL-CAPS lines; falls back to double-newline blocks otherwise.
- **`sentence` strategy** — picked for Q&A pairs and lists of short independent items, where each unit is atomic and self-contained.
- **`paragraph` strategy** *(default)* — picked for coherent prose where sentences build on each other within a paragraph.
- **`fixed` strategy** — picked for unstructured / OCR / wall-of-text where sentence boundaries are unreliable; uses overlap (~12%) to preserve context across boundaries.
- **Q&A vs FAQ distinction** — short atomic answers → `sentence`; long paragraph answers → `paragraph` (decided by Q/A-prefix line ratio and average answer length).
- **Q&A pairing** — within a `sentence` section, each question is paired with its following answer so they never end up in different chunks.

---

## Project structure

```
.
├── index_documents.py      # CLI entry point
├── pipeline/
│   ├── extract.py          # PDF / DOCX text extraction
│   ├── clean.py            # text normalization
│   ├── chunk.py            # per-section classification + chunking
│   ├── embed.py            # Gemini embeddings (batched, retried)
│   └── db.py               # pgvector connection, schema, storage
├── docker-compose.yml      # pgvector/pg16 container
├── requirements.txt
└── .env.example
```
