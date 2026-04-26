
# Document Indexing Pipeline

PDF/DOCX → Extract → Clean → Split into Sections → Classify & Chunk → Embed → PostgreSQL (pgvector)

---

## Pipeline Overview

### 1) Extraction (`pipeline/extract.py`)
- Supports `.pdf` and `.docx`
- PDF: extracts text page-by-page and joins with page separators
- DOCX: extracts paragraph text and joins into a single string

---

### 2) Cleaning (`pipeline/clean.py`)
- Unicode normalization (`NFKC`)
- Removes control characters
- Removes repeated headers/footers
- Removes page-number-only lines
- Normalizes whitespace while preserving structure

---

## 3) Classify & Chunk (`pipeline/chunk.py`)

The document is split into **sections first**, then each section is chunked independently.

---

### 🔹 Section Splitting

Sections are created using:

- **Heading-based split (preferred)**
  - Markdown headings (`# ...`)
  - Title-case / ALL CAPS short lines

- **Fallback: paragraph split**
  - Split by double newlines (`\n\n`)
  - Merge small fragments

- **Special case: Q&A detection**
  - If the whole document looks like Q&A → treated as one section

---

### 🔹 Section Classification (per section)

Each section is assigned a strategy:

#### `sentence`
Used when:
- Q&A / FAQ style content
- Bullet lists or independent short items

---

#### `fixed`
Used when:
- Unstructured / OCR-like text
- Weak sentence or paragraph boundaries

---

#### `paragraph` (default)
Used when:
- Structured prose
- Sentences form a coherent narrative

---

### 🔹 Chunking Strategies

#### Sentence strategy
- Split into sentences
- Q&A pairs are kept together
- Sentences grouped up to `chunk_size`

---

#### Paragraph strategy
- Split by paragraphs (`\n\n`)
- Merge small paragraphs
- Oversized paragraphs are split into sentences
- Supports Q1:/Q2: style FAQ structures

---

#### Fixed-size strategy
- Word-based splitting
- Builds chunks up to `chunk_size`
- Adds overlap between chunks for context continuity

---

#### Overlap
- Used only in fixed-size strategy
- Preserves context between chunks (~12% default)

---

## 4) Embedding (`pipeline/embed.py`)
- Uses `gemini-embedding-001`
- Produces 768-dimensional vectors
- Batching for efficiency
- Retries with exponential backoff

---

## 5) Storage (`pipeline/db.py`)
- PostgreSQL + pgvector
- Ensures schema + vector index

Stores per chunk:
- `id` (UUID)
- `filename`
- `chunk_text`
- `embedding (VECTOR(768))`
- `split_strategy`
- `created_at`

---

## 6) Orchestration (`index_documents.py`)
- Validates environment variables
- Runs full pipeline:
  - extract → clean → chunk → embed → store
- Outputs JSON summary:
  - total chunks
  - strategy breakdown
  - chunk previews

---


## How to Run

### 1) Start PostgreSQL (Docker)

```bash

docker compose up -d

```

### 2) Install dependencies

```bash

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

```

### 3) Configure environment (`.env`)

```env

GEMINI_API_KEY=your_key

POSTGRES_URL=postgresql://vectoruser:vectorpass@localhost:5433/vectordb

```

### 4) Index documents

```bash

python index_documents.py file.pdf

```