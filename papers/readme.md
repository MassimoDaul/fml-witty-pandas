# Papers Dataset

Academic paper retrieval for the Student Research Helper. Given a text description of a research idea, returns the most semantically similar papers using vector embeddings and nearest-neighbor search.

---

## Data Source

**Kaggle arXiv metadata snapshot** — `Cornell-University/arxiv`

- ~2.4 million papers, all fields and categories
- Each record includes: `id`, `title`, `abstract`, `categories`, `authors_parsed`, `versions`, `update_date`
- Default ingest: first 10,000 papers (configurable via `--qty` and `--offset`)

---

## Stack

| Component | Choice |
|---|---|
| Dataset | Kaggle `Cornell-University/arxiv` (JSON Lines, ~4 GB) |
| Embedding model | `nomic-ai/nomic-embed-text-v1.5` via `sentence-transformers` |
| Embedding dimensions | 768 (float32, L2-normalized) |
| Vector database | Supabase Postgres + `pgvector` extension |
| Similarity index | HNSW (`vector_cosine_ops`, m=16, ef_construction=128) |
| Language | Python 3.11+ |

---

## Key Algorithms

**Embedding — asymmetric semantic search**

Text is encoded with `nomic-embed-text-v1.5`, which requires task-type prefixes for asymmetric retrieval:
- Documents at ingest: `"search_document: {title}. {abstract}"`
- Queries at search time: `"search_query: {user input}"`

Embeddings are L2-normalized so cosine similarity reduces to a dot product.

**Nearest-neighbor search — HNSW via pgvector**

Similarity search uses the `<=>` cosine distance operator in Postgres:

```sql
SELECT *, 1 - (embedding <=> query_vec) AS similarity
FROM papers
ORDER BY embedding <=> query_vec
LIMIT k;
```

The HNSW index provides approximate nearest-neighbor search in sub-linear time. Without the index, this is an exact brute-force scan (O(n) dot products).

**Text cleaning**

Abstract text is pre-processed before embedding: display and inline math (`$$...$$`, `$...$`) is stripped, LaTeX commands (`\textbf{}`, `\cite{}`, etc.) are removed or replaced with their inner content, and whitespace is normalized.

---

## Database Schema

```sql
CREATE TABLE papers (
    arxiv_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    abstract    TEXT NOT NULL,
    categories  TEXT[]  NOT NULL,
    authors     TEXT[],
    published   DATE,
    updated     DATE,
    embedding   vector(768),
    inserted_at TIMESTAMPTZ DEFAULT now()
);
```

---

## File Structure

```
papers/
├── run.py          Ingest pipeline CLI
├── query.py        KNN search CLI
├── setup_db.sql    Schema (run once in Supabase SQL editor)
├── requirements.txt
└── ingest/
    ├── config.py       Paths, batch sizes, model name
    ├── clean.py        LaTeX stripping, embed input builder
    ├── embed.py        Model load and batched encode
    ├── db.py           Postgres connection, upsert, index build
    ├── checkpoint.py   Resume support per offset
    └── pipeline.py     Main ingest loop
```

---

## Setup

```bash
pip install -r requirements.txt
```

Enable `pgvector` in Supabase (once, in the SQL editor):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Then create the table:
```bash
python run.py setup-db
```

---

## Ingest — `run.py`

```
python run.py <command> [options]
```

| Command | Description |
|---|---|
| `setup-db` | Create the `papers` table in Supabase |
| `download` | Download the Kaggle dataset to `data/` |
| `ingest` | Embed and upload papers |
| `build-index` | Build HNSW index (run after ingest is complete) |
| `verify` | Print row counts and embedding coverage |

**`ingest` options:**

| Flag | Default | Description |
|---|---|---|
| `--qty N` | `10000` | Number of papers to process (`0` = all) |
| `--offset N` | `0` | Line in the dataset to start from |

**Examples:**

```bash
# First 10,000 papers (default)
python run.py ingest

# Next 10,000
python run.py ingest --offset 10000 --qty 10000

# Everything from line 50,000 onward
python run.py ingest --offset 50000 --qty 0

# Build index after all uploads are done
python run.py build-index
```

Ingest is **resumable** — if interrupted, re-running the same command picks up from the last checkpoint (`data/checkpoint_{offset}.txt`). Re-running a completed range is safe; existing rows are skipped.

---

## Query — `query.py`

```
python query.py "<research topic>" [options]
```

| Flag | Default | Description |
|---|---|---|
| `--k N` | `10` | Number of nearest neighbors to return |
| `--no-abstract` | off | Hide abstract snippets in output |

**Examples:**

```bash
python query.py "machine learning for protein folding"
python query.py "transformer attention mechanisms" --k 20
python query.py "quantum computing error correction" --k 5 --no-abstract
```

**Importing `search()` in other code:**

```python
from query import search

results = search("neural networks for image segmentation", k=10)
# results: list of dicts with keys:
#   arxiv_id, title, abstract, categories, authors, published, similarity
for r in results:
    print(r["similarity"], r["title"])
```
