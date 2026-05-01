# Nomic Embedding Approach

Baseline semantic search pipeline using `nomic-ai/nomic-embed-text-v1.5`. Embeds each paper as a single vector (title + abstract concatenated) stored in the `nomic` pgvector column, then retrieves via IVF cosine similarity.

This serves as the single-field baseline against Massimo's weighted multi-field approach (title / abstract / metadata with query-aware weight adjustment).

---

## Files

| File | Description |
|------|-------------|
| `embed.py` | Fetches unembedded papers from the `papers` table, calls the Nomic API (`search_document` task type) to generate 384-dim embeddings of `title + abstract`, and upserts into the `nomic` pgvector column. Processes in chunks of 2,000 papers (256 texts per API call). Drops and rebuilds IVF indexes on completion. |
| `query.py` | Loads `nomic-ai/nomic-embed-text-v1.5` weights from HuggingFace (no API key required), embeds a natural-language query with the `search_query:` prefix, truncates to 384 dims, and returns ranked results from the `nomic` column via IVF search. |
| `evaluation/generate_nomic_results.py` | Runs all benchmark queries through `query.py` and writes results to a JSONL file. Scores are `1.0 - distance`. Output run ID: `nomic`. |

## Pipeline Flow

```
embed.py
└─► nomic column in papers table (PostgreSQL / pgvector)

query.py (search interface)
└─► ranked results by cosine similarity

evaluation/generate_nomic_results.py
└─► evaluation/nomic-results.jsonl
```

## Embedding Model

- **Model:** `nomic-ai/nomic-embed-text-v1.5`
- **Dim:** 384 (native 768, truncated at query time and via API `dimensionality` param)
- **Indexing prefix:** `search_document:` (set via Nomic API `task_type`)
- **Query prefix:** `search_query: ` (prepended manually before HuggingFace encode)
- **Storage:** single `nomic` pgvector column in the `fml` PostgreSQL database

## Usage

```bash
# Embed all unembedded papers (requires NOMIC_API_KEY)
python nomic-embedding/embed.py

# Embed a subset
python nomic-embedding/embed.py --offset 5000 --amount 5000

# Search
python nomic-embedding/query.py "attention mechanisms in transformers"
python nomic-embedding/query.py "graph neural networks" --k 20 --nprobe 30

# Generate evaluation results
python nomic-embedding/evaluation/generate_nomic_results.py
python nomic-embedding/evaluation/generate_nomic_results.py \
    --queries evaluation/benchmark_queries.jsonl \
    --output submissions/nomic-results.jsonl --k 10 --nprobe 25
```

## Environment Variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `NOMIC_API_KEY` | `embed.py` | Authenticates with the Nomic API for indexing |
| `POSTGRES_CONN_STRING` | both | Database connection |

`query.py` loads the model directly from HuggingFace — no API key needed at query time.

## Key Design Decisions

- Single embedding per paper (title + abstract joined by newline) keeps the schema simple and serves as a clean baseline.
- Embedding via the Nomic API at index time vs. local HuggingFace weights at query time — API gives exact parity with the hosted model; local inference avoids rate limits and API costs during search.
- IVF indexes are dropped before a bulk embed run and rebuilt after to avoid incremental index overhead on large inserts.
