# database/

Scripts for initializing, populating, enriching, and querying the Postgres paper corpus.

## Prerequisites

- PostgreSQL with the `pgvector` extension
- `POSTGRES_CONN_STRING` env var (e.g. via `.env`)
- `S2_API_KEY` env var for Semantic Scholar API access

## Files

### `init.py`
Creates the `papers` table and a GIN index on `author_ids`. Run once before any ingest.

```
python database/init.py
```

**Schema — `papers` table:**

| Column | Type | Notes |
|---|---|---|
| `corpus_id` | TEXT PK | Semantic Scholar CorpusId |
| `s2_paper_id` | TEXT UNIQUE | S2 paper hash |
| `url` | TEXT | |
| `title` | TEXT | |
| `abstract` | TEXT | |
| `comments` | TEXT | |
| `conclusion` | TEXT | |
| `year` | INT | |
| `venue` | TEXT | |
| `citation_count` | INT | |
| `reference_count` | INT | |
| `fields_of_study` | TEXT[] | All S2 FoS categories |
| `subfields` | TEXT[] | s2-fos-model categories only |
| `author_ids` | TEXT[] | Denormalized from `paper_authors`; GIN-indexed |
| `nomic` | vector(384) | |
| `massimo_title` | vector(384) | |
| `massimo_abstract` | vector(384) | |
| `massimo_metadata` | vector(384) | |
| `andrew` | vector(128) | |
| `autoresearch` | vector(128) | |
| `autoresearch_new` | vector(128) | |
| `audrey` | halfvec(64) | Euclidean tangent proxy for ANN |
| `audrey_hyp` | halfvec(64) | True Poincaré coordinates (rerank only) |
| `audrey_curvature` | real | Default 1.0 |
| `inserted_at` | TIMESTAMPTZ | |

---

### `ingest.py`
Pages through the S2 Bulk Search API and inserts up to 25,000 CS papers (year ≥ 2021) with abstracts.

```
python database/ingest.py
```

Config constants at top of file: `MIN_YEAR`, `TARGET_N`, `INSERT_BATCH`. Uses `ON CONFLICT DO NOTHING` so re-runs are safe.

---

### `enrich.py`
Fetches author and reference data from S2 for all papers already in the DB, then builds precomputed `eval_pairs`.

```
python database/enrich.py
```

Creates three additional tables:

**`paper_authors`** — `(corpus_id, author_id, author_name)`

**`paper_references`** — `(corpus_id, ref_corpus_id)` — referenced paper need not be in `papers`

**`eval_pairs`** — `(query_id, target_id, pair_type, weight)` — stored in both directions (A→B and B→A) for O(1) lookup

- `pair_type = 'author'`: weight = number of shared authors
- `pair_type = 'coupling'`: weight = number of shared references (threshold: `MIN_COUPLING = 5`)

After fetching, `author_ids` on the `papers` table is synced from `paper_authors` for fast GIN-based graph lookups.

---

### `backfill_subfields.py`
Backfills the `subfields` column (s2-fos-model categories only) for papers where it is NULL. Safe to re-run.

```
python database/backfill_subfields.py
```

---

### `utils.py`
Shared library for embedding contributors. Import this instead of writing raw SQL.

**Env var:** `POSTGRES_CONN_STRING`

**Embedding columns registry (`EMBEDDING_COLS`):**

| Column | Dim | Notes |
|---|---|---|
| `nomic` | 384 | |
| `massimo_title` | 384 | |
| `massimo_abstract` | 384 | |
| `massimo_metadata` | 384 | |
| `andrew` | 128 | |
| `autoresearch` | 128 | |
| `autoresearch_new` | 128 | |
| `audrey` | 64 | halfvec, Euclidean ANN |
| `audrey_hyp` | 64 | halfvec, Poincaré rerank |

**Key functions:**

```python
get_connection()                          # psycopg2 conn with pgvector registered
fetch_embeddings(conn, column, corpus_ids=None)  # → {corpus_id: np.ndarray}
get_unembedded(conn, column)              # → [corpus_id, ...] where column IS NULL
upsert_embeddings(conn, column, [(id, vec), ...])
build_ivf_indexes(conn, column, nlist=25) # run AFTER bulk write
drop_ivf_indexes(conn, column)            # run BEFORE bulk write
list_indexes(conn)                        # → [{index, size}, ...]
search_similar(conn, column, query_vec, k=10, nprobe=10)  # → [{corpus_id, title, dist}]
```

**Standard bulk-write workflow:**
```python
drop_ivf_indexes(conn, "andrew")
upsert_embeddings(conn, "andrew", pairs)
build_ivf_indexes(conn, "andrew")
```

IVF index name convention: `papers_{column}_ivf`. Tuned for ~25k papers (`nlist=25`).

## Run order (fresh setup)

```
python database/init.py
python database/ingest.py
python database/enrich.py
# then run each contributor's embedding script
# then build IVF indexes via utils.build_ivf_indexes
```
