# Cephalo

Cephalo is a research-paper search application and retrieval-evaluation project for **Fundamentals of Machine Learning**. Our research question:

> **What is the impact of research paper embeddings on search effectiveness?**

The app helps users find papers for a question, topic, method, or project idea. The technical work compares multiple paper-representation strategies over the same 25,000-paper Semantic Scholar corpus and evaluates how those strategies change the **top-10 results** users actually inspect.

---

## What Cephalo Does

Cephalo supports a researcher workflow from query to follow-up reading:

1. **Search**: A user enters a plain-language research question or topic.
2. **Retrieve**: The backend embeds the query and retrieves ranked papers from Postgres using `pgvector`.
3. **Refine**: Users can constrain results by subject / arXiv-style field, year range, sorting mode, and result count.
4. **Inspect**: Each paper page shows title, abstract, venue/year, field tags, Semantic Scholar link, and related papers.
5. **Explore citation context**: Paper pages include a D3 reference/citation graph with cached Semantic Scholar lookups.
6. **Visualize embedding space**: The results page exposes a PCA-projected 2D scatter of Massimo embedding vectors for the query and its top results, with a 3D Three.js embedding cloud view in progress.

The interface is intentionally stable across retrieval methods: the same user workflow can support different embedding backends, so retrieval quality can improve without changing how users search.

---

## Dataset

The project uses a local Postgres database containing **25,000 Semantic Scholar papers**.

> Note: we decided to host the 25k paper database locally, but accessing this database is a bit tricky (involves downloading Tailscale). For demo purposes we've uploaded a subset of 1000 papers to Supabase.

| Field | Value |
|---|---|
| Primary task | Research paper search |
| Source | Semantic Scholar |
| Field of study | Computer Science |
| Date range | 2021–2026 |
| Database | Postgres with `pgvector` |
| Main table | `papers` |

The canonical schema stores paper metadata and embedding columns in one table:

- `corpus_id`, `s2_paper_id`, `url`
- `title`, `abstract`, `comments`, `conclusion`
- `year`, `venue`
- `citation_count`, `reference_count`
- `fields_of_study`, `subfields`, `author_ids`
- embedding columns for the different retrieval backends

The current app requires access to a populated Postgres database through `POSTGRES_CONN_STRING`.

---

## Retrieval Backends

Cephalo compares four retrieval strategies. Each strategy returns the same JSONL shape for evaluation: one row per benchmark query and exactly 10 ranked papers per query.

| Backend | Representation | Main idea |
|---|---|---|
| `nomic` | 384-dimensional title + abstract embedding | Baseline semantic retrieval using `nomic-embed-text-v1.5`, cosine distance, and `pgvector` |
| `massimo` | Segmented title, abstract, and metadata embeddings | Field-aware weighted scoring that adjusts title / abstract / metadata influence based on query language |
| `andrew` | 128-dimensional graph-refined embedding | Heterogeneous GNN / HGT model combining Nomic semantic vectors with venue, field, and nearest-neighbor structure |
| `audrey` | 64-dimensional hyperbolic representation | Poincaré-ball representation supervised by a two-level hierarchy from broad field labels and in-field clusters |

### Baseline Nomic Retrieval

The baseline embeds the query with the `search_query:` prefix and ranks papers by cosine distance against the `nomic` column. The main app path uses this backend for live paper search and related-paper retrieval.

### Weighted Segmented Retrieval

The segmented approach avoids compressing all paper information into a single vector. It stores separate title, abstract, and metadata representations, then combines their cosine scores with query-dependent weights. This tests whether different fields carry different retrieval signals.

### Graph / HGT Retrieval

The graph approach starts from Nomic paper embeddings and adds structural relationships from the research landscape. The training graph includes:

- semantic KNN edges from Nomic vectors
- paper-to-venue edges
- paper-to-field edges

The model uses two HGTConv layers with attention over heterogeneous node/edge types, then outputs 128-dimensional paper vectors. The final stored representation combines graph output and the projected semantic branch.

### Hyperbolic Retrieval

The hyperbolic backend tests whether hierarchical geometry helps exploratory paper search. It freezes the Nomic encoder, trains a projection into the Poincaré ball, stores both a tangent-space ANN proxy and true hyperbolic coordinates, then reranks candidates by Poincaré distance.

---

## Evaluation

The evaluation layer compares final **user-facing top-10 result lists**, not model internals.

### Benchmark Design

- **100 shared benchmark queries**
- **50 broad queries** for field exploration / literature review
- **50 specific queries** for targeted citation-finding
- **50 queries include year filters**
- The same query set is used for every retrieval strategy
- Each system returns exactly 10 ranked papers per query

### Judging and Metrics

The evaluator hydrates returned paper IDs with metadata from the shared database and uses an LLM-as-judge layer for:

- paper relevance on a 0–3 scale
- broad-query exploration range on a 1–5 scale
- anonymous win-rate comparison across systems

Reported metrics include:

- `Precision@10`
- `nDCG@10`
- `Mean relevance@10`
- `Reciprocal rank@10`
- `Pooled relevant recall@10`
- broad-query exploration diagnostics
- pairwise preferences
- corpus coverage and unique-paper contribution

### Current Four-Model Results

Latest evaluated runs are stored in:

```text
evaluation/real_four_model_comparison/
```

Summary from the committed evaluation report:

| Run | Precision@10 | nDCG@10 | Mean relevance@10 | Win rate |
|---|---:|---:|---:|---:|
| `andrew` | 0.2630 | 0.7342 | 1.0540 | 9% |
| `audrey` | 0.2050 | 0.6235 | 0.8130 | 1% |
| `massimo` | 0.4060 | 0.8726 | 1.4990 | 46% |
| `nomic` | 0.4010 | 0.8801 | 1.4960 | 44% |

Across all runs, the comparison surfaced **1,826 unique papers**, or approximately **7.30%** of the 25,000-paper corpus. The results show that the strongest average systems were the Nomic baseline and weighted segmented backend, while the graph and hyperbolic approaches contributed different retrieval behavior and coverage patterns.

---

## Web App

The web app lives in `app/` and uses:

| Layer | Implementation |
|---|---|
| Server | FastAPI + Uvicorn |
| Templates | Jinja2 |
| Frontend behavior | HTMX; JavaScript / static assets |
| Citation graph | D3 |
| Embedding space visualization | 2D PCA scatter (`embedding-space.js`); 3D cloud (`embedding-cloud.js`, Three.js — implemented, not yet wired into app routes) |
| Database | Postgres connection pool with `pgvector` registration |
| External paper graph data | Semantic Scholar API lookups with TTL caching |

Important routes:

| Route | Purpose |
|---|---|
| `GET /` | Search page |
| `GET /results` | Ranked paper or author results with filters |
| `POST /search` | Legacy form redirect into `/results` |
| `GET /paper/{corpus_id}` | Paper detail page |
| `GET /paper/{corpus_id}/related` | Related papers by title + abstract similarity |
| `GET /api/paper/{corpus_id}/references` | Reference/citation graph data |
| `GET /api/embedding-space` | PCA-projected Massimo embedding coordinates for a query and its top results (JSON) |
| `GET /author` | Author lookup and associated papers |

---

## Project Structure

```text
.
├── app/                                  # FastAPI app, templates, static JS/CSS, citation graph, embedding viz
├── data/                                 # Raw Semantic Scholar snapshot and ingest checkpoint
├── database/                             # Canonical Postgres schema, ingest, enrichment, shared pgvector utilities
├── papers/                               # Baseline paper search, related-paper retrieval, older ingest helpers
├── nomic-embedding/                      # Nomic baseline embedding and query pipeline
├── massimo-embedding/                    # Weighted segmented embedding notebooks (title/abstract/metadata) and evaluation results
├── andrew-embedding/                     # HGT graph neural network training, embedding export, query path, and autoresearch LLM agent
├── audrey-embedding/                     # Hyperbolic Poincaré-ball retrieval, schema migration, and index helpers
├── evaluation/                           # Benchmark queries, validators, LLM judges, metrics, reports, graphs
├── design-system/                        # Design notes for search/results UI
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment variable template
└── README.md                             # Project overview
```

---

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file at the repo root:

```bash
POSTGRES_CONN_STRING="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"
HF_TOKEN="..."
```

Additional variables are required for specific workflows:

```bash
NOMIC_API_KEY="..."   # required for regenerating Nomic embeddings
S2_API_KEY="..."      # required for Semantic Scholar ingest/enrichment scripts
OPENAI_API_KEY="..."  # required for LLM-as-judge evaluation
```

For a fresh database setup, run the following in order after setting env vars:

```bash
python database/init.py        # create the papers table and indexes
python database/ingest.py      # fetch ~25,000 CS papers (requires S2_API_KEY)
python database/enrich.py      # fetch author and reference data (requires S2_API_KEY)
psql "$POSTGRES_CONN_STRING" -f audrey-embedding/migrations/001_init_hyperbolic.sql
```

The app and evaluator expect the `papers` table to be populated and reachable through `POSTGRES_CONN_STRING`.

---

## Running the App

From the repo root:

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://localhost:8000
```

The embedding model is warmed up on startup, so the first launch can be slower than later requests.

---

## Running Evaluation

The committed four-model comparison already exists under:

```text
evaluation/real_four_model_comparison/report_openai/
evaluation/real_four_model_comparison/presentation_graphs/
```

To rerun the full evaluation with the committed normalized inputs:

```bash
python -m evaluation.run_full_evaluation \
  --metadata-source postgres \
  --queries evaluation/benchmark_queries.jsonl \
  --runs \
    evaluation/real_four_model_comparison/inputs/andrew.normalized.jsonl \
    evaluation/real_four_model_comparison/inputs/audrey.normalized.jsonl \
    evaluation/real_four_model_comparison/inputs/massimo.normalized.jsonl \
    evaluation/real_four_model_comparison/inputs/nomic.normalized.jsonl \
  --paper-cache evaluation/real_four_model_comparison/cache/openai_paper_judgments.jsonl \
  --range-cache evaluation/real_four_model_comparison/cache/openai_range_judgments.jsonl \
  --win-rate-cache evaluation/real_four_model_comparison/cache/openai_win_rate_judgments.jsonl \
  --output-dir evaluation/real_four_model_comparison/report_openai \
  --judge-provider openai
```

The report outputs include:

- `summary.md`
- `overall_summary.csv`
- `query_type_summary.csv`
- `pairwise_preferences.csv`
- `win_rate_summary.csv`
- `per_query_metrics.csv`
- presentation graphs in `.png` and `.svg` format

---

## Implemented Components

- A focused research-paper search app for studying how paper representation affects top-10 retrieval.
- A structured 25,000-paper Semantic Scholar Computer Science corpus stored in Postgres with `pgvector`.
- Multiple retrieval backends: baseline Nomic title + abstract search, weighted segmented retrieval, heterogeneous graph / HGT refinement, and hyperbolic Poincaré-ball retrieval.
- A FastAPI web app with semantic search, subject/year filters, sorting, paper detail pages, related-paper retrieval, author lookup, and citation graph exploration.
- A 100-query benchmark comparing retrieval backends on shared top-10 outputs.
- An LLM-as-judge evaluation layer reporting Precision@10, nDCG@10, mean relevance, reciprocal rank, pooled relevant recall, win rates, and coverage diagnostics.
- Four-model comparison reports and presentation-ready evaluation graphs.

Key limitations:

- The corpus is intentionally small relative to full scholarly search: 25,000 papers and limited to one primary field: Computer Science.
- The date range is limited to 2021–2026.
- Metadata quality varies by paper.
- The LLM judge gives a scalable relevance signal, but it is not a human gold label.
- Search quality depends on more than embeddings; reranking, intent detection, and diversity control remain important.

---

## Contributors

- [Massimo Daul](https://github.com/MassimoDaul)
- [Benny Yuan](https://github.com/bennyyuan12)
- [Andrew Jiang](https://github.com/minghanminghan)
- [Audrey McKnight](https://github.com/audrey-mcknight)
- [Rushil Johal](https://github.com/rushil-johal)
