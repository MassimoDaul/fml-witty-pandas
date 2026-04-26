# Evaluation Layer

This folder contains the current evaluation harness for comparing paper-search
systems over the shared 25k-paper Postgres corpus.

The evaluator is designed to compare final user-facing top-10 result lists. It
does not inspect model internals, retriever code, embedding logic, or UI output.

## What It Evaluates

For each benchmark query, each submitted model/run returns a ranked top-10 list
of papers. The evaluator:

1. Loads benchmark queries.
2. Loads canonical paper metadata from the remote Postgres corpus.
3. Validates and normalizes submitted run files.
4. Hydrates returned paper IDs with DB metadata.
5. Uses OpenAI API judges for paper relevance, broad-query range, and anonymous
   query-level comparison.
6. Computes absolute, pooled, and diagnostic metrics.
7. Writes CSV, JSON, and Markdown reports.

## Required Environment

Create or update `.env` in the repo root:

```bash
POSTGRES_CONN_STRING="postgresql://USER:PASSWORD@TAILSCALE_HOST:PORT/DB_NAME"
OPENAI_API_KEY="sk-..."
```

The Postgres database must be reachable through Tailscale before running the
evaluator.

Install dependencies:

```bash
./.venv/bin/pip install -r requirements.txt
```

## Canonical Paper IDs

The database canonical paper ID is:

```text
papers.corpus_id
```

Submissions should use `corpus_id` when possible.

For compatibility, the evaluator also accepts Semantic Scholar `paperId` values
when they match `papers.s2_paper_id`; those are normalized internally to
`corpus_id`.

## Benchmark Queries

Current query file:

```text
evaluation/benchmark_queries.jsonl
```

The current file uses Semantic Scholar-style fields:

```json
{"queryId":"q_001","query":"machine learning for healthcare","year":null,"fields":"paperId,title,abstract,authors,year,publicationDate,fieldsOfStudy,s2FieldsOfStudy,url"}
```

The evaluator normalizes this internally to:

```json
{"query_id":"q_001","query_text":"machine learning for healthcare"}
```

The `year` field may still appear in benchmark files for compatibility with
the search pipelines, but year filters are not scored as a standalone
evaluation metric.

If `query_type` is not present, the evaluator currently infers:

- `q_001` through `q_050`: broad
- `q_051` through `q_100`: specific

## Submission Format

Each model should submit one JSONL file. Each line is one query.

Preferred shape:

```json
{"runId":"nomic_baseline","queryId":"q_001","results":[{"rank":1,"paperId":"123456","score":0.91},{"rank":2,"paperId":"789012","score":0.88}]}
```

Rules:

- one row per benchmark query
- same `runId` on every line
- exactly 10 results per query
- ranks must be integers 1 through 10 with no gaps
- no duplicate paper IDs within a query
- returned IDs must exist in the Postgres corpus

Both camelCase and snake_case are accepted:

- `runId` or `run_id`
- `queryId` or `query_id`
- `paperId`, `paper_id`, `corpusId`, or `corpus_id`

## Running A Full Evaluation

Example with three submitted runs:

```bash
./.venv/bin/python -m evaluation.run_full_evaluation \
  --metadata-source postgres \
  --queries evaluation/benchmark_queries.jsonl \
  --runs submissions/run_a.jsonl submissions/run_b.jsonl submissions/run_c.jsonl \
  --paper-cache outputs/cache/paper_judgments.jsonl \
  --range-cache outputs/cache/range_judgments.jsonl \
  --win-rate-cache outputs/cache/win_rate_judgments.jsonl \
  --output-dir outputs/evaluation_report \
  --judge-provider openai \
  --openai-model gpt-5.4-mini \
  --openai-reasoning-effort low
```

The default OpenAI judge model is `gpt-5.4-mini`. Use `gpt-5.4` if you want a
stronger but more expensive judge.

## Caches

The evaluator uses append-only JSONL caches:

- `paper_judgments.jsonl`: paper-level relevance labels
- `range_judgments.jsonl`: broad-query Exploration Range labels
- `win_rate_judgments.jsonl`: anonymous query-level winner judgments

Do not delete caches unless you intentionally want to pay for rejudging.

If judge prompts, judge model, or metric definitions change, use a new cache path
so old judgments are not mixed with new ones.

## Metrics

### Absolute Metrics

These are computed per run and per query:

- `precision_at_10`: fraction of results judged relevant, where relevance is 2 or 3
- `ndcg_at_10`: ranking quality using graded 0-3 relevance
- `mean_relevance_at_10`: average raw relevance score
- `mean_relevance_norm_at_10`: average relevance normalized to 0-1
- `high_relevance_rate_at_10`: fraction of papers with relevance 3
- `weak_or_irrelevant_rate_at_10`: fraction of papers with relevance 0 or 1
- `reciprocal_rank_at_10`: reciprocal rank of first relevant paper

Precision@10 is still reported, but it should not be treated as the main metric
because low precision can reflect limited corpus coverage rather than model
quality.

### Broad-Query Metrics

Broad queries also receive:

- `exploration_range_raw`: OpenAI judge score from 1 to 5
- `exploration_range_norm`: normalized range score
- `broad_summary_score`: legacy composite summary

Use the component metrics, not only the composite.

### Pooled Cross-Run Metrics

These compare models relative to the pool of papers surfaced by all submitted
runs for the same query:

- `pooled_relevant_recall_at_10`
- `pooled_high_relevance_recall_at_10`
- `pooled_coverage_at_10`
- `unique_paper_count_at_10`
- `unique_relevant_count_at_10`
- `overlap_jaccard_mean_at_10`

These are relative to submitted runs only. They are not corpus-wide recall.

### Metadata Diagnostics

Andrew-style metadata diagnostics are included:

- field/subfield diversity
- field/subfield Jaccard coherence
- venue overlap
- citation/reference count averages

These are useful diagnostics, not final user-quality scores.

## Anonymous Win-Rate Layer

For each query, the evaluator gives the OpenAI judge all submitted result sets
anonymized as `A`, `B`, `C`, etc.

The judge sees:

- query text
- query type
- top-10 hydrated paper metadata for each anonymous system
- five support metrics:
  - `ndcg_at_10`
  - `mean_relevance_norm_at_10`
  - `high_relevance_rate_at_10`
  - `pooled_relevant_recall_at_10`
  - `field_subfield_diversity_at_10`

The judge does not see model names, run IDs, embedding names, or internal scores.

The reports include:

- strict win rate
- fractional win rate for ties
- tie count
- no-clear-winner count
- pairwise preference matrix

## Optional Intrinsic Embedding Diagnostics

You can also compute DB-backed intrinsic metrics over embedding columns:

```bash
./.venv/bin/python -m evaluation.run_full_evaluation \
  ... \
  --intrinsic-columns nomic andrew autoresearch autoresearch_new \
  --intrinsic-sample-limit 500
```

This uses `eval_pairs` for author-overlap and bibliographic-coupling retrieval
diagnostics. It is separate from the 100-query benchmark and should not replace
the OpenAI win-rate layer.

## Report Outputs

The output directory contains:

- `per_query_metrics.csv`
- `overall_summary.csv`
- `query_type_summary.csv`
- `summary.md`
- `comparison_summary.json`
- `corpus_coverage.csv`
- `win_rate_per_query.csv`
- `win_rate_summary.csv`
- `win_rate_summary.json`
- `pairwise_preferences.csv`
- `intrinsic_metrics.csv` when intrinsic metrics are enabled
- `intrinsic_metrics.json` when intrinsic metrics are enabled

Start with `summary.md`, then inspect `win_rate_summary.csv`,
`pairwise_preferences.csv`, and `per_query_metrics.csv`.

## Recommended Pilot Procedure

Before running the full 100-query evaluation:

1. Run 2-3 models on 5-10 queries.
2. Use OpenAI judging with fresh cache files.
3. Read the generated rationales in `win_rate_per_query.csv`.
4. Check whether the judge is over-weighting citations, diversity, or support
   metrics.
5. Only then run all 100 queries.

## Current Limitations

- The evaluator does not yet have a built-in `--limit-queries` or dry-run mode.
- Cache invalidation is manual; use new cache paths when prompts or judge models
  change.
- OpenAI judging cost can be significant for all 100 queries and three runs.
- Pooled metrics are relative to submitted systems, not the whole 25k-paper
  corpus.
- Intrinsic embedding diagnostics are useful but are not direct user-query
  quality metrics.
