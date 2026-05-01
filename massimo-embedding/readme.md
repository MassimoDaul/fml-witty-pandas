# Massimo's Embedding Approach

Here are my Jupyter notebooks that show how I explored and created an embedding approach for more accurate retrieval of papers. 
My approach is pretty similar to a baseline of just embedding (title + abstract), but with these differences:

1) I add metadata
2) I use a custom weighted score to combine embeddings from different segments
3) I scan queries and update this weighted score to return more relevant results

Note: When we were using the Arxiv dataset, I scanned the pdf of each paper to extract different sections. But in my exploration, I found that this did not add any more signal than just having the abstract. 
The metadata combined with a good weighted score was more effective.

The layout is:

1) ``ss-build-dataset.ipynb`` this is the script I used to create, match, and index the raw data to be embedded. For consistency and reproducibility
2) ``seg-embeddings-POC.ipynb`` this is my notebook for exploration
3) ``seg-embeddings.ipynb`` this is the final pipeline that embeds and retrives
4) ``results/`` this folder contains the evaluation results from my runs. This can be updated if we want to specificy a different paper set for retrival

---

## Files

| File | Description |
|------|-------------|
| `ss-build-dataset.ipynb` | Builds the Semantic Scholar dataset by streaming the 2026-03-10 SS release, reservoir-sampling ~3000 abstracts, joining on `corpusid`, and outputting `large_joined_sample.parquet/csv` with columns: `corpusid, title, abstract, year, venue, authors, s2fieldsofstudy, citationcount, referencecount, url` |
| `seg-embeddings-POC.ipynb` | Original arXiv exploration using `nomic-ai/nomic-embed-text-v1.5`. Tested segmented fields (title, abstract, comments, PDF-extracted conclusions). Concluded PDF extraction adds no signal over abstract alone; metadata + weighted scores are more effective. |
| `seg-embeddings.ipynb` | Final retrieval pipeline. Embeds title (w=0.30), abstract (w=0.55), metadata (w=0.15) using Nomic v1.5. Implements query-aware weight adjustment. Produces ranked JSONL results. |
| `embedding-upload-MASSIMO.ipynb` | Batches over the `papers` table in PostgreSQL (`postgresql://fml_app:witty-pandas!@100.70.231.127:5432/fml`), generates 384-dim Nomic embeddings (truncated + renormalized from 768), and writes to three pgvector columns: `massimo_title`, `massimo_abstract`, `massimo_metadata`. Processed 25,000 rows at last run. |
| `results/massimo_weighted_semantic_v1.jsonl` | Evaluation output. 100 benchmark queries × 10 ranked results each (1,000 records). Run ID: `massimo_weighted_semantic_v1`. Scores typically in 0.55–0.67 range. |

## Pipeline Flow

```
ss-build-dataset.ipynb
└─► large_joined_sample.parquet
└─► (manual upload to papers table in PostgreSQL)

embedding-upload-MASSIMO.ipynb
└─► massimo_title / massimo_abstract / massimo_metadata columns in DB

seg-embeddings.ipynb (query + retrieval)
└─► results/massimo_weighted_semantic_v1.jsonl
```

## Embedding Model

- **Model:** `nomic-ai/nomic-embed-text-v1.5`
- **Native dim:** 768 → **truncated to 384** and renormalized
- **Prefix:** `search_document:` for indexing, `search_query:` for queries
- **Storage:** pgvector columns in `fml` PostgreSQL database

## Key Design Decisions

- PDF section extraction (via PyMuPDF) was evaluated and abandoned — low coverage and parse errors meant it added no retrieval signal beyond the abstract.
- Three separate embedding columns (title, abstract, metadata) allow flexible weight tuning at query time without re-embedding.
- Reservoir sampling is used during dataset construction for memory-efficient processing of large Semantic Scholar release files.
