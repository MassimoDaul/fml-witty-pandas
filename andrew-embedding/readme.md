# Andrew's GNN Embedding Approach

A Graph Neural Network embedding pipeline for research paper retrieval, evaluated against the same 1500-paper closed corpus as `massimo-embedding`.

## Key idea

Massimo's approach embeds each paper in isolation — the vector for paper A depends only on paper A's text.

This pipeline adds a second step: a GNN that propagates information across a semantic similarity graph built over the corpus. After message passing, each paper's embedding reflects not only its own text but also the topics of its nearest semantic neighbors. A paper with a sparse abstract can "borrow" signal from adjacent papers, and papers that cluster around a topic get pulled into tighter, more coherent regions of the embedding space.

## Differences from Massimo's approach

| | Massimo | Andrew |
|---|---|---|
| Embedding input | title, abstract, metadata separately | title + abstract concatenated |
| Corpus structure | Papers treated independently | kNN graph over corpus |
| Neighborhood signal | None | 2-hop GraphSAGE message passing |
| Training | None (pretrained Nomic only) | Link prediction on kNN graph |
| Output embedding | Raw Nomic (768-dim) | GNN-refined (768-dim) |

Both use `nomic-ai/nomic-embed-text-v1.5` as the base text encoder.

## Notebook layout

1. `ss-load-dataset.ipynb` — loads Massimo's `large_joined_sample.csv`, extracts `paperId` from each paper's URL, saves `papers.parquet` and `idx_to_paperid.json`
2. `gnn-embeddings.ipynb` — five phases in one notebook:
   - **Phase 1**: embed all 1500 papers with Nomic → `X.npy` (1500, 768)
   - **Phase 2**: build kNN graph (default K=7) from cosine similarity → `edge_index.pt`
   - **Phase 3**: train 2-layer GraphSAGE with link prediction → `gnn_weights.pt`
   - **Phase 4**: generate final embeddings → `Z.npy`; ablation vs raw Nomic; over-smoothing check
   - **Phase 5**: run 100 benchmark queries → `results/andrew_gnn_v1.jsonl`

## Results

`results/andrew_gnn_v1.jsonl` — submission in the shared evaluation format (`runId`, `queryId`, `results[rank, paperId, score]`).

## Design decisions

**Why kNN graph?** The 1500 papers are reservoir-sampled across all fields and years — expected within-set citation edges ≈ 0. A semantic kNN graph is the only way to get a meaningful graph structure from this corpus without rescraping.

**Why K=7?** Balances connectivity (graph is fully connected) against over-smoothing (papers stay distinguishable). K=5 and K=10 ablations are included in Phase 4.

**Why link prediction?** No labels are available. Link prediction trains the GNN to produce embeddings where graph-adjacent papers are closer in vector space — a natural proxy for topical relatedness.

**Why GraphSAGE?** Inductive, works well on sparse graphs, mean aggregation is stable and interpretable.

## Python Deps

```bash
pip install sentence-transformers torch torch-geometric pandas numpy tqdm
```