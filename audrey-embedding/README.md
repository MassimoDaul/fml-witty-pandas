# audrey-embedding

Hyperbolic (Poincaré-ball) representation for paper retrieval, supervised by a
two-level hierarchy derived from the data we have on hand.

## Hypothesis

Does hierarchical (hyperbolic) geometry improve retrieval *inside a modern
retrieval pipeline*, especially for exploratory queries? This module tests the
**representation** axis only — encoder is frozen, ranking pipeline is shared
with other contributors.

## Architecture (three-object split)

| Stage           | Space                | Used for                                  |
|-----------------|----------------------|-------------------------------------------|
| Representation  | Poincaré ball P^64   | the actual learned object                 |
| Index           | Euclidean tangent    | pgvector ANN candidate retrieval          |
| Final score     | True Poincaré dist.  | Python rerank of top-N candidates         |

Stored on Andrew's DB as:

| Column              | Type         | Purpose                                              |
|---------------------|--------------|------------------------------------------------------|
| `audrey`            | `halfvec(64)`| tangent-space proxy = `logmap0(x_hyp)` (pgvector ANN)|
| `audrey_hyp`        | `halfvec(64)`| true Poincaré coordinates (rerank only)              |
| `audrey_curvature`  | `real`       | curvature `c` (default 1.0)                          |

Both columns are written in a **single atomic UPDATE** per row, so a crash
mid-chunk can't leave half-written state.

## Supervision: a real two-level hierarchy

The S2 `s2-fos-model` API only delivers ~24 broad fields (Computer Science,
Engineering, Medicine, …) — not the granular taxonomy hyperbolic geometry
benefits from. So we **build** a hierarchy from the data we have:

```
Level 1 (broad)         fields_of_study               20 broad fields
    └─ Level 2 (sub_cluster)  MiniBatchKMeans on nomic vectors        ~7 sub-clusters per field
       (within each broad field)                      → 139 leaf clusters total
```

Built once by `python audrey-embedding/cluster.py`; saved to
`audrey-embedding/hierarchy.npz` (no DB schema change). Sub-clusters average
~180 papers each, so a batch of 256 sees ~1–2 in-batch positives per anchor —
sharp discriminative signal where coarse fields had ~80 (every random pair
shares ≥1 field in this Medicine/CS-skewed corpus).

## v2 design choices

| Decision      | Choice                                                          |
|---------------|-----------------------------------------------------------------|
| Dimension     | 64                                                              |
| Encoder       | freeze nomic (`nomic-embed-text-v1.5`), train projection head only |
| Supervision   | two-level: broad field → in-field nomic cluster                 |
| Loss          | hyperbolic SupCon: strong (sub_cluster) + 0.3·weak (field) + 0.01·norm |
| Curvature     | `c = 1.0` fixed                                                 |
| Hardware      | Mac CPU (projection head + clustering only)                     |

The `0.01·‖v‖²` norm penalty + `×0.1` final-layer init prevent the
boundary-collapse failure mode (without them, the model trivially minimizes
loss by spreading sub-clusters to the ball boundary, where distances are large
regardless of direction).

## Files

| File                                  | Purpose                                                          |
|---------------------------------------|------------------------------------------------------------------|
| `manifold.py`                         | Single source of truth for Poincaré ops (curvature lives here)   |
| `dbio.py`                             | Local DB helpers (halfvec opclass + `vec_to_np`); avoids touching shared `database/utils.py` |
| `cluster.py`                          | Build the field → in-field-cluster hierarchy from frozen nomic   |
| `hierarchy.npz`                       | Output of cluster.py: per-paper (field, sub_cluster) assignments |
| `train.py`                            | Train projection head with hierarchical SupCon                   |
| `embed.py`                            | Apply head to all papers; atomic write of audrey + audrey_hyp    |
| `query.py`                            | Search: tangent ANN -> Poincaré rerank                            |
| `weights/projection_v1.pt`            | Trained checkpoint                                               |
| `migrations/001_init_hyperbolic.sql`  | Adds the three columns; guards against dropping a non-empty audrey |

## Usage

```bash
# 0. install deps (geoopt was added to root requirements.txt)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. migrate schema (DESTRUCTIVE on the audrey column; aborts if non-empty)
psql "$POSTGRES_CONN_STRING" -f audrey-embedding/migrations/001_init_hyperbolic.sql

# 2. build the hierarchy from existing nomic vectors + fields_of_study
python audrey-embedding/cluster.py

# 3. train (Mac is fine — only the projection head is trainable)
python audrey-embedding/train.py --epochs 10 --batch 256

# 4. embed all papers
python audrey-embedding/embed.py

# 5. search
python audrey-embedding/query.py "attention mechanisms in transformers"
```

## Results so far (v2, 25k corpus on Andrew's DB)

- **Loss**: random ≈ 5.55 → strong sub-cluster SupCon final ≈ 3.27 (10 epochs)
- **Geometry**: median Poincaré norm 0.907, max 0.925, 0% on the boundary —
  ball interior is being used (was 100% > 0.9 without the regularizer)
- **Hyperbolic-rerank vs tangent-only diagnostic**: Spearman 0.31–0.50 across
  test queries, top-10 overlap 2–5 of 10. The Poincaré rerank replaces 50–80%
  of what the tangent ANN would return — the geometry is genuinely doing work
  the Euclidean stage can't replicate.

## Scope discipline

All v1/v2 changes live inside `audrey-embedding/`. The only files touched
outside this folder are:

- `database/init.py` — adds the three audrey columns to the canonical schema
- `database/utils.py::EMBEDDING_COLS` — registers `audrey` and `audrey_hyp`
- `requirements.txt` — adds `geoopt`

`database/utils.py::build_ivf_indexes` was *not* modified; this module uses
its own `dbio.build_audrey_ivf_index` because the team helper hardcodes
`vector_cosine_ops` and audrey is `halfvec`. If/when Massimo or Andrew want
that fix upstream, it's a one-line change for them to make.

## Known v2 limitations / phase-3 candidates

- Coarse-field "weak" loss component doesn't drop (~6.2 throughout) — field
  supervision is too noisy to optimize directly. May want to drop or replace.
- Norm regularizer keeps points off the boundary but median norm is still
  0.91; could push interior usage further with a stronger penalty or with a
  tanh-cap on tangent magnitude.
- No entailment-cone loss — the obvious next step for explicit hierarchy
  geometry (sub_cluster ⊂ field).
- No hybrid scoring — combining `-d_Poincaré` with BM25 is the chat's
  strongest framing for "where hyperbolic shines inside a modern pipeline."
- Cluster boundaries are kmeans-derived — could be replaced with hierarchical
  agglomerative clustering for a true tree (rather than two-level flat).
