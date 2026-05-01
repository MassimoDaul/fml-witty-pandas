# andrew-embedding

A heterogeneous graph neural network (HGT) that learns 128-dimensional paper embeddings by combining semantic signals from Nomic text embeddings with structural signals from the publication graph (venue, field-of-study edges).

---

## Rationale

Nomic embeddings capture what a paper says. The GNN augments that with where a paper lives in the research landscape — venue, field, and which other papers are semantically adjacent. Contrastive training on KNN edges pulled from the Nomic space encourages the GNN to preserve semantic neighbourhoods while allowing structural co-publication signals to reshape the geometry.

The output embeddings are 128-dim (vs. Nomic's 384-dim), which reduces storage cost and query latency without sacrificing much recall at the scales we operate at (~25k papers).

---

## Files

### `train.py`

Builds the heterogeneous graph, defines the model, and runs training.

**Graph construction**
- Fetches all Nomic embeddings from the `nomic` column.
- Builds KNN edges (k=30) from cosine similarity over Nomic vectors — these encode semantic proximity as graph structure.
- Adds `published_in` (paper→venue) and `has_field` (paper→field) edges from paper metadata.
- Venue and field node features are 384-dim Nomic-averaged embeddings (L2-normalised mean of member paper vectors) rather than random initialisations.
- Performs an 80/20 edge split: all edges are used for message passing; only the train split contributes to the contrastive loss; the val split is held out for loss monitoring.

**Model: `PaperGNN`**
Two-branch architecture:
- **GNN branch**: projects all node types to 512-dim, runs three HGTConv layers (8 attention heads each), then projects paper nodes to 128-dim (`gnn_out`).
- **Base branch**: directly projects the 384-dim Nomic embedding to 128-dim (`paper_proj`).
- Final output: `α * gnn_out + (1-α) * base`, where α is a learnable sigmoid-gated parameter (initialised at ≈0.70). The base branch regularises training and serves as the query encoder at inference time (see `query.py`).

**Training**
- Per-epoch loop runs `STEPS_PER_EPOCH=10` inner gradient steps, each sharing one GNN forward pass.
- KNN contrastive loss uses hard negative mining: `NEG_POOL_SIZE=40` random candidates are sampled per anchor; the hardest (highest cosine similarity) is chosen as the negative.
- A field-level contrastive term (same-field paper pairs as positives) is added at weight `FIELD_LOSS_WEIGHT=0.5`.
- Temperature τ=0.05, Adam LR=1e-3, 50 epochs, batch size 1024.
- Checkpoint saved when both train loss and val loss improve simultaneously.
- Recall@20 and current α value logged every 5 epochs.

**Requirements**: CUDA GPU. The training script asserts `torch.cuda.is_available()` at startup.

```bash
python andrew-embedding/train.py
```

---

### `embed.py`

Exports trained embeddings to the database. Loads a checkpoint, runs full-graph inference, and upserts the resulting (N, 128) matrix to the `andrew` column via `database/utils.upsert_embeddings`. Rebuilds the IVF index afterward.

Run this once after training completes (or whenever the checkpoint is updated).

The `--autoresearch` flag writes to the `autoresearch_new` column instead and loads the checkpoint from `autoresearch/weights/`.

```bash
python andrew-embedding/embed.py
python andrew-embedding/embed.py --checkpoint weights/my_checkpoint.pt
python andrew-embedding/embed.py --autoresearch True
```

---

### `query.py`

Query encoder for the `andrew` column at inference time.

**Problem**: the HGTConv layers require the full paper graph, so they cannot embed an unseen text query. `paper_proj` (the base branch) is the solution — it is a trained `Linear(384, 128)` that maps Nomic vectors into Andrew's latent space. It was optimised under the same contrastive loss as the GNN and does not require graph context.

**Pipeline**:
1. Encode query text with `nomic-ai/nomic-embed-text-v1.5` (HuggingFace, `search_query:` prefix).
2. Load `paper_proj` weights directly from the checkpoint state dict — no need to instantiate the full model, and no CUDA requirement.
3. Apply the projection and L2-normalise → 128-dim query vector.
4. Run `search_similar` against the `andrew` column.

**Known limitation**: stored document vectors blend the GNN branch (weight α) with `paper_proj` (weight 1-α), where α is the learned blend parameter, while query vectors come from `paper_proj` alone. Cosine distances between query and document are therefore slightly lower than paper-to-paper distances. Ranking order is not affected.

```bash
python andrew-embedding/query.py "attention mechanisms in transformers"
python andrew-embedding/query.py "graph neural networks" --k 20 --nprobe 30
python andrew-embedding/query.py "contrastive learning" --checkpoint weights/checkpoint_best.pt
```

---

### `weights/`

Directory for model checkpoints. Each checkpoint is a `torch.save` dict with keys:

| Key | Description |
|-----|-------------|
| `epoch` | Epoch at which this checkpoint was saved |
| `model_state_dict` | Full model weights |
| `optimizer_state_dict` | Adam state for resuming training |
| `train_loss` | Training loss at save time |
| `val_loss` | Validation loss at save time |
| `corpus_ids` | Ordered list of corpus IDs matching the embedding rows |

`checkpoint_best.pt` is the default used by `embed.py` and `query.py`.

---

### `autoresearch/`

Autonomous hyperparameter/architecture search loop powered by an LLM agent.

- **`exec.py`** — orchestrator loop. Drives a litellm-backed agent through up to 20 tool-call steps: read files → hypothesise → modify `train.py` → run experiment → write report.
- **`tools.py`** — tool implementations exposed to the agent: `read_file`, `write_file` (sandboxed to `train.py` and `EXPERIMENTS.md`), `run_experiment` (chains `train.py → embed.py → eval.py` with `--autoresearch True`), `create_report` (appends structured results to `EXPERIMENTS.md`).
- **`EXPERIMENTS.md`** — append-only log of every autoresearch iteration.
- **`weights/checkpoint_best.pt`** — best checkpoint produced by autoresearch runs (separate from the main `weights/` checkpoint).

Requires `LITELLM_MODEL` and `LITELLM_API_KEY` env vars in `.env`.

```bash
cd andrew-embedding/autoresearch
python exec.py
```

---

### `evaluation/generate_andrew_results.py`

Runs all benchmark queries through the Andrew search pipeline and writes results to a JSONL file for offline evaluation.

```bash
python andrew-embedding/evaluation/generate_andrew_results.py
python andrew-embedding/evaluation/generate_andrew_results.py \
    --queries evaluation/benchmark_queries.jsonl \
    --output submissions/andrew-results.jsonl --k 10 --nprobe 25
```

---

## Typical workflow

```
# 1. Populate nomic column first (required — andrew uses nomic as input features)
python nomic-embedding/embed.py

# 2. Train the GNN
python andrew-embedding/train.py

# 3. Export embeddings to the database
python andrew-embedding/embed.py

# 4. Query
python andrew-embedding/query.py "your query here"

# 5. (Optional) Run benchmark evaluation
python andrew-embedding/evaluation/generate_andrew_results.py

# 6. (Optional) Run autonomous hyperparameter search
cd andrew-embedding/autoresearch && python exec.py
```

---

## Config reference

| Constant | Value | Description |
|----------|-------|-------------|
| `HIDDEN_DIM` | 512 | HGTConv hidden dimension |
| `OUT_DIM` | 128 | Output embedding dimension (must match DB schema) |
| `NUM_HEADS` | 8 | Attention heads per HGTConv layer |
| `NUM_LAYERS` | 3 | Number of HGTConv layers |
| `LR` | 1e-3 | Adam learning rate |
| `EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 1024 | Contrastive pairs per step |
| `TAU` | 0.05 | Contrastive temperature |
| `K_NEIGHBORS` | 30 | KNN edges per paper (from Nomic cosine similarity) |
| `STEPS_PER_EPOCH` | 10 | Inner gradient steps per epoch (share one GNN forward pass) |
| `FIELD_LOSS_WEIGHT` | 0.5 | Weight of the field-level contrastive term |
| `NEG_POOL_SIZE` | 40 | Hard negative candidate pool size per anchor |
