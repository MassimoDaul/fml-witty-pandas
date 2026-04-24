# v0

## eval result

  NOMIC
────────────────────────────────────────
  Field Jaccard@10  (n=500) ... 0.5714
  Venue Precision@10 (n=500) ... 0.0146
  Silhouette           (n=5000) ... -0.0134

────────────────────────────────────────
  ANDREW
────────────────────────────────────────
  Field Jaccard@10  (n=500) ... 0.5829
  Venue Precision@10 (n=500) ... 0.0130
  Silhouette           (n=5000) ... -0.0331

══════════════════════════════════════════════════════════
  Metric                        nomic     andrew   winner
══════════════════════════════════════════════════════════
  Field Jaccard@k              0.5714     0.5829   andrew
  Venue Precision@k            0.0146     0.0130    nomic
  Silhouette                  -0.0134    -0.0331    nomic
══════════════════════════════════════════════════════════

# v1

## changes

- `K_NEIGHBORS` 10 → 20: denser KNN graph gives each paper more positive training pairs
- Venue/field node features: replaced random 64-dim with L2-normalised average of member paper Nomic embeddings (384-dim), so venue and field nodes carry real semantic signal into message passing
- Hard negative mining: `sample_batch` now draws a pool of `NEG_POOL_SIZE=10` random candidates per anchor and selects the one with highest cosine similarity as the negative, replacing fully random negatives
- Field-level contrastive term: added `sample_field_batch` which samples same-field paper pairs as positives; loss is `knn_loss + 0.3 * field_loss`, directly supervising field cluster geometry
- Learnable blend weight α: replaced fixed `0.7 * gnn_out + 0.3 * base` with `sigmoid(α) * gnn_out + (1 - sigmoid(α)) * base` where α is an `nn.Parameter` initialised to 0.847 (≈ 0.70); α is logged every 5 epochs
- `STEPS_PER_EPOCH=10` inner gradient steps per epoch sharing one GNN forward pass, giving 500 total gradient steps vs. 50 in v0; PyTorch's saved-activation backward correctly propagates gradients through the retained graph on each step

## eval result

TODO:
- setup autoresearch-style workflow here
    - pick files to lock and stuff for autoresearch to keep iterating over