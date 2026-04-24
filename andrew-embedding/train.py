'''
Heterogeneous graph (papers, venues, fields)
Semantic KNN edges from nomic embeddings
HGT (GAT-style) model with contrastive training
Per-epoch checkpointing when both train and val loss improve

Improvements over v1:
- K_NEIGHBORS 10 → 20 for a richer graph
- Venue/field node features: Nomic-averaged 384-dim instead of random 64-dim
- Hard negative mining: pool of NEG_POOL_SIZE candidates per anchor, pick hardest
- Field-level contrastive term weighted at FIELD_LOSS_WEIGHT
- Learnable blend weight α (sigmoid-gated, init ≈ 0.70)
- STEPS_PER_EPOCH inner gradient steps sharing one GNN forward pass per epoch
'''

import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.transforms import ToUndirected
import numpy as np

from database.utils import get_connection, fetch_embeddings

# ----------------------------
# Config
# ----------------------------
assert torch.cuda.is_available(), "CUDA GPU required for training"
DEVICE = "cuda"

HIDDEN_DIM       = 256
OUT_DIM          = 128        # must match vector(128) in DB schema
NUM_HEADS        = 4
NUM_LAYERS       = 2
LR               = 1e-3
EPOCHS           = 50
BATCH_SIZE       = 2048
TAU              = 0.1
K_NEIGHBORS      = 20         # was 10
STEPS_PER_EPOCH  = 10         # inner gradient steps sharing one forward pass
FIELD_LOSS_WEIGHT = 0.3       # weight for field-level contrastive term
NEG_POOL_SIZE    = 10         # candidates per anchor for hard negative mining
ALPHA_INIT       = 0.847      # sigmoid(0.847) ≈ 0.70, matches original fixed blend
CHECKPOINT_PATH  = str(WEIGHTS_DIR / "checkpoint_best.pt")


# ----------------------------
# KNN edge builder
# ----------------------------
def _build_knn_edges(embeddings: np.ndarray, k: int = K_NEIGHBORS, chunk: int = 500) -> np.ndarray:
    N = len(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_norm = (embeddings / norms).astype(np.float32)

    src_list, dst_list = [], []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch = end - start
        sim = emb_norm[start:end] @ emb_norm.T  # (batch, N)

        local_idx = np.arange(batch)
        sim[local_idx, start + local_idx] = -np.inf  # mask self-loops

        top_k = np.argpartition(sim, -k, axis=1)[:, -k:]  # (batch, k) unordered
        global_src = np.repeat(np.arange(start, end), k)
        global_dst = top_k.ravel()
        src_list.append(global_src)
        dst_list.append(global_dst)

    return np.array([np.concatenate(src_list), np.concatenate(dst_list)], dtype=np.int64)


# ----------------------------
# Data loading
# ----------------------------
def load_data():
    conn = get_connection()

    emb_dict = fetch_embeddings(conn, 'nomic')
    corpus_ids = sorted(emb_dict.keys())
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    N = len(corpus_ids)
    paper_embeddings = np.stack([emb_dict[cid] for cid in corpus_ids]).astype(np.float32)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id, venue, fields_of_study FROM papers WHERE corpus_id = ANY(%s)",
            (corpus_ids,),
        )
        rows = cur.fetchall()
    conn.close()

    venue_to_idx: dict[str, int] = {}
    field_to_idx: dict[str, int] = {}
    paper_venue_src, paper_venue_dst = [], []
    paper_field_src, paper_field_dst = [], []

    for corpus_id, venue, fields in rows:
        if corpus_id not in corpus_id_to_idx:
            continue
        pidx = corpus_id_to_idx[corpus_id]
        if venue:
            if venue not in venue_to_idx:
                venue_to_idx[venue] = len(venue_to_idx)
            paper_venue_src.append(pidx)
            paper_venue_dst.append(venue_to_idx[venue])
        for field in (fields or []):
            if field:
                if field not in field_to_idx:
                    field_to_idx[field] = len(field_to_idx)
                paper_field_src.append(pidx)
                paper_field_dst.append(field_to_idx[field])

    num_venues = max(len(venue_to_idx), 1)
    num_fields = max(len(field_to_idx), 1)

    # Venue features: L2-normalised average of member paper Nomic embeddings
    venue_emb = np.zeros((num_venues, 384), dtype=np.float32)
    venue_cnt = np.zeros(num_venues, dtype=np.float32)
    for pidx, vidx in zip(paper_venue_src, paper_venue_dst):
        venue_emb[vidx] += paper_embeddings[pidx]
        venue_cnt[vidx] += 1
    venue_cnt = np.maximum(venue_cnt, 1)
    venue_emb /= venue_cnt[:, None]
    venue_emb /= np.linalg.norm(venue_emb, axis=1, keepdims=True) + 1e-8

    # Field features: L2-normalised average of member paper Nomic embeddings
    field_emb = np.zeros((num_fields, 384), dtype=np.float32)
    field_cnt = np.zeros(num_fields, dtype=np.float32)
    for pidx, fidx in zip(paper_field_src, paper_field_dst):
        field_emb[fidx] += paper_embeddings[pidx]
        field_cnt[fidx] += 1
    field_cnt = np.maximum(field_cnt, 1)
    field_emb /= field_cnt[:, None]
    field_emb /= np.linalg.norm(field_emb, axis=1, keepdims=True) + 1e-8

    # field_to_papers: used for field-level contrastive sampling
    field_to_papers: dict[int, list[int]] = {}
    for pidx, fidx in zip(paper_field_src, paper_field_dst):
        field_to_papers.setdefault(fidx, []).append(pidx)

    paper_venue_edges = (
        np.array([paper_venue_src, paper_venue_dst], dtype=np.int64)
        if paper_venue_src else np.zeros((2, 0), dtype=np.int64)
    )
    paper_field_edges = (
        np.array([paper_field_src, paper_field_dst], dtype=np.int64)
        if paper_field_src else np.zeros((2, 0), dtype=np.int64)
    )

    print(f"Loaded {N} papers | {num_venues} venues | {num_fields} fields")
    print("Building KNN edges...")
    knn_edges = _build_knn_edges(paper_embeddings)

    n_edges = knn_edges.shape[1]
    perm = np.random.permutation(n_edges)
    split = int(0.8 * n_edges)
    train_edges = knn_edges[:, perm[:split]]
    val_edges   = knn_edges[:, perm[split:]]
    print(f"KNN edges: {n_edges} total | {train_edges.shape[1]} train | {val_edges.shape[1]} val")

    return (
        paper_embeddings,
        train_edges,
        val_edges,
        paper_venue_edges,
        paper_field_edges,
        num_venues,
        num_fields,
        corpus_ids,
        venue_emb,
        field_emb,
        field_to_papers,
    )


# ----------------------------
# Build Graph
# ----------------------------
def build_graph():
    (
        paper_embeddings,
        train_edges,
        val_edges,
        paper_venue_edges,
        paper_field_edges,
        num_venues,
        num_fields,
        corpus_ids,
        venue_emb,
        field_emb,
        field_to_papers,
    ) = load_data()

    data = HeteroData()
    data['paper'].x = torch.tensor(paper_embeddings)
    data['venue'].x = torch.tensor(venue_emb)   # 384-dim semantic features
    data['field'].x = torch.tensor(field_emb)   # 384-dim semantic features

    # All KNN edges in graph for message passing; train/val split is for loss only
    all_knn = np.hstack([train_edges, val_edges])
    data['paper', 'similar', 'paper'].edge_index = torch.tensor(all_knn, dtype=torch.long)
    data['paper', 'published_in', 'venue'].edge_index = torch.tensor(paper_venue_edges, dtype=torch.long)
    data['paper', 'has_field', 'field'].edge_index = torch.tensor(paper_field_edges, dtype=torch.long)

    data = ToUndirected()(data)

    train_edge_index = torch.tensor(train_edges, dtype=torch.long)
    val_edge_index   = torch.tensor(val_edges,   dtype=torch.long)

    return data, train_edge_index, val_edge_index, corpus_ids, field_to_papers


# ----------------------------
# Model
# ----------------------------
class PaperGNN(nn.Module):
    def __init__(self, metadata):
        super().__init__()

        self.lin_dict = nn.ModuleDict({
            'paper': nn.Linear(384, HIDDEN_DIM),
            'venue': nn.Linear(384, HIDDEN_DIM),  # semantic features, was 64-dim random
            'field': nn.Linear(384, HIDDEN_DIM),  # semantic features, was 64-dim random
        })

        self.convs = nn.ModuleList([
            HGTConv(in_channels=HIDDEN_DIM, out_channels=HIDDEN_DIM, metadata=metadata, heads=NUM_HEADS)
            for _ in range(NUM_LAYERS)
        ])

        self.out        = nn.Linear(HIDDEN_DIM, OUT_DIM)
        self.paper_proj = nn.Linear(384, OUT_DIM)
        self.alpha      = nn.Parameter(torch.tensor(ALPHA_INIT))  # learned blend weight

    def forward(self, data):
        x_dict = {k: self.lin_dict[k](data[k].x) for k in data.node_types}

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)

        gnn_out = self.out(x_dict['paper'])
        base    = self.paper_proj(data['paper'].x)

        alpha = torch.sigmoid(self.alpha)
        return alpha * gnn_out + (1 - alpha) * base


# ----------------------------
# Sampling
# ----------------------------
def sample_batch(edge_index, z, num_nodes, batch_size):
    """Sample positive pairs from KNN edges; mine hard negatives from a pool."""
    idx = torch.randint(0, edge_index.size(1), (batch_size,), device=edge_index.device)
    src = edge_index[0, idx]
    pos = edge_index[1, idx]

    with torch.no_grad():
        neg_pool = torch.randint(0, num_nodes, (batch_size, NEG_POOL_SIZE), device=z.device)
        # Replace any pool entry that coincides with the true positive
        clash = neg_pool == pos.unsqueeze(1)
        if clash.any():
            neg_pool[clash] = torch.randint(0, num_nodes, (int(clash.sum()),), device=z.device)

        src_emb  = F.normalize(z[src],           dim=1).unsqueeze(1)         # (B, 1, D)
        pool_emb = F.normalize(z[neg_pool.view(-1)], dim=1)                  # (B*P, D)
        pool_emb = pool_emb.view(batch_size, NEG_POOL_SIZE, -1)              # (B, P, D)
        pool_sim = (src_emb * pool_emb).sum(dim=2)                           # (B, P)
        hard_idx = pool_sim.argmax(dim=1)
        neg = neg_pool[torch.arange(batch_size, device=z.device), hard_idx]

    return src, pos, neg


def sample_field_batch(field_to_papers: dict, num_nodes: int, batch_size: int, device: str):
    """Sample same-field positive pairs with random negatives."""
    eligible = [(fidx, papers) for fidx, papers in field_to_papers.items() if len(papers) >= 2]
    if not eligible:
        return None, None, None

    src_list, pos_list = [], []
    for _ in range(batch_size):
        _, papers = random.choice(eligible)
        a, b = random.sample(papers, 2)
        src_list.append(a)
        pos_list.append(b)

    src = torch.tensor(src_list, dtype=torch.long, device=device)
    pos = torch.tensor(pos_list, dtype=torch.long, device=device)
    neg = torch.randint(0, num_nodes, (batch_size,), device=device)
    return src, pos, neg


# ----------------------------
# Loss
# ----------------------------
def contrastive_loss(z, src, pos, neg):
    pos_sim = F.cosine_similarity(z[src], z[pos])
    neg_sim = F.cosine_similarity(z[src], z[neg])
    return -torch.log(
        torch.exp(pos_sim / TAU) / (torch.exp(pos_sim / TAU) + torch.exp(neg_sim / TAU))
    ).mean()


# ----------------------------
# Evaluation
# ----------------------------
def recall_at_k(emb, edge_index, k=20, max_eval=2000):
    emb = emb.detach().cpu().numpy()
    edge_index = edge_index.cpu()

    candidates = list({int(i) for i in edge_index[0].tolist()})
    if len(candidates) > max_eval:
        candidates = random.sample(candidates, max_eval)
    candidates = np.array(candidates)

    sim = emb[candidates] @ emb.T  # (max_eval, N)

    recalls = []
    for local_i, global_i in enumerate(candidates):
        true_neighbors = set(edge_index[1][edge_index[0] == global_i].tolist())
        if not true_neighbors:
            continue
        top_k = set(np.argpartition(sim[local_i], -k)[-k:].tolist())
        recalls.append(len(top_k & true_neighbors) / len(true_neighbors))

    return float(np.mean(recalls)) if recalls else 0.0


# ----------------------------
# Training
# ----------------------------
def train():
    data, train_edge_index, val_edge_index, corpus_ids, field_to_papers = build_graph()
    data = data.to(DEVICE)
    train_edge_index = train_edge_index.to(DEVICE)
    val_edge_index   = val_edge_index.to(DEVICE)

    model     = PaperGNN(data.metadata()).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_train_loss = float('inf')
    best_val_loss   = float('inf')

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            optimizer.zero_grad()
            z = model(data)

            src, pos, neg = sample_batch(train_edge_index, z, z.size(0), BATCH_SIZE)
            knn_loss = contrastive_loss(z, src, pos, neg)

            f_src, f_pos, f_neg = sample_field_batch(
                field_to_papers, z.size(0), BATCH_SIZE // 4, DEVICE
            )
            field_loss = (
                contrastive_loss(z, f_src, f_pos, f_neg)
                if f_src is not None
                else torch.tensor(0.0, device=DEVICE)
            )

            loss = knn_loss + FIELD_LOSS_WEIGHT * field_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / STEPS_PER_EPOCH

        model.eval()
        with torch.no_grad():
            z_eval  = model(data)
            val_neg = torch.randint(0, z_eval.size(0), (val_edge_index.size(1),), device=DEVICE)
            val_loss = contrastive_loss(z_eval, val_edge_index[0], val_edge_index[1], val_neg).item()

        if avg_train_loss < best_train_loss and val_loss < best_val_loss:
            best_train_loss, best_val_loss = avg_train_loss, val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'corpus_ids': corpus_ids,
            }, CHECKPOINT_PATH)
            print(f"  [checkpoint saved epoch={epoch} train={avg_train_loss:.4f} val={val_loss:.4f}]")

        if epoch % 5 == 0:
            alpha  = torch.sigmoid(model.alpha).item()
            recall = recall_at_k(z_eval, train_edge_index, k=20)
            print(
                f"Epoch {epoch:03d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} "
                f"| Recall@20: {recall:.4f} | α={alpha:.3f}"
            )

    return model, data, corpus_ids


# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    model, data, corpus_ids = train()
