"""
audrey-embedding/train.py

Train a Euclidean -> Poincaré projection head on top of frozen nomic features,
supervised by Semantic Scholar fields_of_study (papers.fields_of_study).

Pipeline (v1):
    nomic[384]  ->  MLP  ->  v[64]  ->  expmap0  ->  x in P^64

Loss: in-batch hyperbolic InfoNCE.
    For (anchor, positive) pairs sharing >=1 subfield, treat the rest of the
    batch as negatives. Similarity is -d_H (negated Poincaré distance), so
    closer = higher logit. Temperature tau scales the softmax.

Usage:
    python audrey-embedding/train.py
    python audrey-embedding/train.py --epochs 20 --batch 256 --dim 64

Requires:
    pip install torch geoopt psycopg2-binary pgvector python-dotenv tqdm numpy
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.utils import get_connection

# The audrey-embedding/ directory has a hyphen, so it can't be imported as a
# package; add it to sys.path and import manifold.py as a top-level module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold as M  # type: ignore  # noqa: E402
from dbio import vec_to_np  # type: ignore  # noqa: E402

load_dotenv()

WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_CKPT = WEIGHTS_DIR / "projection_v1.pt"
HIERARCHY_PATH = Path(__file__).parent / "hierarchy.npz"


# ── Model ─────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    nomic[384] -> hidden -> dim, then expmap0 to Poincaré ball.

    Final-layer weights are scaled down at init (×0.1) so the early forward
    yields small tangent vectors → Poincaré points near the origin. This is
    the standard recipe for avoiding the well-known hyperbolic-contrastive
    boundary-collapse failure: without it, the model finds a trivial minimum
    by pushing all points to the ball boundary (where distances are large
    regardless of direction).
    """

    def __init__(self, in_dim: int = 384, hidden: int = 128, out_dim: int = M.DIM, c: float = M.CURVATURE):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.mul_(0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (poincare_point, tangent_v).  v exposed for norm regularization."""
        v = self.net(x)
        x_hyp = M.expmap0(v, c=self.c)
        return M.project(x_hyp, c=self.c), v

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper: returns just the Poincaré point."""
        return self.forward(x)[0]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_training_data(conn) -> tuple[list[str], np.ndarray]:
    """Pull (corpus_id, nomic_vector) for papers with nomic + fields_of_study."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, nomic
            FROM papers
            WHERE nomic IS NOT NULL
              AND fields_of_study IS NOT NULL
              AND array_length(fields_of_study, 1) > 0
            ORDER BY corpus_id
        """)
        rows = cur.fetchall()
    corpus_ids = [r[0] for r in rows]
    feats = np.stack([vec_to_np(r[1]) for r in rows])
    return corpus_ids, feats


def load_hierarchy(path: Path, corpus_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load (sub_cluster, field_idx, field_for_cluster) aligned to corpus_ids.

    Both cluster.py and load_training_data() ORDER BY corpus_id, so the order
    matches by construction. We still verify by id to fail loudly on mismatch.

    Returns:
        sub_cluster: (N,) int64 -- leaf-level cluster id per paper
        field_idx:   (N,) int64 -- broad-field id per paper
        field_names: list[str]  -- field name for each field_idx
    """
    h = np.load(path, allow_pickle=True)
    h_ids = list(h["corpus_ids"])
    if h_ids != corpus_ids:
        raise RuntimeError(
            f"hierarchy.npz corpus_ids do not match training query "
            f"(hierarchy={len(h_ids)}, training={len(corpus_ids)}). "
            f"Re-run cluster.py."
        )
    sub_cluster = h["sub_cluster"].astype(np.int64)
    field_for_cluster = list(h["field_for_cluster"])
    # Distinct field names, deterministically ordered
    field_names = sorted(set(field_for_cluster))
    field_to_id = {f: i for i, f in enumerate(field_names)}
    cluster_to_field_id = np.array([field_to_id[field_for_cluster[c]] for c in range(len(field_for_cluster))])
    field_idx = cluster_to_field_id[sub_cluster]
    return sub_cluster, field_idx, field_names


class HierarchyPairDataset(Dataset):
    """
    Each item is (anchor_idx, positive_idx) where both share a sub_cluster.

    Sub-clusters are the LEAF level of the field -> sub_cluster hierarchy
    (built by cluster.py). With ~140 sub-clusters and ~180 papers each, this
    gives a much sharper supervision signal than fields_of_study (24 broad cats).

    Resampled every epoch via resample(seed).
    """

    def __init__(self, sub_cluster: np.ndarray, pairs_per_anchor: int = 4, seed: int = 0):
        cluster_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(sub_cluster.tolist()):
            cluster_to_idx[c].append(i)

        self.anchor_indices: list[int] = []
        self.candidates_by_anchor: list[list[int]] = []
        for i, c in enumerate(sub_cluster.tolist()):
            cands = [j for j in cluster_to_idx[c] if j != i]
            if not cands:
                continue
            self.anchor_indices.append(i)
            self.candidates_by_anchor.append(cands)

        self.pairs_per_anchor = pairs_per_anchor
        self.pairs: list[tuple[int, int]] = []
        self.resample(seed)

    def resample(self, seed: int) -> None:
        rng = random.Random(seed)
        self.pairs = []
        for anchor, cands in zip(self.anchor_indices, self.candidates_by_anchor):
            for _ in range(self.pairs_per_anchor):
                self.pairs.append((anchor, rng.choice(cands)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.pairs[idx]


# ── Loss ──────────────────────────────────────────────────────────────────────

def hyperbolic_supcon(
    x_anchor: torch.Tensor,
    x_positive: torch.Tensor,
    positives_mask: torch.Tensor,
    tau: float,
    c: float,
) -> torch.Tensor:
    """
    Supervised contrastive loss in hyperbolic space (Khosla et al. 2020), adapted
    for our (anchor, positive) batches with field-of-study supervision.

    For batch size B:
      sim[i, j] = -d_H(anchor_i, positive_j) / tau
      P(i) = {j : positives_mask[i, j] is True}   (must include j=i)

      L_i = -1/|P(i)| * sum_{j in P(i)} log( exp(sim[i,j]) / sum_k exp(sim[i,k]) )

    SupCon is the right loss for our regime: with 24 broad fields and ~2.75
    fields/paper, ~24% of in-batch pairs share >=2 fields (i.e. are true semantic
    matches). InfoNCE with one labeled positive treats all ~60 of those as
    negatives and gets confused. SupCon uses them all as positives instead.
    """
    B = x_anchor.shape[0]
    a = x_anchor.unsqueeze(1).expand(B, B, -1)
    p = x_positive.unsqueeze(0).expand(B, B, -1)
    d = M.dist(a.reshape(-1, a.shape[-1]), p.reshape(-1, p.shape[-1]), c=c).reshape(B, B)
    logits = -d / tau
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = positives_mask.sum(dim=1).clamp(min=1).float()
    per_row = -((log_prob * positives_mask.float()).sum(dim=1) / n_pos)
    return per_row.mean()


# ── Train ─────────────────────────────────────────────────────────────────────

def train(
    epochs: int,
    batch_size: int,
    lr: float,
    tau: float,
    dim: int,
    pairs_per_anchor: int,
    ckpt_path: Path,
    device: str,
) -> None:
    print(f"[train] device={device} dim={dim} c={M.CURVATURE} tau={tau}")
    conn = get_connection()
    try:
        corpus_ids, feats = load_training_data(conn)
    finally:
        conn.close()
    print(f"[train] loaded {len(corpus_ids):,} papers")

    sub_cluster, field_idx, field_names = load_hierarchy(HIERARCHY_PATH, corpus_ids)
    n_clusters = int(sub_cluster.max() + 1)
    n_fields = len(field_names)
    print(f"[train] hierarchy: {n_fields} broad fields, {n_clusters} sub-clusters")

    feats_t = torch.from_numpy(feats).to(device)
    sub_cluster_t = torch.from_numpy(sub_cluster).to(device)
    field_idx_t = torch.from_numpy(field_idx).to(device)

    dataset = HierarchyPairDataset(sub_cluster, pairs_per_anchor=pairs_per_anchor)
    print(f"[train] {len(dataset):,} (anchor, positive) pairs from same sub-cluster (resampled each epoch)")

    model = ProjectionHead(in_dim=feats.shape[1], out_dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        dataset.resample(seed=epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model.train()
        running = 0.0
        running_strong = 0.0
        running_weak = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for anchor_idx, pos_idx in pbar:
            x_a = feats_t[anchor_idx]
            x_p = feats_t[pos_idx]
            # Hierarchical SupCon positives:
            #   strong (sub_cluster match) = primary positives
            #   weak (same field, different sub_cluster) = optional auxiliary positives
            # Strong is always positive; the diagonal (anchor_i, pos_i) is by
            # construction in the same sub_cluster, so it's covered.
            sc_a = sub_cluster_t[anchor_idx]
            sc_p = sub_cluster_t[pos_idx]
            fi_a = field_idx_t[anchor_idx]
            fi_p = field_idx_t[pos_idx]
            strong_mask = sc_a.unsqueeze(1) == sc_p.unsqueeze(0)
            weak_mask = (fi_a.unsqueeze(1) == fi_p.unsqueeze(0)) & (~strong_mask)
            h_a, v_a = model(x_a)
            h_p, v_p = model(x_p)
            loss_strong = hyperbolic_supcon(h_a, h_p, strong_mask, tau=tau, c=M.CURVATURE)
            loss_weak = hyperbolic_supcon(h_a, h_p, weak_mask | strong_mask, tau=tau, c=M.CURVATURE)
            # Norm penalty: keep tangent vectors small so Poincaré points stay
            # in the ball interior rather than collapsing to the boundary.
            loss_norm = ((v_a ** 2).sum(dim=-1).mean() + (v_p ** 2).sum(dim=-1).mean()) * 0.5
            loss = loss_strong + 0.3 * loss_weak + 0.01 * loss_norm
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            running_strong += loss_strong.item()
            running_weak += loss_weak.item()
            n_batches += 1
            pbar.set_postfix(
                total=f"{running/n_batches:.3f}",
                strong=f"{running_strong/n_batches:.3f}",
                weak=f"{running_weak/n_batches:.3f}",
            )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": feats.shape[1],
            "out_dim": dim,
            "curvature": M.CURVATURE,
        },
        ckpt_path,
    )
    print(f"[train] saved {ckpt_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.1, help="InfoNCE temperature")
    p.add_argument("--dim", type=int, default=M.DIM)
    p.add_argument("--pairs-per-anchor", type=int, default=4)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = p.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        tau=args.tau,
        dim=args.dim,
        pairs_per_anchor=args.pairs_per_anchor,
        ckpt_path=args.ckpt,
        device=args.device,
    )


if __name__ == "__main__":
    main()
