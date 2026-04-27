"""
audrey-embedding/query.py

Hyperbolic search pipeline.

    text query
        -> nomic encoder (frozen, same model used in training)
        -> projection head -> Poincaré point q_hyp
        -> logmap0 -> tangent q_tan
        -> pgvector ANN on `audrey` (cosine on tangent) -> top-N candidates
        -> Poincaré rerank using `audrey_hyp`
        -> top-k results

Usage:
    python audrey-embedding/query.py "attention mechanisms in transformers"
    python audrey-embedding/query.py "graph neural networks" --k 20 --candidates 200

Requires:
    pip install sentence-transformers torch geoopt psycopg2-binary pgvector python-dotenv numpy einops
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.utils import get_connection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold as M  # type: ignore  # noqa: E402
from train import ProjectionHead  # type: ignore  # noqa: E402
from dbio import vec_to_np  # type: ignore  # noqa: E402

load_dotenv()

DEFAULT_CKPT = Path(__file__).parent / "weights" / "projection_v1.pt"

HF_MODEL    = "nomic-ai/nomic-embed-text-v1.5"
TASK_PREFIX = "search_query: "
NOMIC_DIM   = 384  # match nomic-embedding/embed.py dimensionality

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

_nomic: SentenceTransformer | None = None
_head: ProjectionHead | None = None


def _get_nomic() -> SentenceTransformer:
    global _nomic
    if _nomic is None:
        _nomic = SentenceTransformer(HF_MODEL, trust_remote_code=True, device=DEVICE)
    return _nomic


def _get_head(ckpt_path: Path = DEFAULT_CKPT) -> ProjectionHead:
    global _head
    if _head is None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        _head = ProjectionHead(in_dim=ckpt["in_dim"], out_dim=ckpt["out_dim"], c=ckpt["curvature"])
        _head.load_state_dict(ckpt["state_dict"])
        _head.eval().to(DEVICE)
    return _head


def encode_query(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_hyp, q_tan) as 1-D float32 arrays."""
    feat = _get_nomic().encode(TASK_PREFIX + text, normalize_embeddings=True)[:NOMIC_DIM]
    feat_t = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_hyp = _get_head().encode(feat_t)
        q_tan = M.logmap0(q_hyp, c=_get_head().c)
    return q_hyp.squeeze(0).cpu().numpy().astype(np.float32), q_tan.squeeze(0).cpu().numpy().astype(np.float32)


def search(query: str, k: int = 10, candidates: int = 100, nprobe: int = 10) -> list[dict]:
    """
    Two-stage search:
      1. Tangent ANN on `audrey` -> top `candidates` by cosine.
      2. Rerank by true Poincaré distance using `audrey_hyp`.
    """
    q_hyp, q_tan = encode_query(query)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET ivfflat.probes = {int(nprobe)}")
            cur.execute(
                """
                SELECT corpus_id, title, audrey <=> %s AS tangent_dist, audrey_hyp
                FROM papers
                WHERE audrey IS NOT NULL AND audrey_hyp IS NOT NULL
                ORDER BY tangent_dist
                LIMIT %s
                """,
                (np.array(q_tan, dtype=np.float32), candidates),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    q_hyp_t = torch.from_numpy(q_hyp).unsqueeze(0)
    cand_hyp = torch.from_numpy(np.stack([vec_to_np(r[3]) for r in rows]))
    d_hyp = M.dist(q_hyp_t.expand_as(cand_hyp), cand_hyp).numpy()

    enriched = [
        {
            "corpus_id": rows[i][0],
            "title": rows[i][1],
            "tangent_dist": float(rows[i][2]),
            "hyperbolic_dist": float(d_hyp[i]),
        }
        for i in range(len(rows))
    ]
    enriched.sort(key=lambda r: r["hyperbolic_dist"])
    return enriched[:k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=100, help="Tangent-ANN candidates before Poincaré rerank")
    p.add_argument("--nprobe", type=int, default=10)
    args = p.parse_args()

    results = search(args.query, k=args.k, candidates=args.candidates, nprobe=args.nprobe)
    for i, r in enumerate(results, 1):
        print(f"{i:>3}. [d_H={r['hyperbolic_dist']:.4f}  d_T={r['tangent_dist']:.4f}] {r['title']}  ({r['corpus_id']})")


if __name__ == "__main__":
    main()
