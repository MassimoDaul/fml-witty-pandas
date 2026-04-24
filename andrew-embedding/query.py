"""
andrew-embedding/query.py

Semantic search over the `andrew` column using a two-stage query encoder:
  1. Nomic embed-text-v1.5 maps query text → 384-dim semantic vector.
  2. The trained paper_proj layer (Linear 384→128) lifts that into Andrew's space.

Why paper_proj and not the full GNN?
The GNN's HGTConv layers require the full paper graph at inference time, so they
cannot process an unseen text query. paper_proj is the fallback branch already
baked into the model (the 0.3 * base term in the forward pass). It was trained
end-to-end under the same contrastive loss, so it projects Nomic vectors into a
128-dim space that is compatible with the stored andrew embeddings.

Limitation: stored document embeddings blend GNN context (0.7) with paper_proj
(0.3), while query embeddings use paper_proj alone. This asymmetry means cosine
distances are slightly deflated compared to paper-to-paper retrieval, but ranking
order is unaffected.

Usage:
    python andrew-embedding/query.py "attention mechanisms in transformers"
    python andrew-embedding/query.py "graph neural networks" --k 20 --nprobe 30
    python andrew-embedding/query.py "contrastive learning" --checkpoint weights/checkpoint_best.pt

Requires:
    pip install sentence-transformers einops torch python-dotenv psycopg2-binary pgvector numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.utils import get_connection, search_similar

load_dotenv()

COLUMN           = "andrew"
HF_MODEL         = "nomic-ai/nomic-embed-text-v1.5"
TASK_PREFIX      = "search_query: "
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH  = str(Path(__file__).resolve().parent / "weights" / "checkpoint_best.pt")

_nomic_model: SentenceTransformer | None = None
_paper_proj:  torch.nn.Linear | None     = None


def _get_nomic() -> SentenceTransformer:
    global _nomic_model
    if _nomic_model is None:
        _nomic_model = SentenceTransformer(HF_MODEL, trust_remote_code=True, device=DEVICE)
    return _nomic_model


def _get_proj(checkpoint_path: str) -> torch.nn.Linear:
    """Extract paper_proj weights from checkpoint without loading the full model."""
    global _paper_proj
    if _paper_proj is None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        proj = torch.nn.Linear(384, 128, bias=True)
        proj.weight = torch.nn.Parameter(ckpt["model_state_dict"]["paper_proj.weight"])
        proj.bias   = torch.nn.Parameter(ckpt["model_state_dict"]["paper_proj.bias"])
        proj.eval()
        _paper_proj = proj
    return _paper_proj


def embed_query(text: str, checkpoint_path: str = CHECKPOINT_PATH) -> np.ndarray:
    """Encode a text query into Andrew's 128-dim embedding space."""
    nomic_vec = _get_nomic().encode(TASK_PREFIX + text, normalize_embeddings=True)
    nomic_t   = torch.tensor(nomic_vec[:384], dtype=torch.float32).unsqueeze(0)

    proj = _get_proj(checkpoint_path)
    with torch.no_grad():
        out = F.normalize(proj(nomic_t).squeeze(0), dim=0)

    return out.numpy().astype(np.float32)


def search(query: str, k: int = 10, nprobe: int = 10,
           checkpoint_path: str = CHECKPOINT_PATH) -> list[dict]:
    vec  = embed_query(query, checkpoint_path)
    conn = get_connection()
    try:
        return search_similar(conn, COLUMN, vec, k=k, nprobe=nprobe)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic paper search via Andrew GNN embeddings.")
    parser.add_argument("query",        help="Natural-language search query")
    parser.add_argument("--k",          type=int, default=10,           help="Results to return (default 10)")
    parser.add_argument("--nprobe",     type=int, default=10,           help="IVF clusters to probe (default 10)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,        help="Path to model checkpoint")
    args = parser.parse_args()

    results = search(args.query, k=args.k, nprobe=args.nprobe, checkpoint_path=args.checkpoint)
    for i, r in enumerate(results, 1):
        print(f"{i:>3}. [{r['dist']:.4f}] {r['title']}  ({r['corpus_id']})")


if __name__ == "__main__":
    main()
