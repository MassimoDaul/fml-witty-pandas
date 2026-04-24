"""
nomic/query.py

Semantic search over the `nomic` embedding column using nomic-embed-text-v1.5.
Task type: search_query.

Uses the open HuggingFace weights — no Nomic API key required.

Usage:
    python nomic-embedding/query.py "attention mechanisms in transformers"
    python nomic-embedding/query.py "graph neural networks" --k 20 --nprobe 30

Requires:
    pip install sentence-transformers einops python-dotenv psycopg2-binary pgvector numpy
"""

import argparse
import os
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.utils import get_connection, search_similar

load_dotenv()

COLUMN      = "nomic"
HF_MODEL    = "nomic-ai/nomic-embed-text-v1.5"
TASK_PREFIX = "search_query: "

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(HF_MODEL, trust_remote_code=True, device=DEVICE)
    return _model


def embed_query(text: str) -> np.ndarray:
    vec = _get_model().encode(TASK_PREFIX + text, normalize_embeddings=True)
    return vec[:384].astype(np.float32)


def search(query: str, k: int = 10, nprobe: int = 10) -> list[dict]:
    vec  = embed_query(query)
    conn = get_connection()
    try:
        return search_similar(conn, COLUMN, vec, k=k, nprobe=nprobe)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic paper search via nomic embeddings.")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--k",      type=int, default=10, help="Results to return (default 10)")
    parser.add_argument("--nprobe", type=int, default=10, help="IVF clusters to probe (default 10)")
    args = parser.parse_args()

    results = search(args.query, k=args.k, nprobe=args.nprobe)
    for i, r in enumerate(results, 1):
        print(f"{i:>3}. [{r['dist']:.4f}] {r['title']}  ({r['corpus_id']})")


if __name__ == "__main__":
    main()
