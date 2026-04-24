"""
nomic/embed.py

Populate the `nomic` column with nomic-embed-text-v1.5 embeddings.
Task type: search_document (title + abstract).

Uses the open HuggingFace weights — no Nomic API key required.

Usage:
    python nomic-embedding/embed.py

Requires:
    pip install sentence-transformers einops python-dotenv psycopg2-binary pgvector tqdm numpy
"""

import os
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.utils import (
    build_ivf_indexes,
    drop_ivf_indexes,
    get_connection,
    get_unembedded,
    upsert_embeddings,
)

load_dotenv()

COLUMN = "nomic"
HF_MODEL = "nomic-ai/nomic-embed-text-v1.5"
TASK_PREFIX = "search_document: "
EMBED_BATCH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(HF_MODEL, trust_remote_code=True, device=DEVICE)
        print(f"Model loaded on {DEVICE.upper()}")
    return _model


def _fetch_texts(conn, corpus_ids: list[str]) -> list[tuple[str, str]]:
    """Return [(corpus_id, 'title\nabstract'), ...] for the given ids."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id, title, abstract FROM papers WHERE corpus_id = ANY(%s)",
            (corpus_ids,),
        )
        return [
            (cid, f"{title or ''}\n{abstract or ''}".strip())
            for cid, title, abstract in cur.fetchall()
        ]


def _embed_batches(pairs: list[tuple[str, str]]) -> list[tuple[str, np.ndarray]]:
    model = _get_model()
    id_vector_pairs = []
    for i in tqdm(range(0, len(pairs), EMBED_BATCH), desc="embedding"):
        batch = pairs[i : i + EMBED_BATCH]
        cids  = [p[0] for p in batch]
        texts = [TASK_PREFIX + p[1] for p in batch]
        vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for cid, vec in zip(cids, vecs):
            id_vector_pairs.append((cid, vec.astype(np.float32)))
    return id_vector_pairs


def main() -> None:
    # conn = get_connection()
    # try:
    #     todo = get_unembedded(conn, COLUMN)
    #     if not todo:
    #         print("All papers already have nomic embeddings.")
    #         return

    #     print(f"Papers missing nomic embeddings: {len(todo):,}")
    #     pairs = _fetch_texts(conn, todo)

    #     drop_ivf_indexes(conn, COLUMN)
    #     id_vector_pairs = _embed_batches(pairs)

    #     print(f"Upserting {len(id_vector_pairs):,} embeddings...")
    #     upsert_embeddings(conn, COLUMN, id_vector_pairs)
    #     build_ivf_indexes(conn, COLUMN)
    #     print("Done.")
    # finally:
    #     conn.close()
    from database.utils import get_connection, build_ivf_indexes
    conn = get_connection()
    build_ivf_indexes(conn, COLUMN)
    conn.close()
    print('Done.')


if __name__ == "__main__":
    main()
