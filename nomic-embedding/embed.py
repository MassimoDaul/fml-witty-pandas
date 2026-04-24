"""
nomic/embed.py

Populate the `nomic` column with nomic-embed-text-v1.5 embeddings via the Nomic API.
Task type: search_document (title + abstract).

Usage:
    python nomic-embedding/embed.py
    python nomic-embedding/embed.py --offset 5000 --amount 5000

Requires:
    pip install nomic python-dotenv psycopg2-binary pgvector tqdm numpy
Env vars:
    NOMIC_API_KEY, POSTGRES_CONN_STRING
"""

import argparse
import os
import sys

import numpy as np
from dotenv import load_dotenv
from nomic import embed
import nomic
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

COLUMN     = "nomic"
MODEL      = "nomic-embed-text-v1.5"
EMBED_BATCH = 256  # texts per API call
DB_CHUNK    = 2000  # papers fetched, embedded, and upserted per checkpoint


def _fetch_texts(conn, corpus_ids: list[str]) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id, title, abstract FROM papers WHERE corpus_id = ANY(%s)",
            (corpus_ids,),
        )
        return [
            (cid, f"{title or ''}\n{abstract or ''}".strip())
            for cid, title, abstract in cur.fetchall()
        ]


def _embed_chunk(pairs: list[tuple[str, str]]) -> list[tuple[str, np.ndarray]]:
    id_vector_pairs = []
    for i in range(0, len(pairs), EMBED_BATCH):
        batch = pairs[i : i + EMBED_BATCH]
        cids  = [p[0] for p in batch]
        texts = [p[1] for p in batch]
        result = embed.text(
            texts=texts,
            model=MODEL,
            task_type="search_document",
            dimensionality=384,
        )
        for cid, vec in zip(cids, result["embeddings"]):
            id_vector_pairs.append((cid, np.array(vec, dtype=np.float32)))
    return id_vector_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed papers into the nomic column.")
    parser.add_argument("--offset", type=int, default=0,    help="Skip the first N unembedded papers")
    parser.add_argument("--amount", type=int, default=None, help="Embed at most N papers (default: all)")
    args = parser.parse_args()

    nomic.login(os.environ["NOMIC_API_KEY"])

    conn = get_connection()
    try:
        todo = get_unembedded(conn, COLUMN)
        if not todo:
            print("All papers already have nomic embeddings.")
            return

        todo = todo[args.offset:]
        if args.amount is not None:
            todo = todo[:args.amount]

        if not todo:
            print("No papers to embed after applying --offset/--amount.")
            return

        print(f"Embedding {len(todo):,} papers (offset={args.offset}, amount={args.amount})")
        drop_ivf_indexes(conn, COLUMN)

        for chunk_start in tqdm(range(0, len(todo), DB_CHUNK), desc="chunks"):
            chunk_ids = todo[chunk_start : chunk_start + DB_CHUNK]
            pairs = _fetch_texts(conn, chunk_ids)
            id_vector_pairs = _embed_chunk(pairs)
            upsert_embeddings(conn, COLUMN, id_vector_pairs)

        build_ivf_indexes(conn, COLUMN)
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
