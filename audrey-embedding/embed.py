"""
audrey-embedding/embed.py

Apply the trained projection head to every paper's frozen nomic vector.
Writes two columns:
    audrey      = logmap0(x_hyp)   (Euclidean tangent proxy; pgvector ANN target)
    audrey_hyp  = x_hyp            (true Poincaré-ball point; rerank source)

Usage:
    python audrey-embedding/embed.py
    python audrey-embedding/embed.py --offset 5000 --amount 5000

Requires:
    pip install torch geoopt psycopg2-binary pgvector python-dotenv tqdm numpy
Env:
    POSTGRES_CONN_STRING
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg2.extras
from database.utils import (
    drop_ivf_indexes,
    get_connection,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold as M  # type: ignore  # noqa: E402
from train import ProjectionHead  # type: ignore  # noqa: E402
from dbio import build_audrey_ivf_index, vec_to_np  # type: ignore  # noqa: E402

load_dotenv()

DEFAULT_CKPT = Path(__file__).parent / "weights" / "projection_v1.pt"
DB_CHUNK = 2000


def load_model(ckpt_path: Path, device: str) -> ProjectionHead:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ProjectionHead(in_dim=ckpt["in_dim"], out_dim=ckpt["out_dim"], c=ckpt["curvature"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model


def get_papers_to_embed(conn) -> list[str]:
    """Papers with nomic but missing either audrey or audrey_hyp."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id FROM papers
            WHERE nomic IS NOT NULL AND (audrey IS NULL OR audrey_hyp IS NULL)
        """)
        return [row[0] for row in cur.fetchall()]


def upsert_audrey_pair(
    conn,
    triples: list[tuple[str, np.ndarray, np.ndarray]],
    batch_size: int = 500,
) -> None:
    """
    Atomically write (audrey, audrey_hyp) for each (corpus_id, tangent, hyp) triple.

    Single combined UPDATE per row -> no chance of half-written state if a crash
    or interrupt happens between the two columns. Replaces the prior pattern of
    two separate upsert_embeddings() calls (each with its own commit).
    """
    with conn.cursor() as cur:
        for i in range(0, len(triples), batch_size):
            batch = triples[i : i + batch_size]
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE papers SET audrey = %s, audrey_hyp = %s WHERE corpus_id = %s",
                [
                    (np.asarray(tan, dtype=np.float32), np.asarray(hyp, dtype=np.float32), cid)
                    for cid, tan, hyp in batch
                ],
            )
    conn.commit()


def fetch_nomic_chunk(conn, corpus_ids: list[str]) -> list[tuple[str, np.ndarray]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id, nomic FROM papers WHERE corpus_id = ANY(%s) AND nomic IS NOT NULL",
            (corpus_ids,),
        )
        return [(row[0], vec_to_np(row[1])) for row in cur.fetchall()]


def project_chunk(model: ProjectionHead, pairs: list[tuple[str, np.ndarray]], device: str):
    cids = [p[0] for p in pairs]
    feats = torch.from_numpy(np.stack([p[1] for p in pairs])).to(device)
    with torch.no_grad():
        x_hyp = model.encode(feats)
        x_tan = M.logmap0(x_hyp, c=model.c)
    x_hyp_np = x_hyp.cpu().numpy().astype(np.float32)
    x_tan_np = x_tan.cpu().numpy().astype(np.float32)
    return cids, x_hyp_np, x_tan_np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--amount", type=int, default=None)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = p.parse_args()

    print(f"[embed] device={args.device} ckpt={args.ckpt}")
    model = load_model(args.ckpt, args.device)

    conn = get_connection()
    try:
        todo = get_papers_to_embed(conn)
        if not todo:
            print("[embed] All papers already have audrey embeddings.")
            return
        todo = todo[args.offset:]
        if args.amount is not None:
            todo = todo[: args.amount]
        if not todo:
            print("[embed] No papers to embed after applying --offset/--amount.")
            return
        print(f"[embed] embedding {len(todo):,} papers")

        drop_ivf_indexes(conn, "audrey")

        for chunk_start in tqdm(range(0, len(todo), DB_CHUNK), desc="chunks"):
            chunk_ids = todo[chunk_start : chunk_start + DB_CHUNK]
            pairs = fetch_nomic_chunk(conn, chunk_ids)
            if not pairs:
                continue
            cids, x_hyp, x_tan = project_chunk(model, pairs, args.device)
            upsert_audrey_pair(conn, list(zip(cids, x_tan, x_hyp)))

        build_audrey_ivf_index(conn)
        print("[embed] done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
