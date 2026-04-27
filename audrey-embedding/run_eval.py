"""
audrey-embedding/run_eval.py

Run the team's 100-query benchmark through the hyperbolic search pipeline and
emit a JSONL submission compatible with Massimo's format.

Output schema (one JSON object per line):
    {"runId": str, "queryId": str, "results": [{"rank": int, "paperId": str, "score": float}, ...]}

`paperId` is `papers.s2_paper_id` (Massimo's convention).  `score` is the
negated Poincaré distance (higher = closer / more relevant).

Usage:
    python audrey-embedding/run_eval.py
    python audrey-embedding/run_eval.py --candidates 200 --out audrey-embedding/results/audrey_hyperbolic_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.utils import get_connection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold as M  # type: ignore  # noqa: E402
from query import encode_query  # type: ignore  # noqa: E402
from dbio import vec_to_np  # type: ignore  # noqa: E402

load_dotenv()

DEFAULT_BENCH = Path(__file__).parents[1] / "evaluation" / "benchmark_queries.jsonl"
DEFAULT_OUT = Path(__file__).parents[1] / "evaluation" / "audrey-results.jsonl"
DEFAULT_RUN_ID = "audrey"


def search_eval(conn, query: str, k: int, candidates: int, nprobe: int) -> list[tuple[str, float]]:
    """
    Tangent ANN -> Poincaré rerank.  Returns [(corpus_id, score), ...]
    sorted by score descending (higher = closer).

    NOTE: emits `corpus_id` (e.g. "278776303"), the canonical paper identifier
    on Andrew's shared Tailscale Postgres DB. Earlier versions emitted
    `s2_paper_id` (40-char hex), which doesn't resolve against the current
    shared corpus per Benny's 2026-04-27 directive.
    """
    q_hyp, q_tan = encode_query(query)
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {int(nprobe)}")
        cur.execute(
            """
            SELECT corpus_id, audrey_hyp, audrey <=> %s::halfvec AS tan_dist
            FROM papers
            WHERE audrey IS NOT NULL AND audrey_hyp IS NOT NULL
            ORDER BY tan_dist
            LIMIT %s
            """,
            (np.array(q_tan, dtype=np.float32), candidates),
        )
        rows = cur.fetchall()
    if not rows:
        return []
    cand_hyp = torch.from_numpy(np.stack([vec_to_np(r[1]) for r in rows]))
    q_hyp_t = torch.from_numpy(q_hyp).unsqueeze(0).expand_as(cand_hyp)
    d_hyp = M.dist(q_hyp_t, cand_hyp).numpy()
    enriched = sorted(
        [(rows[i][0], -float(d_hyp[i])) for i in range(len(rows))],
        key=lambda r: -r[1],  # higher score first
    )
    return enriched[:k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--k", type=int, default=10, help="Top-k results per query")
    p.add_argument("--candidates", type=int, default=200, help="Tangent-ANN candidates before Poincaré rerank")
    p.add_argument("--nprobe", type=int, default=10)
    args = p.parse_args()

    queries = []
    with args.bench.open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"[eval] {len(queries)} queries from {args.bench}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    t0 = time.time()
    written = 0
    try:
        with args.out.open("w") as out_f:
            for q in tqdm(queries, desc="queries"):
                qid = q["queryId"]
                qtext = q["query"]
                hits = search_eval(conn, qtext, k=args.k, candidates=args.candidates, nprobe=args.nprobe)
                results = [
                    {"rank": i + 1, "paperId": pid, "score": float(score)}
                    for i, (pid, score) in enumerate(hits)
                ]
                out_f.write(json.dumps({
                    "runId": args.run_id,
                    "queryId": qid,
                    "results": results,
                }) + "\n")
                written += 1
    finally:
        conn.close()
    print(f"[eval] wrote {written} rows to {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
