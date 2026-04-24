"""
Intrinsic evaluation suite: Nomic (384-dim) vs Andrew GNN (128-dim).

Tests — all use paper-to-paper retrieval (stored embedding as query vec):
  1. Field-of-study Jaccard@k     — semantic precision proxy
  2. Same-venue Precision@k       — structural signal proxy
  3. Silhouette score by field    — global cluster cohesion

Usage:
    python "evaluation layer/v2/eval.py" [--k 10] [--sample 500] [--nprobe 25]
"""

import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import silhouette_score

from database.utils import get_connection, fetch_embeddings, search_similar

# ── Defaults ──────────────────────────────────────────────────────────────────
K = 10
SAMPLE_N = 500
NPROBE = 25          # full recall at nlist=25
SILHOUETTE_N = 5000  # subsample to keep O(n²) tractable
SEED = 42
COLUMNS = ["nomic", "andrew"]


# ── Metadata ──────────────────────────────────────────────────────────────────

def fetch_metadata(conn) -> dict[str, dict]:
    """Return {corpus_id: {venue, fields}} for papers with both embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, venue, fields_of_study
            FROM papers
            WHERE nomic IS NOT NULL AND andrew IS NOT NULL
        """)
        return {
            row[0]: {"venue": row[1], "fields": row[2] or []}
            for row in cur.fetchall()
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _batch_fetch(conn, corpus_ids: list[str], col: str) -> dict[str, np.ndarray]:
    """Fetch embeddings for a specific list of corpus_ids in one query."""
    return fetch_embeddings(conn, col, corpus_ids)


def test_field_jaccard(conn, meta: dict, col: str, queries: list[str],
                       k: int, nprobe: int) -> float:
    """Mean Jaccard of fields_of_study between query paper and its top-k results."""
    vecs = _batch_fetch(conn, queries, col)
    scores = []
    for cid in queries:
        if cid not in vecs:
            continue
        results = search_similar(conn, col, vecs[cid], k=k + 1, nprobe=nprobe)
        results = [r for r in results if r["corpus_id"] != cid][:k]
        if not results:
            continue
        qfields = meta[cid]["fields"]
        j = np.mean([
            _jaccard(qfields, meta.get(r["corpus_id"], {}).get("fields", []))
            for r in results
        ])
        scores.append(j)
    return float(np.mean(scores)) if scores else 0.0


def test_venue_precision(conn, meta: dict, col: str, queries: list[str],
                         k: int, nprobe: int) -> float:
    """Fraction of top-k results sharing the query paper's venue."""
    vecs = _batch_fetch(conn, queries, col)
    scores = []
    for cid in queries:
        if cid not in vecs:
            continue
        results = search_similar(conn, col, vecs[cid], k=k + 1, nprobe=nprobe)
        results = [r for r in results if r["corpus_id"] != cid][:k]
        if not results:
            continue
        qvenue = meta[cid]["venue"]
        hits = sum(1 for r in results if meta.get(r["corpus_id"], {}).get("venue") == qvenue)
        scores.append(hits / len(results))
    return float(np.mean(scores)) if scores else 0.0


def test_silhouette(meta: dict, emb_dict: dict[str, np.ndarray], sample_n: int) -> float:
    """Silhouette score using primary field_of_study as cluster label."""
    eligible = [
        (cid, m["fields"][0])
        for cid, m in meta.items()
        if m["fields"] and cid in emb_dict
    ]
    if len(eligible) < 2:
        return float("nan")

    if len(eligible) > sample_n:
        eligible = random.sample(eligible, sample_n)

    cids, labels = zip(*eligible)
    if len(set(labels)) < 2:
        return float("nan")

    X = np.stack([emb_dict[cid] for cid in cids])
    return float(silhouette_score(X, labels, metric="cosine"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Intrinsic embedding eval: nomic vs andrew")
    parser.add_argument("--k", type=int, default=K, help="top-k for retrieval tests")
    parser.add_argument("--sample", type=int, default=SAMPLE_N, help="query papers to sample")
    parser.add_argument("--nprobe", type=int, default=NPROBE, help="IVF clusters to probe")
    parser.add_argument("--silhouette-sample", type=int, default=SILHOUETTE_N)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    conn = get_connection()

    print("Fetching metadata...")
    meta = fetch_metadata(conn)
    print(f"  {len(meta)} papers with both embeddings")

    print("Fetching all embeddings for silhouette test...")
    emb = {col: fetch_embeddings(conn, col) for col in COLUMNS}

    # Pre-sample query sets once so both columns are evaluated on identical papers.
    field_queries = random.sample(
        [cid for cid, m in meta.items() if m["fields"]],
        min(args.sample, sum(1 for m in meta.values() if m["fields"]))
    )
    venue_queries = random.sample(
        [cid for cid, m in meta.items() if m["venue"]],
        min(args.sample, sum(1 for m in meta.values() if m["venue"]))
    )

    results: dict[str, dict[str, float]] = {col: {} for col in COLUMNS}

    for col in COLUMNS:
        print(f"\n{'─' * 40}")
        print(f"  {col.upper()}")
        print(f"{'─' * 40}")

        print(f"  Field Jaccard@{args.k}  (n={len(field_queries)}) ...", end=" ", flush=True)
        results[col]["field_jaccard"] = test_field_jaccard(
            conn, meta, col, field_queries, args.k, args.nprobe
        )
        print(f"{results[col]['field_jaccard']:.4f}")

        print(f"  Venue Precision@{args.k} (n={len(venue_queries)}) ...", end=" ", flush=True)
        results[col]["venue_precision"] = test_venue_precision(
            conn, meta, col, venue_queries, args.k, args.nprobe
        )
        print(f"{results[col]['venue_precision']:.4f}")

        print(f"  Silhouette           (n={args.silhouette_sample}) ...", end=" ", flush=True)
        results[col]["silhouette"] = test_silhouette(meta, emb[col], args.silhouette_sample)
        print(f"{results[col]['silhouette']:.4f}")

    conn.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    metrics = [
        ("Field Jaccard@k",   "field_jaccard"),
        ("Venue Precision@k", "venue_precision"),
        ("Silhouette",        "silhouette"),
    ]

    w = 24
    print(f"\n{'═' * 58}")
    print(f"  {'Metric':<{w}} {'nomic':>10} {'andrew':>10} {'winner':>8}")
    print(f"{'═' * 58}")
    for label, key in metrics:
        nv = results["nomic"][key]
        av = results["andrew"][key]
        if np.isnan(nv) and np.isnan(av):
            winner = "n/a"
        elif np.isnan(nv):
            winner = "andrew"
        elif np.isnan(av):
            winner = "nomic"
        else:
            winner = "nomic" if nv > av else ("andrew" if av > nv else "tie")
        nv_s = f"{nv:.4f}" if not np.isnan(nv) else "  n/a"
        av_s = f"{av:.4f}" if not np.isnan(av) else "  n/a"
        print(f"  {label:<{w}} {nv_s:>10} {av_s:>10} {winner:>8}")
    print(f"{'═' * 58}")


if __name__ == "__main__":
    main()
