"""
Intrinsic evaluation suite: five embedding approaches.

Tests — all use paper-to-paper retrieval (stored embedding as query vec):
  1. Author MRR               — ground-truth pairs from eval_pairs (pair_type='author')
  2. Coupling MRR             — ground-truth pairs from eval_pairs (pair_type='coupling')
  3. Exact Title Match@k      — paper's stored embedding should retrieve itself in top-k

Composite scalar (equal weights, 1/3 each):
  author_mrr + coupling_mrr + title_match

Usage:
    python evaluation/eval.py [--k 10] [--k-pair 20] [--sample 500] [--nprobe 25]
"""

import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from database.utils import get_connection, fetch_embeddings, search_similar

# ── Config ────────────────────────────────────────────────────────────────────

K        = 10
K_PAIR   = 20
SAMPLE_N = 500
NPROBE   = 25
SEED     = 42
COLUMNS  = ["nomic", "andrew", "autoresearch", "massimo_title", "audrey"]

WEIGHTS = {
    "author_mrr":   1 / 3,
    "coupling_mrr": 1 / 3,
    "title_match":  1 / 3,
}


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_eval_pairs(conn, pair_type: str) -> dict[str, set[str]]:
    """Return {query_id: set(target_ids)} for a given pair type."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT query_id, target_id
            FROM eval_pairs
            WHERE pair_type = %s
        """, (pair_type,))
        pairs: dict[str, set[str]] = {}
        for query_id, target_id in cur.fetchall():
            pairs.setdefault(query_id, set()).add(target_id)
    return pairs


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_pair_metrics(conn, col: str, queries: list[str],
                      positives: dict[str, set[str]],
                      k: int, nprobe: int) -> tuple[float, float]:
    """Precision@k and MRR against a precomputed positive set."""
    vecs = fetch_embeddings(conn, col, queries)
    p_scores = []
    mrr_scores = []
    for cid in queries:
        if cid not in vecs or cid not in positives:
            continue
        pos = positives[cid]
        results = search_similar(conn, col, vecs[cid], k=k + 1, nprobe=nprobe)
        results = [r["corpus_id"] for r in results if r["corpus_id"] != cid][:k]
        if not results:
            continue
        hits = sum(1 for r in results if r in pos)
        p_scores.append(hits / len(results))
        rank = next((i + 1 for i, r in enumerate(results) if r in pos), 0)
        mrr_scores.append(1.0 / rank if rank > 0 else 0.0)
    return float(np.mean(p_scores)) if p_scores else 0.0, float(np.mean(mrr_scores)) if mrr_scores else 0.0


def test_exact_title_match(conn, col: str, queries: list[str],
                            nprobe: int, k: int = 10) -> float:
    """Fraction of sampled papers whose stored embedding retrieves itself in top-k."""
    vecs = fetch_embeddings(conn, col, queries)
    hits = 0
    total = 0
    for cid in queries:
        if cid not in vecs:
            continue
        results = search_similar(conn, col, vecs[cid], k=k, nprobe=nprobe)
        if any(r["corpus_id"] == cid for r in results):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def composite(scores: dict[str, float]) -> float:
    """Equal-weighted composite, NaN-safe (skips missing metrics)."""
    total_weight = 0.0
    value = 0.0
    for key, w in WEIGHTS.items():
        v = scores.get(key, float("nan"))
        if np.isnan(v):
            continue
        value += w * v
        total_weight += w
    return value / total_weight if total_weight > 0 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",            type=int, default=K)
    parser.add_argument("--k-pair",       type=int, default=K_PAIR)
    parser.add_argument("--sample",       type=int, default=SAMPLE_N)
    parser.add_argument("--nprobe",       type=int, default=NPROBE)
    parser.add_argument("--autoresearch", type=str,
                        choices=["True", "False", "true", "false"], default="False")
    args = parser.parse_args()

    global COLUMNS
    if args.autoresearch.lower() == "true":
        COLUMNS = ["andrew", "autoresearch", "autoresearch_new"]

    random.seed(SEED)
    np.random.seed(SEED)

    conn = get_connection()

    print("Fetching eval pairs...")
    author_pairs   = fetch_eval_pairs(conn, "author")
    coupling_pairs = fetch_eval_pairs(conn, "coupling")
    print(f"  author   pairs: {sum(len(v) for v in author_pairs.values()):,} "
          f"({len(author_pairs):,} query papers)")
    print(f"  coupling pairs: {sum(len(v) for v in coupling_pairs.values()):,} "
          f"({len(coupling_pairs):,} query papers)")

    print("Fetching paper IDs for title match test...")
    with conn.cursor() as cur:
        cur.execute("SELECT corpus_id FROM papers")
        all_corpus_ids = [row[0] for row in cur.fetchall()]
    print(f"  {len(all_corpus_ids):,} papers total")

    def _sample(pool, n):
        return random.sample(pool, min(n, len(pool)))

    author_queries   = _sample(list(author_pairs),   args.sample)
    coupling_queries = _sample(list(coupling_pairs), args.sample)
    title_queries    = _sample(all_corpus_ids,        args.sample)

    results: dict[str, dict[str, float]] = {col: {} for col in COLUMNS}

    for col in COLUMNS:
        print(f"\n{'-' * 44}")
        print(f"  {col.upper()}")
        print(f"{'-' * 44}")

        print(f"  Author P@{args.k_pair}/MRR  (n={len(author_queries)}) ...", end=" ", flush=True)
        p_score, mrr_score = test_pair_metrics(
            conn, col, author_queries, author_pairs, args.k_pair, args.nprobe)
        results[col]["author_precision"] = p_score
        results[col]["author_mrr"] = mrr_score
        print(f"P={p_score:.4f}, MRR={mrr_score:.4f}")

        print(f"  Coupling P@{args.k_pair}/MRR(n={len(coupling_queries)}) ...", end=" ", flush=True)
        p_score, mrr_score = test_pair_metrics(
            conn, col, coupling_queries, coupling_pairs, args.k_pair, args.nprobe)
        results[col]["coupling_precision"] = p_score
        results[col]["coupling_mrr"] = mrr_score
        print(f"P={p_score:.4f}, MRR={mrr_score:.4f}")

        print(f"  Title Match@{args.k}         (n={len(title_queries)}) ...", end=" ", flush=True)
        results[col]["title_match"] = test_exact_title_match(
            conn, col, title_queries, args.nprobe, args.k)
        print(f"{results[col]['title_match']:.4f}")

        results[col]["composite"] = composite(results[col])

    conn.close()

    # ── Summary table ──────────────────────────────────────────────────────────
    metrics = [
        ("Author Precision@k",   "author_precision",   ""),
        ("Author MRR",           "author_mrr",         f"{WEIGHTS['author_mrr']:.0%}"),
        ("Coupling Precision@k", "coupling_precision", ""),
        ("Coupling MRR",         "coupling_mrr",       f"{WEIGHTS['coupling_mrr']:.0%}"),
        ("Title Match@k",        "title_match",        f"{WEIGHTS['title_match']:.0%}"),
        ("── Composite",         "composite",          ""),
    ]

    w = 24
    header_cols = " ".join([f"{c:>14}" for c in COLUMNS])
    sep_width = 32 + 15 * len(COLUMNS) + 9
    print(f"\n{'=' * sep_width}")
    print(f"  {'Metric':<{w}} {'wt':>4} {header_cols} {'winner':>14}")
    print(f"{'=' * sep_width}")
    for label, key, wt in metrics:
        vals = [results[c].get(key, float("nan")) for c in COLUMNS]
        valid_vals = [(v, c) for v, c in zip(vals, COLUMNS) if not np.isnan(v)]
        if not valid_vals:
            winner = "n/a"
        else:
            max_v = max(v for v, _ in valid_vals)
            winners = [c for v, c in valid_vals if v == max_v]
            winner = "tie" if len(winners) > 1 else winners[0]
        vals_s = " ".join([f"{v:>14.4f}" if not np.isnan(v) else "           n/a" for v in vals])
        print(f"  {label:<{w}} {wt:>4} {vals_s} {winner:>14}")
    print(f"{'=' * sep_width}")

    if args.autoresearch.lower() == "true":
        c_auto_new = results["autoresearch_new"].get("composite", float("nan"))
        c_auto     = results["autoresearch"].get("composite", float("nan"))
        c_andrew   = results["andrew"].get("composite", float("nan"))

        if not np.isnan(c_auto_new) and c_auto_new > c_andrew and c_auto_new > c_auto:
            print("\n[SUCCESS] Autoresearch formulation beat both baselines! Updating database embeddings...")
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("UPDATE papers SET autoresearch = autoresearch_new")
            conn.commit()
            conn.close()
            print("  Successfully copied 'autoresearch_new' to 'autoresearch' in postgres.")
        else:
            print("\n[REJECTED] Autoresearch did not strictly beat both baselines.")


if __name__ == "__main__":
    main()
