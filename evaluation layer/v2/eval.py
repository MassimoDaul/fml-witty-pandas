"""
Intrinsic evaluation suite: Nomic (384-dim) vs Andrew GNN (128-dim).

Tests — all use paper-to-paper retrieval (stored embedding as query vec):
  1. Author Precision@k      — ground-truth pairs from eval_pairs (pair_type='author')
  2. Coupling Precision@k    — ground-truth pairs from eval_pairs (pair_type='coupling')
  3. Field Jaccard@k         — semantic precision proxy
  4. Venue Precision@k       — structural signal proxy
  5. Silhouette by field      — global cluster cohesion

Composite scalar (for autoresearch loop):
  0.40 * author_precision
  0.20 * coupling_precision
  0.25 * field_jaccard
  0.10 * venue_precision
  0.05 * (silhouette + 1) / 2   (normalized to [0,1]; omitted if NaN)

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

# ── Config ────────────────────────────────────────────────────────────────────

K            = 10
K_PAIR       = 20
SAMPLE_N     = 500
NPROBE       = 25    # full recall at nlist=25
SILHOUETTE_N = 5000
SEED         = 42
COLUMNS      = ["nomic", "andrew"]

WEIGHTS = {
    "author_mrr":         0.15,
    "coupling_mrr":       0.35,
    "field_jaccard":      0.30,
    "venue_precision":    0.15,
    "silhouette":         0.05,  # normalized to [0,1] before weighting
}


# ── Metadata ──────────────────────────────────────────────────────────────────

def fetch_metadata(conn) -> dict[str, dict]:
    """Return {corpus_id: {venue, fields}} for papers with both embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, venue, fields_of_study, subfields
            FROM papers
            WHERE nomic IS NOT NULL AND andrew IS NOT NULL
        """)
        return {
            row[0]: {"venue": row[1], "fields": row[2] or [], "subfields": row[3] or []}
            for row in cur.fetchall()
        }


def fetch_eval_pairs(conn, pair_type: str) -> dict[str, set[str]]:
    """Return {query_id: set(target_ids)} for papers with both embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ep.query_id, ep.target_id
            FROM eval_pairs ep
            JOIN papers q ON q.corpus_id = ep.query_id
            JOIN papers t ON t.corpus_id = ep.target_id
            WHERE ep.pair_type = %s
              AND q.nomic IS NOT NULL AND q.andrew IS NOT NULL
              AND t.nomic IS NOT NULL AND t.andrew IS NOT NULL
        """, (pair_type,))
        pairs: dict[str, set[str]] = {}
        for query_id, target_id in cur.fetchall():
            pairs.setdefault(query_id, set()).add(target_id)
    return pairs


# ── Metrics ───────────────────────────────────────────────────────────────────

def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def test_pair_metrics(conn, col: str, queries: list[str],
                      positives: dict[str, set[str]],
                      k: int, nprobe: int) -> tuple[float, float]:
    """
    Precision@k and MRR against a precomputed positive set (author or coupling pairs).
    Only papers that appear in `positives` are used as queries.
    """
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
        
        # calculate precision@k
        hits = sum(1 for r in results if r in pos)
        p_scores.append(hits / len(results))
        
        # calculate MRR
        rank = next((i + 1 for i, r in enumerate(results) if r in pos), 0)
        mrr_scores.append(1.0 / rank if rank > 0 else 0.0)
        
    return float(np.mean(p_scores)) if p_scores else 0.0, float(np.mean(mrr_scores)) if mrr_scores else 0.0


def test_field_jaccard(conn, meta: dict, col: str, queries: list[str],
                       k: int, nprobe: int) -> float:
    """Mean Jaccard of fields_of_study between query paper and its top-k results."""
    vecs = fetch_embeddings(conn, col, queries)
    scores = []
    for cid in queries:
        if cid not in vecs:
            continue
        results = search_similar(conn, col, vecs[cid], k=k + 1, nprobe=nprobe)
        results = [r for r in results if r["corpus_id"] != cid][:k]
        if not results:
            continue
        qfields = meta[cid]["fields"]
        scores.append(np.mean([
            _jaccard(qfields, meta.get(r["corpus_id"], {}).get("fields", []))
            for r in results
        ]))
    return float(np.mean(scores)) if scores else 0.0


def test_venue_precision(conn, meta: dict, col: str, queries: list[str],
                         k: int, nprobe: int) -> float:
    """Fraction of top-k results sharing the query paper's venue."""
    vecs = fetch_embeddings(conn, col, queries)
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
    """Silhouette score using most specific s2-fos-model subfield as cluster label."""
    eligible = [
        (cid, m["subfields"][0])
        for cid, m in meta.items()
        if m["subfields"] and cid in emb_dict
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


def composite(scores: dict[str, float]) -> float:
    """Weighted composite of all metrics into a single scalar."""
    total_weight = 0.0
    value = 0.0
    for key, w in WEIGHTS.items():
        v = scores.get(key, float("nan"))
        if np.isnan(v):
            continue
        # silhouette is [-1, 1] — normalize to [0, 1] before weighting
        if key == "silhouette":
            v = (v + 1.0) / 2.0
        value += w * v
        total_weight += w
    if total_weight == 0:
        return float("nan")
    # rescale so weights sum to 1 even when silhouette is NaN
    return value / total_weight


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",                type=int, default=K)
    parser.add_argument("--k-pair",           type=int, default=K_PAIR)
    parser.add_argument("--sample",           type=int, default=SAMPLE_N)
    parser.add_argument("--nprobe",           type=int, default=NPROBE)
    parser.add_argument("--silhouette-sample",type=int, default=SILHOUETTE_N)
    parser.add_argument('--autoresearch',     type=str, choices=['True', 'False', 'true', 'false'], default='False')
    args = parser.parse_args()

    global COLUMNS
    if args.autoresearch.lower() == 'true':
        COLUMNS = ["andrew", "autoresearch", "autoresearch_new"]
    elif "autoresearch" in COLUMNS:
        COLUMNS.remove("autoresearch")

    random.seed(SEED)
    np.random.seed(SEED)

    conn = get_connection()

    print("Fetching metadata...")
    meta = fetch_metadata(conn)
    print(f"  {len(meta):,} papers with both embeddings")

    print("Fetching eval pairs...")
    author_pairs   = fetch_eval_pairs(conn, "author")
    coupling_pairs = fetch_eval_pairs(conn, "coupling")
    print(f"  author   pairs: {sum(len(v) for v in author_pairs.values()):,} "
          f"({len(author_pairs):,} query papers)")
    print(f"  coupling pairs: {sum(len(v) for v in coupling_pairs.values()):,} "
          f"({len(coupling_pairs):,} query papers)")

    print("Fetching all embeddings for silhouette...")
    emb = {col: fetch_embeddings(conn, col) for col in COLUMNS}

    # Sample query sets once — both columns evaluated on identical papers.
    def _sample(pool, n):
        return random.sample(pool, min(n, len(pool)))

    author_queries   = _sample(list(author_pairs), args.sample)
    coupling_queries = _sample(list(coupling_pairs), args.sample)
    field_queries    = _sample([c for c, m in meta.items() if m["fields"]], args.sample)
    venue_queries    = _sample([c for c, m in meta.items() if m["venue"]],  args.sample)

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

        print(f"  Field Jaccard@{args.k}      (n={len(field_queries)}) ...",    end=" ", flush=True)
        results[col]["field_jaccard"] = test_field_jaccard(
            conn, meta, col, field_queries, args.k, args.nprobe)
        print(f"{results[col]['field_jaccard']:.4f}")

        print(f"  Venue Precision@{args.k}    (n={len(venue_queries)}) ...",    end=" ", flush=True)
        results[col]["venue_precision"] = test_venue_precision(
            conn, meta, col, venue_queries, args.k, args.nprobe)
        print(f"{results[col]['venue_precision']:.4f}")

        print(f"  Silhouette             (n={args.silhouette_sample}) ...",      end=" ", flush=True)
        results[col]["silhouette"] = test_silhouette(meta, emb[col], args.silhouette_sample)
        print(f"{results[col]['silhouette']:.4f}")

        results[col]["composite"] = composite(results[col])

    conn.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    metrics = [
        ("Author Precision@k",   "author_precision",   ""),
        ("Author MRR",           "author_mrr",         f"{WEIGHTS['author_mrr']:.0%}"),
        ("Coupling Precision@k", "coupling_precision", ""),
        ("Coupling MRR",         "coupling_mrr",       f"{WEIGHTS['coupling_mrr']:.0%}"),
        ("Field Jaccard@k",      "field_jaccard",      f"{WEIGHTS['field_jaccard']:.0%}"),
        ("Venue Precision@k",    "venue_precision",    f"{WEIGHTS['venue_precision']:.0%}"),
        ("Silhouette",           "silhouette",         f"{WEIGHTS['silhouette']:.0%}"),
        ("── Composite",         "composite",          ""),
    ]

    w = 24
    header_cols = " ".join([f"{c:>10}" for c in COLUMNS])
    print(f"\n{'=' * (32 + 11 * len(COLUMNS) + 9)}")
    print(f"  {'Metric':<{w}} {'wt':>4} {header_cols} {'winner':>8}")
    print(f"{'=' * (32 + 11 * len(COLUMNS) + 9)}")
    for label, key, wt in metrics:
        vals = [results[c].get(key, float("nan")) for c in COLUMNS]
        valid_vals = [(v, c) for v, c in zip(vals, COLUMNS) if not np.isnan(v)]
        
        if not valid_vals:
            winner = "n/a"
        else:
            max_v = max(v for v, _ in valid_vals)
            winners = [c for v, c in valid_vals if v == max_v]
            winner = "tie" if len(winners) > 1 else winners[0]
                
        vals_s = " ".join([f"{v:>10.4f}" if not np.isnan(v) else "       n/a" for v in vals])
        print(f"  {label:<{w}} {wt:>4} {vals_s} {winner:>8}")
    print(f"{'=' * (32 + 11 * len(COLUMNS) + 9)}")

    if args.autoresearch.lower() == 'true':
        c_auto_new = results["autoresearch_new"].get("composite", float("-nan"))
        c_auto = results["autoresearch"].get("composite", float("-nan"))
        c_andrew = results["andrew"].get("composite", float("-nan"))
        
        if not np.isnan(c_auto) and c_auto_new > c_andrew and c_auto_new > c_auto:
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
