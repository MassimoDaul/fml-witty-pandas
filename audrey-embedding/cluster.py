"""
audrey-embedding/cluster.py

Build a two-level hierarchy from the data we already have:

    Level 1 (broad):    fields_of_study  (24 categories: CS, Engineering, Medicine, ...)
    Level 2 (sub_cluster): MiniBatchKMeans of nomic vectors WITHIN each broad field

Why this is the right supervision:
- Hyperbolic geometry's value comes from representing trees / hierarchies.
- The S2 `s2-fos-model` only delivers ~24 broad categories; there is no
  granular hierarchy in the DB. Latent sub-clusters within each field give us
  a real two-level tree to embed.

Output (saved to audrey-embedding/hierarchy.npz, stays in this folder; no DB schema change):
    corpus_ids: (N,)        the corpus_id for each paper
    field:      (N,)        primary fields_of_study (most common one), or "" if missing
    sub_cluster: (N,)       integer cluster ID, unique across all fields
    field_for_cluster: (K,) the field each sub_cluster belongs to (parent map)

Usage:
    python audrey-embedding/cluster.py
    python audrey-embedding/cluster.py --target-cluster-size 200

Requires: scikit-learn (already a transitive dep of sentence-transformers).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.utils import get_connection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dbio import vec_to_np  # type: ignore  # noqa: E402

load_dotenv()

DEFAULT_OUT = Path(__file__).parent / "hierarchy.npz"


def load_corpus(conn) -> tuple[list[str], np.ndarray, list[list[str]]]:
    """Return (corpus_ids, nomic feats N×384, fields_of_study lists) for papers
    with both nomic and fields_of_study populated."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, nomic, fields_of_study
            FROM papers
            WHERE nomic IS NOT NULL
              AND fields_of_study IS NOT NULL
              AND array_length(fields_of_study, 1) > 0
            ORDER BY corpus_id
        """)
        rows = cur.fetchall()
    corpus_ids = [r[0] for r in rows]
    feats = np.stack([vec_to_np(r[1]) for r in rows])
    fields = [list(r[2]) for r in rows]
    return corpus_ids, feats, fields


def primary_field(field_list: list[str]) -> str:
    """Pick the most common (non-duplicated) field; deterministic tie-break."""
    if not field_list:
        return ""
    counts = Counter(field_list)
    # Sort by (-count, name) for determinism
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def cluster_within_fields(
    feats: np.ndarray,
    primary: list[str],
    target_cluster_size: int = 250,
    min_clusters_per_field: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    For each broad field, run MiniBatchKMeans on its papers' nomic vectors.
    Number of sub-clusters per field = max(min_clusters_per_field, n_papers/target_size).

    Returns:
        sub_cluster: (N,) int  -- cluster IDs, unique across all fields
        field_for_cluster: {cluster_id: field_name}
    """
    N = feats.shape[0]
    sub_cluster = np.full(N, -1, dtype=np.int64)
    field_for_cluster: dict[int, str] = {}
    next_id = 0

    by_field: dict[str, list[int]] = {}
    for i, f in enumerate(primary):
        by_field.setdefault(f, []).append(i)

    for f in sorted(by_field):
        idxs = np.array(by_field[f])
        n = len(idxs)
        k = max(min_clusters_per_field, n // target_cluster_size)
        k = min(k, n)  # can't have more clusters than papers
        if k <= 1:
            sub_cluster[idxs] = next_id
            field_for_cluster[next_id] = f
            next_id += 1
            continue

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(1024, max(256, n // 10)),
            n_init=3,
        )
        labels = km.fit_predict(feats[idxs])
        for local_id in range(k):
            sub_cluster[idxs[labels == local_id]] = next_id
            field_for_cluster[next_id] = f
            next_id += 1

    assert (sub_cluster >= 0).all(), "all papers must be assigned"
    return sub_cluster, field_for_cluster


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-cluster-size", type=int, default=250,
                   help="Aim for ~this many papers per sub-cluster")
    p.add_argument("--min-clusters-per-field", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    print("[cluster] loading corpus...")
    conn = get_connection()
    try:
        corpus_ids, feats, fields = load_corpus(conn)
    finally:
        conn.close()
    print(f"[cluster] {len(corpus_ids):,} papers, feats {feats.shape}")

    primary = [primary_field(fl) for fl in fields]
    print(f"[cluster] {len(set(primary))} distinct primary fields")

    sub_cluster, field_for_cluster = cluster_within_fields(
        feats, primary,
        target_cluster_size=args.target_cluster_size,
        min_clusters_per_field=args.min_clusters_per_field,
        seed=args.seed,
    )

    K = len(field_for_cluster)
    sizes = np.bincount(sub_cluster, minlength=K)
    print(f"[cluster] built {K} sub-clusters")
    print(f"[cluster] sub-cluster sizes: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}")

    # Save
    np.savez(
        args.out,
        corpus_ids=np.array(corpus_ids, dtype=object),
        field=np.array(primary, dtype=object),
        sub_cluster=sub_cluster.astype(np.int64),
        field_for_cluster=np.array(
            [field_for_cluster[i] for i in range(K)], dtype=object
        ),
    )
    print(f"[cluster] saved {args.out}")


if __name__ == "__main__":
    main()
