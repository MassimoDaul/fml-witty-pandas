"""DB-backed intrinsic retrieval metrics for embedding columns."""

from __future__ import annotations

import math
import os
import random
from typing import Any

import numpy as np
from dotenv import load_dotenv

DEFAULT_INTRINSIC_COLUMNS = ("nomic", "andrew", "autoresearch", "autoresearch_new")

__all__ = ["DEFAULT_INTRINSIC_COLUMNS", "compute_intrinsic_embedding_metrics"]


def compute_intrinsic_embedding_metrics(
    conn_string: str | None = None,
    *,
    embedding_columns: list[str] | None = None,
    k: int = 10,
    pair_k: int = 20,
    nprobe: int = 25,
    sample_limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Andrew-style intrinsic metrics for populated embedding columns.

    This evaluates stored paper-to-paper retrieval against DB-derived labels and
    metadata. It is a diagnostic track, not the query benchmark win-rate logic.
    """
    load_dotenv()
    connection_string = conn_string or os.environ.get("POSTGRES_CONN_STRING")
    if not connection_string:
        raise ValueError("Intrinsic metrics require POSTGRES_CONN_STRING or --postgres-conn-string.")

    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector
    except ImportError as exc:
        raise ValueError("Intrinsic metrics require psycopg2-binary and pgvector.") from exc

    requested_columns = embedding_columns or list(DEFAULT_INTRINSIC_COLUMNS)
    rng = random.Random(seed)

    conn = psycopg2.connect(connection_string)
    register_vector(conn)
    try:
        available_columns = _available_vector_columns(conn)
        columns = [column for column in requested_columns if column in available_columns]
        populated = _populated_embedding_counts(conn, columns)
        columns = [column for column in columns if populated.get(column, 0) > 0]
        metadata = _fetch_metadata(conn)

        output: dict[str, Any] = {
            "config": {
                "columns": columns,
                "k": k,
                "pair_k": pair_k,
                "nprobe": nprobe,
                "sample_limit": sample_limit,
                "seed": seed,
            },
            "columns": {},
        }

        for column in columns:
            column_metrics: dict[str, Any] = {
                "populated_paper_count": populated[column],
            }
            for pair_type in ("author", "coupling"):
                positives = _fetch_eval_pairs(conn, pair_type, column)
                query_ids = _sample_ids(list(positives), sample_limit, rng)
                prefix = f"{pair_type}_"
                column_metrics.update(
                    {
                        f"{prefix}{metric_name}": value
                        for metric_name, value in _compute_pair_metrics(
                            conn,
                            column,
                            query_ids,
                            positives,
                            k=pair_k,
                            nprobe=nprobe,
                        ).items()
                    }
                )

            metadata_query_ids = _sample_ids(
                [
                    corpus_id
                    for corpus_id, record in metadata.items()
                    if record["fields_of_study"] or record["subfields"] or record["venue"]
                ],
                sample_limit,
                rng,
            )
            column_metrics.update(
                _compute_metadata_metrics(
                    conn,
                    column,
                    metadata,
                    metadata_query_ids,
                    k=k,
                    nprobe=nprobe,
                )
            )
            output["columns"][column] = column_metrics
    finally:
        conn.close()

    return output


def _available_vector_columns(conn: Any) -> set[str]:
    """Return vector columns present on the papers table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'papers'
              AND udt_name = 'vector'
            """
        )
        return {row[0] for row in cur.fetchall()}


def _populated_embedding_counts(conn: Any, columns: list[str]) -> dict[str, int]:
    """Return non-null counts for embedding columns."""
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for column in columns:
            _validate_identifier(column)
            cur.execute(f"SELECT COUNT(*) FROM papers WHERE {column} IS NOT NULL")
            counts[column] = int(cur.fetchone()[0])
    return counts


def _fetch_metadata(conn: Any) -> dict[str, dict[str, Any]]:
    """Fetch metadata needed for field/subfield/venue metrics."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT corpus_id, fields_of_study, subfields, venue
            FROM papers
            """
        )
        return {
            row[0]: {
                "fields_of_study": _clean_list(row[1]),
                "subfields": _clean_list(row[2]),
                "venue": (row[3] or "").strip() if isinstance(row[3], str) else "",
            }
            for row in cur.fetchall()
        }


def _fetch_eval_pairs(conn: Any, pair_type: str, column: str) -> dict[str, dict[str, float]]:
    """Return query_id -> target_id -> weight for one pair type and column."""
    _validate_identifier(column)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ep.query_id, ep.target_id, ep.weight
            FROM eval_pairs ep
            JOIN papers q ON q.corpus_id = ep.query_id
            JOIN papers t ON t.corpus_id = ep.target_id
            WHERE ep.pair_type = %s
              AND q.{column} IS NOT NULL
              AND t.{column} IS NOT NULL
            """,
            (pair_type,),
        )
        positives: dict[str, dict[str, float]] = {}
        for query_id, target_id, weight in cur.fetchall():
            positives.setdefault(query_id, {})[target_id] = float(weight)
        return positives


def _compute_pair_metrics(
    conn: Any,
    column: str,
    query_ids: list[str],
    positives: dict[str, dict[str, float]],
    *,
    k: int,
    nprobe: int,
) -> dict[str, float]:
    """Compute Precision/Recall/MRR/nDCG against positive paper pairs."""
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    embeddings = _fetch_embeddings(conn, column, query_ids)
    for query_id in query_ids:
        query_vec = embeddings.get(query_id)
        query_positives = positives.get(query_id)
        if query_vec is None or not query_positives:
            continue
        results = _search_similar(conn, column, query_vec, query_id, k=k, nprobe=nprobe)
        if not results:
            continue

        result_ids = [result["corpus_id"] for result in results]
        hits = [paper_id for paper_id in result_ids if paper_id in query_positives]
        precision_scores.append(len(hits) / len(result_ids))
        recall_scores.append(len(hits) / len(query_positives))
        mrr_scores.append(_reciprocal_rank(result_ids, set(query_positives)))
        ndcg_scores.append(_weighted_ndcg(result_ids, query_positives, k=k))

    return {
        f"precision_at_{k}": _mean(precision_scores),
        f"recall_at_{k}": _mean(recall_scores),
        f"mrr_at_{k}": _mean(mrr_scores),
        f"ndcg_at_{k}": _mean(ndcg_scores),
        "query_count": len(query_ids),
    }


def _compute_metadata_metrics(
    conn: Any,
    column: str,
    metadata: dict[str, dict[str, Any]],
    query_ids: list[str],
    *,
    k: int,
    nprobe: int,
) -> dict[str, float]:
    """Compute field/subfield Jaccard and venue precision diagnostics."""
    field_scores: list[float] = []
    subfield_scores: list[float] = []
    venue_scores: list[float] = []
    embeddings = _fetch_embeddings(conn, column, query_ids)

    for query_id in query_ids:
        query_vec = embeddings.get(query_id)
        query_meta = metadata.get(query_id)
        if query_vec is None or query_meta is None:
            continue
        results = _search_similar(conn, column, query_vec, query_id, k=k, nprobe=nprobe)
        if not results:
            continue
        result_ids = [result["corpus_id"] for result in results]
        if query_meta["fields_of_study"]:
            field_scores.append(
                _mean(
                    [
                        _jaccard(query_meta["fields_of_study"], metadata.get(result_id, {}).get("fields_of_study", []))
                        for result_id in result_ids
                    ]
                )
            )
        if query_meta["subfields"]:
            subfield_scores.append(
                _mean(
                    [
                        _jaccard(query_meta["subfields"], metadata.get(result_id, {}).get("subfields", []))
                        for result_id in result_ids
                    ]
                )
            )
        if query_meta["venue"]:
            venue_scores.append(
                sum(
                    1
                    for result_id in result_ids
                    if metadata.get(result_id, {}).get("venue") == query_meta["venue"]
                )
                / len(result_ids)
            )

    return {
        f"field_jaccard_at_{k}": _mean(field_scores),
        f"subfield_jaccard_at_{k}": _mean(subfield_scores),
        f"venue_precision_at_{k}": _mean(venue_scores),
        "metadata_query_count": len(query_ids),
    }


def _fetch_embeddings(conn: Any, column: str, corpus_ids: list[str]) -> dict[str, np.ndarray]:
    """Fetch embeddings for selected corpus IDs."""
    if not corpus_ids:
        return {}
    _validate_identifier(column)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT corpus_id, {column}
            FROM papers
            WHERE corpus_id = ANY(%s) AND {column} IS NOT NULL
            """,
            (corpus_ids,),
        )
        return {row[0]: np.array(row[1].tolist(), dtype=np.float32) for row in cur.fetchall()}


def _search_similar(
    conn: Any,
    column: str,
    query_vec: np.ndarray,
    query_id: str,
    *,
    k: int,
    nprobe: int,
) -> list[dict[str, Any]]:
    """Search nearest papers for a stored query vector."""
    _validate_identifier(column)
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {int(nprobe)}")
        cur.execute(
            f"""
            SELECT corpus_id, 1 - ({column} <=> %s::vector) AS similarity
            FROM papers
            WHERE {column} IS NOT NULL AND corpus_id <> %s
            ORDER BY {column} <=> %s::vector
            LIMIT %s
            """,
            (np.array(query_vec, dtype=np.float32), query_id, np.array(query_vec, dtype=np.float32), k),
        )
        return [
            {"corpus_id": row[0], "similarity": float(row[1])}
            for row in cur.fetchall()
        ]


def _weighted_ndcg(result_ids: list[str], positives: dict[str, float], *, k: int) -> float:
    """Compute linear-gain nDCG using eval_pair weights as graded relevance."""
    gains = [positives.get(result_id, 0.0) for result_id in result_ids[:k]]
    ideal_gains = sorted(positives.values(), reverse=True)[:k]
    ideal_dcg = _linear_dcg(ideal_gains)
    if ideal_dcg == 0:
        return 0.0
    return _linear_dcg(gains) / ideal_dcg


def _linear_dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def _reciprocal_rank(result_ids: list[str], positives: set[str]) -> float:
    for rank, result_id in enumerate(result_ids, start=1):
        if result_id in positives:
            return 1 / rank
    return 0.0


def _sample_ids(ids: list[str], sample_limit: int | None, rng: random.Random) -> list[str]:
    ids = sorted(ids)
    if sample_limit is None or len(ids) <= sample_limit:
        return ids
    return rng.sample(ids, sample_limit)


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip() if item is not None else ""
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _validate_identifier(identifier: str) -> None:
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQL identifier {identifier!r}.")
