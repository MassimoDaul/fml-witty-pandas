"""Aggregate per-query evaluation metrics into run-level summaries."""

from __future__ import annotations

from typing import Any

__all__ = ["aggregate_run_metrics"]

_METRIC_FIELDS = (
    "precision_at_10",
    "ndcg_at_10",
    "mean_relevance_at_10",
    "mean_relevance_norm_at_10",
    "high_relevance_rate_at_10",
    "weak_or_irrelevant_rate_at_10",
    "relevant_count_at_10",
    "high_relevance_count_at_10",
    "reciprocal_rank_at_10",
    "high_relevance_reciprocal_rank_at_10",
    "exploration_range_raw",
    "exploration_range_norm",
    "broad_summary_score",
    "field_jaccard_mean_at_10",
    "subfield_jaccard_mean_at_10",
    "field_diversity_at_10",
    "subfield_diversity_at_10",
    "field_subfield_diversity_at_10",
    "venue_pairwise_match_rate_at_10",
    "venue_diversity_count_at_10",
    "mean_citation_count_at_10",
    "mean_reference_count_at_10",
    "pooled_candidate_count",
    "pooled_relevant_count",
    "pooled_relevant_recall_at_10",
    "pooled_high_relevance_recall_at_10",
    "pooled_coverage_at_10",
    "unique_paper_count_at_10",
    "unique_relevant_count_at_10",
    "overlap_jaccard_mean_at_10",
)

_QUERY_TYPES = ("broad", "specific")


def aggregate_run_metrics(per_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute macro-averaged summaries for one run.

    Args:
        per_query_rows: Per-query metrics emitted by
            ``compute_query_metrics_for_run``.

    Returns:
        A nested JSON-serializable summary dictionary.

    Raises:
        ValueError: If the input rows are empty or inconsistent.
    """
    if not per_query_rows:
        raise ValueError("per_query_rows must not be empty")

    run_id = _extract_run_id(per_query_rows)

    by_query_type = {
        query_type: _summarize_rows(
            [row for row in per_query_rows if row.get("query_type") == query_type]
        )
        for query_type in _QUERY_TYPES
    }

    return {
        "run_id": run_id,
        "overall": _summarize_rows(per_query_rows),
        "by_query_type": by_query_type,
    }


def _extract_run_id(per_query_rows: list[dict[str, Any]]) -> str:
    """Ensure all metric rows belong to the same run."""
    run_ids = {row.get("run_id") for row in per_query_rows}
    if len(run_ids) != 1:
        raise ValueError(f"per_query_rows must contain exactly one run_id, got {run_ids!r}")
    run_id = next(iter(run_ids))
    if not isinstance(run_id, str):
        raise ValueError("per_query_rows must contain a string run_id")
    return run_id


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    """Compute macro means for a group of per-query rows."""
    summary: dict[str, int | float | None] = {"query_count": len(rows)}
    for field_name in _METRIC_FIELDS:
        values = _collect_numeric_values(rows, field_name)
        summary[field_name] = _mean(values)
    return summary


def _collect_numeric_values(rows: list[dict[str, Any]], field_name: str) -> list[float]:
    """Collect non-None numeric values for one metric field."""
    values: list[float] = []
    for row_index, row in enumerate(rows):
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"rows[{row_index}][{field_name!r}] must be numeric or None, got {value!r}"
            )
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean or None for an empty set."""
    if not values:
        return None
    return sum(values) / len(values)

