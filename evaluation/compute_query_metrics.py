"""Compute per-query evaluation metrics for one expanded run."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["compute_query_metrics_for_run"]

_VALID_RELEVANCE_SCORES = {0, 1, 2, 3}


def compute_query_metrics_for_run(
    expanded_run: list[dict[str, Any]],
    paper_judgments: dict[tuple[str, str], dict[str, Any]],
    range_judgments: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute benchmark metrics for one expanded system run.

    Args:
        expanded_run: Expanded run rows from ``expand_run_with_metadata``.
        paper_judgments: Paper-level relevance labels keyed by
            ``(query_id, paper_id)``.
        range_judgments: Broad-query range labels keyed by ``(run_id, query_id)``.

    Returns:
        One metrics row per query in the run.

    Raises:
        ValueError: If required judgment records or query fields are missing.
    """
    metric_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(expanded_run):
        if not isinstance(row, dict):
            raise ValueError(f"expanded_run[{row_index}] must be a dict")

        run_id = _require_string_field(row, "run_id", context=f"expanded_run[{row_index}]")
        query_id = _require_string_field(
            row,
            "query_id",
            context=f"expanded_run[{row_index}]",
        )
        query_type = _require_query_type(
            row.get("query_type"),
            context=f"expanded_run[{row_index}]",
        )
        results = row.get("results")
        if not isinstance(results, list):
            raise ValueError(f"expanded_run[{row_index}]['results'] must be a list")

        relevance_scores: list[int] = []
        for result_index, result in enumerate(results):
            if not isinstance(result, dict):
                raise ValueError(
                    f"expanded_run[{row_index}]['results'][{result_index}] must be a dict"
                )
            paper_id = _require_string_field(
                result,
                "paper_id",
                context=f"expanded_run[{row_index}]['results'][{result_index}]",
            )
            judgment_key = (query_id, paper_id)
            paper_judgment = paper_judgments.get(judgment_key)
            if paper_judgment is None:
                raise ValueError(
                    f"Missing paper judgment for query_id {query_id!r} and paper_id {paper_id!r}."
                )
            relevance_scores.append(
                _require_relevance_score(
                    paper_judgment.get("relevance_score"),
                    context=f"paper_judgments[{judgment_key!r}]",
                )
            )

        precision_at_10 = _compute_precision_at_k(relevance_scores)
        ndcg_at_10 = _compute_ndcg_at_k(relevance_scores)
        mean_relevance_at_10 = _compute_mean_relevance(relevance_scores)
        mean_relevance_norm_at_10 = mean_relevance_at_10 / 3
        high_relevance_rate_at_10 = _compute_rate(relevance_scores, lambda score: score == 3)
        weak_or_irrelevant_rate_at_10 = _compute_rate(
            relevance_scores,
            lambda score: score <= 1,
        )
        relevant_count_at_10 = sum(1 for score in relevance_scores if score >= 2)
        high_relevance_count_at_10 = sum(1 for score in relevance_scores if score == 3)
        reciprocal_rank_at_10 = _compute_reciprocal_rank(relevance_scores, min_score=2)
        high_relevance_reciprocal_rank_at_10 = _compute_reciprocal_rank(
            relevance_scores,
            min_score=3,
        )

        field_jaccard_mean_at_10 = _compute_mean_pairwise_jaccard(
            results,
            field_name="fields_of_study",
            fallback_field_name="categories",
        )
        subfield_jaccard_mean_at_10 = _compute_mean_pairwise_jaccard(
            results,
            field_name="subfields",
        )
        field_diversity_at_10 = _invert_optional_score(field_jaccard_mean_at_10)
        subfield_diversity_at_10 = _invert_optional_score(subfield_jaccard_mean_at_10)
        field_subfield_diversity_at_10 = _mean_optional(
            [field_diversity_at_10, subfield_diversity_at_10]
        )
        venue_pairwise_match_rate_at_10 = _compute_venue_pairwise_match_rate(results)
        venue_diversity_count_at_10 = _compute_unique_text_count(results, "venue")
        mean_citation_count_at_10 = _compute_mean_numeric_field(results, "citation_count")
        mean_reference_count_at_10 = _compute_mean_numeric_field(results, "reference_count")

        exploration_range_raw: int | None = None
        exploration_range_norm: float | None = None
        broad_summary_score: float | None = None

        if query_type == "broad":
            range_key = (run_id, query_id)
            range_judgment = range_judgments.get(range_key)
            if range_judgment is None:
                raise ValueError(
                    f"Missing range judgment for run_id {run_id!r} and query_id {query_id!r}."
                )

            exploration_range_raw = _require_exploration_range_score(
                range_judgment.get("exploration_range_score"),
                context=f"range_judgments[{range_key!r}]",
            )
            exploration_range_norm = (exploration_range_raw - 1) / 4
            broad_summary_score = (0.7 * precision_at_10) + (0.3 * exploration_range_norm)

        metric_rows.append(
            {
                "run_id": run_id,
                "query_id": query_id,
                "query_type": query_type,
                "precision_at_10": precision_at_10,
                "ndcg_at_10": ndcg_at_10,
                "mean_relevance_at_10": mean_relevance_at_10,
                "mean_relevance_norm_at_10": mean_relevance_norm_at_10,
                "high_relevance_rate_at_10": high_relevance_rate_at_10,
                "weak_or_irrelevant_rate_at_10": weak_or_irrelevant_rate_at_10,
                "relevant_count_at_10": relevant_count_at_10,
                "high_relevance_count_at_10": high_relevance_count_at_10,
                "reciprocal_rank_at_10": reciprocal_rank_at_10,
                "high_relevance_reciprocal_rank_at_10": high_relevance_reciprocal_rank_at_10,
                "exploration_range_raw": exploration_range_raw,
                "exploration_range_norm": exploration_range_norm,
                "broad_summary_score": broad_summary_score,
                "field_jaccard_mean_at_10": field_jaccard_mean_at_10,
                "subfield_jaccard_mean_at_10": subfield_jaccard_mean_at_10,
                "field_diversity_at_10": field_diversity_at_10,
                "subfield_diversity_at_10": subfield_diversity_at_10,
                "field_subfield_diversity_at_10": field_subfield_diversity_at_10,
                "venue_pairwise_match_rate_at_10": venue_pairwise_match_rate_at_10,
                "venue_diversity_count_at_10": venue_diversity_count_at_10,
                "mean_citation_count_at_10": mean_citation_count_at_10,
                "mean_reference_count_at_10": mean_reference_count_at_10,
            }
        )

    return metric_rows


def _compute_precision_at_k(relevance_scores: list[int]) -> float:
    """Compute Precision@k using relevance >= 2 as relevant."""
    if not relevance_scores:
        return 0.0
    relevant_count = sum(1 for score in relevance_scores if score >= 2)
    return relevant_count / len(relevance_scores)


def _compute_mean_relevance(relevance_scores: list[int]) -> float:
    """Compute the mean graded relevance score."""
    if not relevance_scores:
        return 0.0
    return sum(relevance_scores) / len(relevance_scores)


def _compute_rate(relevance_scores: list[int], predicate: Any) -> float:
    """Compute the fraction of relevance scores matching a predicate."""
    if not relevance_scores:
        return 0.0
    return sum(1 for score in relevance_scores if predicate(score)) / len(relevance_scores)


def _compute_reciprocal_rank(relevance_scores: list[int], *, min_score: int) -> float:
    """Return reciprocal rank for the first result meeting ``min_score``."""
    for index, score in enumerate(relevance_scores, start=1):
        if score >= min_score:
            return 1 / index
    return 0.0


def _compute_ndcg_at_k(relevance_scores: list[int]) -> float:
    """Compute nDCG@k using standard log-base-2 discounting."""
    if not relevance_scores:
        return 0.0

    dcg = _compute_dcg(relevance_scores)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = _compute_dcg(ideal_scores)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def _compute_dcg(relevance_scores: list[int]) -> float:
    """Compute DCG for graded relevance labels."""
    return sum(
        ((2**score) - 1) / math.log2(rank + 2)
        for rank, score in enumerate(relevance_scores)
    )


def _compute_mean_pairwise_jaccard(
    results: list[dict[str, Any]],
    *,
    field_name: str,
    fallback_field_name: str | None = None,
) -> float | None:
    """Compute mean pairwise Jaccard over list-valued metadata."""
    sets: list[set[str]] = []
    for result in results:
        values = _string_set(result.get(field_name))
        if not values and fallback_field_name is not None:
            values = _string_set(result.get(fallback_field_name))
        if values:
            sets.append(values)

    if len(sets) < 2:
        return None

    scores: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            union = sets[left_index] | sets[right_index]
            if not union:
                continue
            scores.append(len(sets[left_index] & sets[right_index]) / len(union))

    return _mean_optional(scores)


def _compute_venue_pairwise_match_rate(results: list[dict[str, Any]]) -> float | None:
    """Compute pairwise venue agreement among returned papers with venues."""
    venues = [
        venue
        for venue in (_optional_clean_text(result.get("venue")) for result in results)
        if venue
    ]
    if len(venues) < 2:
        return None

    comparisons = 0
    matches = 0
    for left_index in range(len(venues)):
        for right_index in range(left_index + 1, len(venues)):
            comparisons += 1
            if venues[left_index] == venues[right_index]:
                matches += 1
    return matches / comparisons if comparisons else None


def _compute_unique_text_count(results: list[dict[str, Any]], field_name: str) -> int:
    """Count unique non-empty text values for a metadata field."""
    return len(
        {
            text
            for text in (_optional_clean_text(result.get(field_name)) for result in results)
            if text
        }
    )


def _compute_mean_numeric_field(results: list[dict[str, Any]], field_name: str) -> float | None:
    """Compute the mean of a numeric result metadata field."""
    values: list[float] = []
    for result in results:
        value = result.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return _mean_optional(values)


def _string_set(value: object) -> set[str]:
    """Return a deduped string set from list-like metadata."""
    if not isinstance(value, list):
        return set()
    return {
        text
        for text in (_optional_clean_text(item) for item in value)
        if text
    }


def _optional_clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _invert_optional_score(value: float | None) -> float | None:
    """Convert a 0-1 coherence score into a 0-1 diversity score."""
    if value is None:
        return None
    return 1 - value


def _mean_optional(values: list[float | None]) -> float | None:
    """Return the mean of non-None values, or None."""
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _require_string_field(record: dict[str, Any], field_name: str, *, context: str) -> str:
    """Return a required string field."""
    value = record.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{context} is missing a string {field_name!r} field")
    return value


def _require_query_type(value: object, *, context: str) -> str:
    """Validate the query type label."""
    if value not in {"broad", "specific"}:
        raise ValueError(f"{context} has invalid query_type {value!r}")
    return value


def _require_relevance_score(value: object, *, context: str) -> int:
    """Validate a paper relevance score."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in _VALID_RELEVANCE_SCORES:
        raise ValueError(f"{context} has invalid relevance_score {value!r}")
    return value


def _require_exploration_range_score(value: object, *, context: str) -> int:
    """Validate an Exploration Range score."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2, 3, 4, 5}:
        raise ValueError(f"{context} has invalid exploration_range_score {value!r}")
    return value
