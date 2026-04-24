"""Cross-run pooled metrics for benchmark submissions."""

from __future__ import annotations

from typing import Any

__all__ = ["add_pooled_metrics"]


def add_pooled_metrics(
    per_query_by_run: dict[str, list[dict[str, Any]]],
    expanded_by_run: dict[str, list[dict[str, Any]]],
    paper_judgments: dict[tuple[str, str], dict[str, Any]],
    *,
    corpus_size: int | None = None,
) -> dict[str, Any]:
    """Mutate per-query rows with pooled cross-run metrics.

    Pooled metrics use only the union of papers surfaced by submitted runs for
    each query. They should be read as relative coverage over the judged pool,
    not corpus-wide recall.
    """
    results_by_query = _collect_result_sets(expanded_by_run)
    rows_by_run_query = {
        run_id: {str(row.get("query_id")): row for row in rows}
        for run_id, rows in per_query_by_run.items()
    }

    for query_id, run_results in results_by_query.items():
        pooled_papers = set().union(*(set(papers) for papers in run_results.values()))
        relevant_pool = {
            paper_id
            for paper_id in pooled_papers
            if _relevance_score(paper_judgments, query_id, paper_id) >= 2
        }
        high_relevance_pool = {
            paper_id
            for paper_id in pooled_papers
            if _relevance_score(paper_judgments, query_id, paper_id) == 3
        }

        for run_id, papers in run_results.items():
            paper_set = set(papers)
            other_sets = [
                set(other_papers)
                for other_run_id, other_papers in run_results.items()
                if other_run_id != run_id
            ]
            papers_returned_elsewhere = set().union(*other_sets) if other_sets else set()
            row = rows_by_run_query[run_id][query_id]
            row["pooled_candidate_count"] = len(pooled_papers)
            row["pooled_relevant_count"] = len(relevant_pool)
            row["pooled_relevant_recall_at_10"] = _safe_fraction(
                len(paper_set & relevant_pool),
                len(relevant_pool),
            )
            row["pooled_high_relevance_recall_at_10"] = _safe_fraction(
                len(paper_set & high_relevance_pool),
                len(high_relevance_pool),
            )
            row["pooled_coverage_at_10"] = _safe_fraction(len(paper_set), len(pooled_papers))
            row["unique_paper_count_at_10"] = len(paper_set - papers_returned_elsewhere)
            row["unique_relevant_count_at_10"] = len(
                (paper_set - papers_returned_elsewhere) & relevant_pool
            )
            row["overlap_jaccard_mean_at_10"] = _mean_jaccard(paper_set, other_sets)

    return _build_corpus_coverage_summary(expanded_by_run, corpus_size=corpus_size)


def _collect_result_sets(
    expanded_by_run: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[str]]]:
    """Return query_id -> run_id -> ordered canonical paper IDs."""
    by_query: dict[str, dict[str, list[str]]] = {}
    for run_id, rows in expanded_by_run.items():
        for row in rows:
            query_id = str(row.get("query_id"))
            results = row.get("results")
            if not isinstance(results, list):
                continue
            by_query.setdefault(query_id, {})[run_id] = [
                str(result.get("paper_id"))
                for result in sorted(
                    (result for result in results if isinstance(result, dict)),
                    key=lambda item: int(item.get("rank", 0)),
                )
            ]
    return by_query


def _relevance_score(
    paper_judgments: dict[tuple[str, str], dict[str, Any]],
    query_id: str,
    paper_id: str,
) -> int:
    judgment = paper_judgments.get((query_id, paper_id), {})
    value = judgment.get("relevance_score", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _mean_jaccard(base_set: set[str], other_sets: list[set[str]]) -> float | None:
    if not other_sets:
        return None
    scores: list[float] = []
    for other_set in other_sets:
        union = base_set | other_set
        scores.append(len(base_set & other_set) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def _build_corpus_coverage_summary(
    expanded_by_run: dict[str, list[dict[str, Any]]],
    *,
    corpus_size: int | None,
) -> dict[str, Any]:
    """Summarize corpus coverage by run and across all runs."""
    all_returned: set[str] = set()
    by_run: dict[str, dict[str, int | float | None]] = {}

    for run_id, rows in expanded_by_run.items():
        run_papers: set[str] = set()
        for row in rows:
            results = row.get("results")
            if not isinstance(results, list):
                continue
            run_papers.update(
                str(result.get("paper_id"))
                for result in results
                if isinstance(result, dict) and result.get("paper_id")
            )
        all_returned.update(run_papers)
        by_run[run_id] = {
            "unique_papers_returned": len(run_papers),
            "corpus_size": corpus_size,
            "corpus_coverage_rate": _safe_fraction(len(run_papers), corpus_size or 0),
        }

    return {
        "corpus_size": corpus_size,
        "overall_unique_papers_returned": len(all_returned),
        "overall_corpus_coverage_rate": _safe_fraction(len(all_returned), corpus_size or 0),
        "by_run": by_run,
    }
