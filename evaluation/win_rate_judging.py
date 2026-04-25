"""Anonymous query-level LLM win-rate judging."""

from __future__ import annotations

import hashlib
import json
import os
import random
from json import JSONDecodeError
from typing import Any, Callable

JudgeFn = Callable[[dict[str, Any]], dict[str, Any]]

__all__ = [
    "WIN_RATE_METRICS",
    "score_query_win_rate_with_cache",
    "summarize_win_rates",
]

CACHE_VERSION = "win_rate_v1"

# Top-five support metrics shown to the LLM judge. These are intentionally
# model-agnostic, widely interpretable, and drawn from the old metrics, new
# absolute metrics, pooled relative metrics, and Andrew-inspired metadata
# diagnostics.
WIN_RATE_METRICS = (
    {
        "field": "ndcg_at_10",
        "label": "nDCG@10",
        "source": "old_rank_quality",
        "higher_is_better": True,
    },
    {
        "field": "mean_relevance_norm_at_10",
        "label": "Mean graded relevance@10",
        "source": "absolute_llm_relevance",
        "higher_is_better": True,
    },
    {
        "field": "high_relevance_rate_at_10",
        "label": "High relevance rate@10",
        "source": "absolute_llm_relevance",
        "higher_is_better": True,
    },
    {
        "field": "pooled_relevant_recall_at_10",
        "label": "Pooled relevant recall@10",
        "source": "relative_pooled_coverage",
        "higher_is_better": True,
    },
    {
        "field": "field_subfield_diversity_at_10",
        "label": "Field/subfield diversity@10",
        "source": "andrew_metadata_diagnostic",
        "higher_is_better": True,
    },
)


def score_query_win_rate_with_cache(
    expanded_by_run: dict[str, list[dict[str, Any]]],
    per_query_by_run: dict[str, list[dict[str, Any]]],
    cache_path: str,
    judge_fn: JudgeFn,
) -> dict[str, dict[str, Any]]:
    """Judge anonymous per-query winners with an append-only JSONL cache."""
    run_ids = sorted(expanded_by_run)
    if len(run_ids) < 2:
        raise ValueError("Win-rate judging requires at least two runs.")

    cached = _load_cache(cache_path, run_ids)
    tasks = _build_tasks(expanded_by_run, per_query_by_run, run_ids)
    uncached_query_ids = [query_id for query_id in sorted(tasks) if query_id not in cached]

    if uncached_query_ids:
        cache_dir = os.path.dirname(os.path.abspath(cache_path))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as cache_file:
            for query_id in uncached_query_ids:
                task = tasks[query_id]
                judged = judge_fn(task)
                judgment = _normalize_judgment(task, judged)
                cache_file.write(
                    json.dumps(
                        judgment,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                cache_file.write("\n")
                cache_file.flush()
                cached[query_id] = judgment

    return {query_id: cached[query_id] for query_id in sorted(tasks)}


def summarize_win_rates(
    win_judgments: dict[str, dict[str, Any]],
    run_ids: list[str],
) -> dict[str, Any]:
    """Compute strict/fractional win rates and pairwise preference counts."""
    query_count = len(win_judgments)
    by_run = {
        run_id: {
            "strict_wins": 0,
            "fractional_wins": 0.0,
            "strict_win_rate": 0.0,
            "fractional_win_rate": 0.0,
        }
        for run_id in sorted(run_ids)
    }
    tie_count = 0
    no_clear_winner_count = 0
    pairwise = {
        run_id: {
            other_run_id: {"wins": 0, "losses": 0, "ties": 0, "no_preference": 0}
            for other_run_id in sorted(run_ids)
            if other_run_id != run_id
        }
        for run_id in sorted(run_ids)
    }

    for judgment in win_judgments.values():
        winner_run_ids = list(judgment.get("winner_run_ids") or [])
        outcome = judgment.get("outcome")
        if outcome == "no_clear_winner":
            no_clear_winner_count += 1
        elif len(winner_run_ids) == 1:
            by_run[winner_run_ids[0]]["strict_wins"] += 1
            by_run[winner_run_ids[0]]["fractional_wins"] += 1.0
        elif winner_run_ids:
            tie_count += 1
            share = 1.0 / len(winner_run_ids)
            for run_id in winner_run_ids:
                by_run[run_id]["fractional_wins"] += share

        _add_pairwise_counts(pairwise, judgment)

    for run_id, row in by_run.items():
        if query_count:
            row["strict_win_rate"] = row["strict_wins"] / query_count
            row["fractional_win_rate"] = row["fractional_wins"] / query_count

    return {
        "query_count": query_count,
        "tie_count": tie_count,
        "no_clear_winner_count": no_clear_winner_count,
        "by_run": by_run,
        "pairwise": pairwise,
        "metrics_used": list(WIN_RATE_METRICS),
    }


def _build_tasks(
    expanded_by_run: dict[str, list[dict[str, Any]]],
    per_query_by_run: dict[str, list[dict[str, Any]]],
    run_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Build one anonymous comparison task per query."""
    expanded_rows = {
        run_id: {str(row.get("query_id")): row for row in rows}
        for run_id, rows in expanded_by_run.items()
    }
    metric_rows = {
        run_id: {str(row.get("query_id")): row for row in rows}
        for run_id, rows in per_query_by_run.items()
    }
    query_ids = sorted(set.intersection(*(set(rows) for rows in expanded_rows.values())))
    tasks: dict[str, dict[str, Any]] = {}

    for query_id in query_ids:
        label_to_run_id = _anonymous_mapping(query_id, run_ids)
        first_row = expanded_rows[label_to_run_id["A"]][query_id]
        systems: dict[str, Any] = {}
        for label, run_id in label_to_run_id.items():
            systems[label] = {
                "support_metrics": _select_support_metrics(metric_rows[run_id][query_id]),
                "results": _public_result_payload(expanded_rows[run_id][query_id]),
            }

        tasks[query_id] = {
            "task_type": "anonymous_result_set_comparison",
            "cache_version": CACHE_VERSION,
            "query_id": query_id,
            "query_text": first_row.get("query_text"),
            "query_type": first_row.get("query_type"),
            "metrics_used": list(WIN_RATE_METRICS),
            "systems": systems,
            "expected_output_schema": {
                "winner": "A | B | C | tie | no_clear_winner",
                "tied_systems": ["A", "B"],
                "ranked_systems": ["A", "B", "C"],
                "pairwise_preferences": {"A_vs_B": "A | B | tie | no_preference"},
                "confidence": "low | medium | high",
                "short_rationale": "brief explanation",
            },
            "_label_to_run_id": label_to_run_id,
            "_run_ids": run_ids,
        }

    return tasks


def _anonymous_mapping(query_id: str, run_ids: list[str]) -> dict[str, str]:
    """Return a deterministic per-query anonymous label mapping."""
    labels = [chr(ord("A") + index) for index in range(len(run_ids))]
    shuffled_run_ids = list(run_ids)
    seed = hashlib.sha256(
        f"{CACHE_VERSION}|{query_id}|{'|'.join(run_ids)}".encode("utf-8")
    ).hexdigest()
    random.Random(seed).shuffle(shuffled_run_ids)
    return dict(zip(labels, shuffled_run_ids))


def _select_support_metrics(metric_row: dict[str, Any]) -> dict[str, Any]:
    """Return only the top-five support metrics shown to the judge."""
    return {
        metric["field"]: metric_row.get(metric["field"])
        for metric in WIN_RATE_METRICS
    }


def _public_result_payload(expanded_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return result metadata safe for the anonymous judge prompt."""
    results = expanded_row.get("results")
    if not isinstance(results, list):
        return []

    payload: list[dict[str, Any]] = []
    for result in sorted(
        (result for result in results if isinstance(result, dict)),
        key=lambda item: int(item.get("rank", 0)),
    ):
        payload.append(
            {
                "rank": result.get("rank"),
                "paper_id": result.get("paper_id"),
                "title": result.get("title"),
                "abstract": result.get("abstract"),
                "authors": result.get("authors") or [],
                "year": result.get("year"),
                "venue": result.get("venue") or "",
                "fields_of_study": result.get("fields_of_study") or result.get("categories") or [],
                "subfields": result.get("subfields") or [],
                "citation_count": result.get("citation_count"),
                "reference_count": result.get("reference_count"),
            }
        )
    return payload


def _normalize_judgment(task: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    """Validate and de-anonymize one judge output for storage."""
    if not isinstance(judgment, dict):
        raise ValueError(f"Win-rate judge output for {task['query_id']!r} must be a dict.")

    label_to_run_id = task["_label_to_run_id"]
    labels = set(label_to_run_id)
    winner = judgment.get("winner")
    if not isinstance(winner, str):
        raise ValueError(f"Win-rate judge output for {task['query_id']!r} is missing winner.")

    tied_labels = _normalize_label_list(judgment.get("tied_systems"), labels)
    outcome = "single_winner"
    winner_labels: list[str]
    if winner in labels:
        winner_labels = [winner]
    elif winner == "tie":
        winner_labels = tied_labels or sorted(labels)
        outcome = "tie"
    elif winner == "no_clear_winner":
        winner_labels = []
        outcome = "no_clear_winner"
    else:
        raise ValueError(
            f"Win-rate judge output for {task['query_id']!r} has invalid winner {winner!r}."
        )

    confidence = judgment.get("confidence", "")
    if confidence not in {"", "low", "medium", "high"}:
        raise ValueError(f"Win-rate judge output for {task['query_id']!r} has invalid confidence.")
    rationale = judgment.get("short_rationale", "")
    if not isinstance(rationale, str):
        raise ValueError(f"Win-rate judge output for {task['query_id']!r} has invalid rationale.")

    pairwise_preferences = _normalize_pairwise_preferences(
        judgment.get("pairwise_preferences", {}),
        labels,
    )

    return {
        "cache_version": CACHE_VERSION,
        "query_id": task["query_id"],
        "run_ids": list(task["_run_ids"]),
        "label_to_run_id": label_to_run_id,
        "winner_label": winner,
        "winner_run_ids": [label_to_run_id[label] for label in winner_labels],
        "outcome": outcome,
        "tied_labels": winner_labels if outcome == "tie" else [],
        "ranked_systems": judgment.get("ranked_systems", []),
        "pairwise_preferences": pairwise_preferences,
        "confidence": confidence,
        "short_rationale": rationale,
        "metrics_used": list(WIN_RATE_METRICS),
    }


def _normalize_label_list(value: Any, labels: set[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("tied_systems must be a list when provided.")
    normalized: list[str] = []
    for item in value:
        if item not in labels:
            raise ValueError(f"Unknown anonymous system label {item!r}.")
        if item not in normalized:
            normalized.append(item)
    return normalized


def _normalize_pairwise_preferences(value: Any, labels: set[str]) -> dict[str, str]:
    """Validate pairwise preferences while allowing missing entries."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("pairwise_preferences must be an object when provided.")

    normalized: dict[str, str] = {}
    allowed_values = labels | {"tie", "no_preference"}
    for raw_key, raw_preference in value.items():
        key = str(raw_key)
        preference = str(raw_preference)
        if preference not in allowed_values:
            raise ValueError(f"Invalid pairwise preference {preference!r} for {key!r}.")
        normalized[key] = preference
    return normalized


def _load_cache(cache_path: str, run_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Load cache rows matching the current run set and cache version."""
    cached: dict[str, dict[str, Any]] = {}
    if not os.path.exists(cache_path):
        return cached

    expected_run_ids = list(run_ids)
    with open(cache_path, "r", encoding="utf-8") as cache_file:
        for line_number, raw_line in enumerate(cache_file, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in win-rate cache row {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Win-rate cache row {line_number} must be a JSON object.")
            if row.get("cache_version") != CACHE_VERSION:
                continue
            if row.get("run_ids") != expected_run_ids:
                continue
            query_id = row.get("query_id")
            if isinstance(query_id, str):
                cached[query_id] = row
    return cached


def _add_pairwise_counts(pairwise: dict[str, dict[str, dict[str, int]]], judgment: dict[str, Any]) -> None:
    """Accumulate pairwise preference counts from one judgment row."""
    label_to_run_id = judgment.get("label_to_run_id") or {}
    preferences = judgment.get("pairwise_preferences") or {}
    if not isinstance(label_to_run_id, dict) or not isinstance(preferences, dict):
        return

    for raw_key, preference in preferences.items():
        labels = _extract_pair_labels(str(raw_key))
        if labels is None:
            continue
        left_label, right_label = labels
        left_run = label_to_run_id.get(left_label)
        right_run = label_to_run_id.get(right_label)
        if not isinstance(left_run, str) or not isinstance(right_run, str):
            continue

        if preference == left_label:
            pairwise[left_run][right_run]["wins"] += 1
            pairwise[right_run][left_run]["losses"] += 1
        elif preference == right_label:
            pairwise[right_run][left_run]["wins"] += 1
            pairwise[left_run][right_run]["losses"] += 1
        elif preference == "tie":
            pairwise[left_run][right_run]["ties"] += 1
            pairwise[right_run][left_run]["ties"] += 1
        elif preference == "no_preference":
            pairwise[left_run][right_run]["no_preference"] += 1
            pairwise[right_run][left_run]["no_preference"] += 1


def _extract_pair_labels(key: str) -> tuple[str, str] | None:
    """Parse pair keys such as ``A_vs_B`` or ``A>B``."""
    for separator in ("_vs_", ">", "|", ":"):
        if separator in key:
            left, right = key.split(separator, 1)
            return left.strip(), right.strip()
    return None
