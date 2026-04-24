"""Broad-query Exploration Range judging with append-only JSONL caching."""

from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Callable

__all__ = ["score_broad_exploration_range_with_cache"]

_VALID_RANGE_SCORES = {1, 2, 3, 4, 5}


def score_broad_exploration_range_with_cache(
    expanded_runs: list[list[dict[str, Any]]],
    cache_path: str,
    judge_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Judge Exploration Range for broad-query result lists with JSONL caching.

    Args:
        expanded_runs: Expanded run payloads from ``expand_run_with_metadata``.
        cache_path: Path to an append-only JSONL cache file.
        judge_fn: Callback that receives one exploration-range task and returns a
            structured judgment payload.

    Returns:
        A mapping keyed by ``(run_id, query_id)``.

    Raises:
        ValueError: If inputs, cached rows, or judge outputs are malformed.
    """
    judgments: dict[tuple[str, str], dict[str, Any]] = {}

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            for line_number, raw_line in enumerate(cache_file, start=1):
                if not raw_line.strip():
                    continue
                cached_row = _read_and_validate_cache_row(raw_line, line_number)
                key = (cached_row["run_id"], cached_row["query_id"])
                judgments[key] = cached_row

    tasks_by_key = _build_broad_query_tasks(expanded_runs)
    uncached_keys = [key for key in sorted(tasks_by_key) if key not in judgments]

    if uncached_keys:
        cache_dir = os.path.dirname(os.path.abspath(cache_path))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with open(cache_path, "a", encoding="utf-8") as cache_file:
            for key in uncached_keys:
                run_id, query_id = key
                judged = judge_fn(tasks_by_key[key])
                judgment = _normalize_judgment(
                    run_id,
                    query_id,
                    judged,
                    context=f"Judge output for ({run_id!r}, {query_id!r})",
                )
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
                judgments[key] = judgment

    return judgments


def _build_broad_query_tasks(
    expanded_runs: list[list[dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build deterministic exploration-range tasks for broad queries only."""
    tasks: dict[tuple[str, str], dict[str, Any]] = {}

    for run_index, run in enumerate(expanded_runs):
        if not isinstance(run, list):
            raise ValueError(
                f"expanded_runs[{run_index}] must be a list, got {type(run).__name__}"
            )

        for row_index, row in enumerate(run):
            if not isinstance(row, dict):
                raise ValueError(
                    f"expanded_runs[{run_index}][{row_index}] must be a dict"
                )

            query_type = row.get("query_type")
            if query_type == "specific":
                continue
            if query_type != "broad":
                raise ValueError(
                    f"expanded_runs[{run_index}][{row_index}] has invalid query_type {query_type!r}"
                )

            run_id = _require_string_field(
                row,
                "run_id",
                context=f"expanded_runs[{run_index}][{row_index}]",
            )
            query_id = _require_string_field(
                row,
                "query_id",
                context=f"expanded_runs[{run_index}][{row_index}]",
            )
            payload = _build_task_payload(
                row,
                context=f"expanded_runs[{run_index}][{row_index}]",
            )
            key = (run_id, query_id)

            existing = tasks.get(key)
            if existing is None:
                tasks[key] = payload
            elif existing != payload:
                raise ValueError(
                    f"Conflicting broad-query payloads found for key {(run_id, query_id)!r}."
                )

    return tasks


def _build_task_payload(row: dict[str, Any], *, context: str) -> dict[str, Any]:
    """Build the judge payload for one broad-query result list."""
    papers: list[dict[str, Any]] = []
    results = row.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{context}['results'] must be a list")
    if len(results) != 10:
        raise ValueError(f"{context}['results'] must contain exactly 10 papers")

    for paper_index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(
                f"{context}['results'][{paper_index}] must be a dict"
            )
        paper_payload = {
            "rank": _require_int(
                result,
                "rank",
                context=f"{context}['results'][{paper_index}]",
            ),
            "paper_id": _require_string_field(
                result,
                "paper_id",
                context=f"{context}['results'][{paper_index}]",
            ),
            "title": _require_string_field(
                result,
                "title",
                context=f"{context}['results'][{paper_index}]",
            ),
            "abstract": _require_string_field(
                result,
                "abstract",
                context=f"{context}['results'][{paper_index}]",
            ),
            "year": _require_int(
                result,
                "year",
                context=f"{context}['results'][{paper_index}]",
            ),
            "categories": _require_string_list(
                result,
                "categories",
                context=f"{context}['results'][{paper_index}]",
            ),
        }
        for field_name in ("fields_of_study", "subfields", "authors"):
            value = result.get(field_name)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                paper_payload[field_name] = list(value)
        for field_name in ("venue", "s2_paper_id", "url"):
            value = result.get(field_name)
            if isinstance(value, str):
                paper_payload[field_name] = value
        for field_name in ("citation_count", "reference_count"):
            value = result.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool):
                paper_payload[field_name] = value
        papers.append(paper_payload)

    return {
        "task_type": "exploration_range",
        "run_id": _require_string_field(row, "run_id", context=context),
        "query_id": _require_string_field(row, "query_id", context=context),
        "query_text": _require_string_field(row, "query_text", context=context),
        "papers": papers,
    }


def _normalize_judgment(
    run_id: str,
    query_id: str,
    judgment: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate a judge response and return the canonical stored row."""
    if not isinstance(judgment, dict):
        raise ValueError(f"{context} must be a dict")

    raw_score = judgment.get("exploration_range_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, int):
        raise ValueError(f"{context} has a non-integer 'exploration_range_score'")
    if raw_score not in _VALID_RANGE_SCORES:
        raise ValueError(
            f"{context} has an invalid 'exploration_range_score': {raw_score!r}"
        )

    represented_subtopics = judgment.get("represented_subtopics", [])
    if represented_subtopics is None:
        represented_subtopics = []
    if not isinstance(represented_subtopics, list) or any(
        not isinstance(item, str) for item in represented_subtopics
    ):
        raise ValueError(f"{context} has an invalid 'represented_subtopics' list")

    short_rationale = judgment.get("short_rationale", "")
    if short_rationale is None:
        short_rationale = ""
    if not isinstance(short_rationale, str):
        raise ValueError(f"{context} has a non-string 'short_rationale'")

    return {
        "run_id": run_id,
        "query_id": query_id,
        "exploration_range_score": raw_score,
        "represented_subtopics": list(represented_subtopics),
        "short_rationale": short_rationale,
    }


def _read_and_validate_cache_row(raw_line: str, line_number: int) -> dict[str, Any]:
    """Parse one cache row and normalize its contents."""
    try:
        row = json.loads(raw_line)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in cache row {line_number}: {exc.msg}") from exc

    if not isinstance(row, dict):
        raise ValueError(f"Cache row {line_number} must decode to a JSON object")

    run_id = _require_string_field(row, "run_id", context=f"Cache row {line_number}")
    query_id = _require_string_field(
        row,
        "query_id",
        context=f"Cache row {line_number}",
    )
    return _normalize_judgment(
        run_id,
        query_id,
        row,
        context=f"Cache row {line_number}",
    )


def _require_string_field(record: dict[str, Any], field_name: str, *, context: str) -> str:
    """Return a required string field."""
    value = record.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{context} is missing a string {field_name!r} field")
    return value


def _require_int(record: dict[str, Any], field_name: str, *, context: str) -> int:
    """Return a required integer field."""
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} has a non-integer {field_name!r} field")
    return value


def _require_string_list(
    record: dict[str, Any],
    field_name: str,
    *,
    context: str,
) -> list[str]:
    """Return a required list of strings."""
    value = record.get(field_name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context} has an invalid {field_name!r} list")
    return list(value)
