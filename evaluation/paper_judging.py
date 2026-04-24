"""Paper relevance judging with append-only JSONL caching."""

from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Callable

__all__ = ["score_paper_relevance_with_cache"]

_VALID_RELEVANCE_SCORES = {0, 1, 2, 3}


def _require_string_field(record: dict[str, Any], field_name: str, *, context: str) -> str:
    """Return a required string field or raise a descriptive validation error."""
    value = record.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{context} is missing a string {field_name!r} field")
    return value


def _normalize_judgment(
    query_id: str,
    paper_id: str,
    judgment: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate a judgment payload and return its canonical stored shape."""
    if not isinstance(judgment, dict):
        raise ValueError(f"{context} must be a dict")

    relevance_score = judgment.get("relevance_score")
    if isinstance(relevance_score, bool) or not isinstance(relevance_score, int):
        raise ValueError(f"{context} has a non-integer 'relevance_score'")
    if relevance_score not in _VALID_RELEVANCE_SCORES:
        raise ValueError(f"{context} has an invalid 'relevance_score': {relevance_score!r}")

    short_rationale = judgment.get("short_rationale", "")
    if not isinstance(short_rationale, str):
        raise ValueError(f"{context} has a non-string 'short_rationale'")

    return {
        "query_id": query_id,
        "paper_id": paper_id,
        "relevance_score": relevance_score,
        "short_rationale": short_rationale,
    }


def _read_and_validate_cache_row(raw_line: str, line_number: int) -> dict[str, Any]:
    """Parse one JSONL cache row and return a normalized judgment record."""
    try:
        row = json.loads(raw_line)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in cache row {line_number}: {exc.msg}") from exc

    if not isinstance(row, dict):
        raise ValueError(f"Cache row {line_number} must decode to a JSON object")

    query_id = _require_string_field(row, "query_id", context=f"Cache row {line_number}")
    paper_id = _require_string_field(row, "paper_id", context=f"Cache row {line_number}")
    return _normalize_judgment(
        query_id,
        paper_id,
        row,
        context=f"Cache row {line_number}",
    )


def score_paper_relevance_with_cache(
    tasks: list[dict[str, Any]],
    cache_path: str,
    judge_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Score unique paper judging tasks with an append-only JSONL cache.

    The cache is keyed by ``(query_id, paper_id)``. Existing cached judgments are
    loaded first, and only uncached pairs are sent to ``judge_fn``. Newly scored
    results are appended to ``cache_path`` immediately in task order so reruns can
    resume without rejudging completed pairs.

    Args:
        tasks: Task payloads built by ``build_unique_paper_judging_tasks``.
        cache_path: Path to a JSONL cache file.
        judge_fn: Callback that accepts one task dict and returns a dict with
            ``relevance_score`` and an optional ``short_rationale``.

    Returns:
        A mapping from ``(query_id, paper_id)`` to normalized judgment records.

    Raises:
        ValueError: If task identifiers, cached rows, or judge outputs are invalid.
    """
    judgments: dict[tuple[str, str], dict[str, Any]] = {}

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            for line_number, raw_line in enumerate(cache_file, start=1):
                if not raw_line.strip():
                    continue
                cached_row = _read_and_validate_cache_row(raw_line, line_number)
                key = (cached_row["query_id"], cached_row["paper_id"])
                judgments[key] = cached_row

    uncached_tasks: list[tuple[tuple[str, str], dict[str, Any]]] = []
    seen_uncached: set[tuple[str, str]] = set()

    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"Task {index} must be a dict")

        query_id = _require_string_field(task, "query_id", context=f"Task {index}")
        paper_id = _extract_task_paper_id(task, context=f"Task {index}")
        key = (query_id, paper_id)

        if key in judgments or key in seen_uncached:
            continue

        seen_uncached.add(key)
        uncached_tasks.append((key, task))

    if uncached_tasks:
        cache_dir = os.path.dirname(os.path.abspath(cache_path))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with open(cache_path, "a", encoding="utf-8") as cache_file:
            for (query_id, paper_id), task in uncached_tasks:
                judged = judge_fn(task)
                judgment = _normalize_judgment(
                    query_id,
                    paper_id,
                    judged,
                    context=f"Judge output for ({query_id!r}, {paper_id!r})",
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
                judgments[(query_id, paper_id)] = judgment

    return judgments


def _extract_task_paper_id(task: dict[str, Any], *, context: str) -> str:
    """Read a paper identifier from either the top level or nested task payload."""
    if "paper_id" in task:
        return _require_string_field(task, "paper_id", context=context)

    paper = task.get("paper")
    if not isinstance(paper, dict):
        raise ValueError(f"{context} is missing a dict 'paper' field")

    return _require_string_field(paper, "paper_id", context=f"{context}['paper']")
