"""Utilities for building deduplicated paper-level judging tasks.

This module prepares structured task payloads for the paper relevance judge.
It does not call the judge itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

QueryType = Literal["broad", "specific"]


class PaperPayload(TypedDict):
    """Paper metadata included in a paper relevance judging task."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    categories: list[str]


class PaperJudgingTask(TypedDict):
    """Task payload consumed by the paper relevance judge."""

    task_type: Literal["paper_relevance"]
    query_id: str
    query_text: str
    query_type: QueryType
    paper: PaperPayload


__all__ = ["build_unique_paper_judging_tasks"]


def build_unique_paper_judging_tasks(
    expanded_runs: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build deduplicated paper relevance judging tasks.

    The input is expected to be a list of expanded runs, where each run contains
    one record per query and each query record includes a `results` list with
    joined paper metadata. Tasks are deduplicated strictly by the
    `(query_id, paper_id)` pair so the same query-paper judgment can be reused
    across multiple submitted runs.

    Args:
        expanded_runs: Expanded run objects produced by
            `expand_run_with_metadata`.

    Returns:
        A deterministically ordered list of paper relevance judging tasks,
        sorted by `query_id` and then `paper_id`.

    Raises:
        TypeError: If the input structure is not list-like or mapping-like where
            required.
        ValueError: If a required field is missing or has an invalid value.
    """

    unique_tasks: dict[tuple[str, str], PaperJudgingTask] = {}

    for run_index, run in enumerate(expanded_runs):
        if not isinstance(run, list):
            raise TypeError(
                f"expanded_runs[{run_index}] must be a list of query records, "
                f"got {type(run).__name__}."
            )

        for record_index, query_record in enumerate(run):
            _require_mapping(
                query_record,
                context=(
                    f"expanded_runs[{run_index}][{record_index}] must be a "
                    "mapping."
                ),
            )

            query_id = _require_string(
                query_record, "query_id", f"expanded_runs[{run_index}][{record_index}]"
            )
            query_text = _require_string(
                query_record, "query_text", f"expanded_runs[{run_index}][{record_index}]"
            )
            query_type = _require_query_type(
                query_record.get("query_type"),
                f"expanded_runs[{run_index}][{record_index}]['query_type']",
            )
            results = _require_list(
                query_record,
                "results",
                f"expanded_runs[{run_index}][{record_index}]",
            )

            for result_index, result in enumerate(results):
                _require_mapping(
                    result,
                    context=(
                        "expanded_runs"
                        f"[{run_index}][{record_index}]['results'][{result_index}] "
                        "must be a mapping."
                    ),
                )
                paper = _build_paper_payload(
                    result,
                    context=(
                        "expanded_runs"
                        f"[{run_index}][{record_index}]['results'][{result_index}]"
                    ),
                )
                task_key = (query_id, paper["paper_id"])
                task: PaperJudgingTask = {
                    "task_type": "paper_relevance",
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_type": query_type,
                    "paper": paper,
                }

                existing_task = unique_tasks.get(task_key)
                if existing_task is None:
                    unique_tasks[task_key] = task
                elif existing_task != task:
                    raise ValueError(
                        "Conflicting metadata found for duplicated "
                        f"(query_id, paper_id) pair {task_key!r}."
                    )

    ordered_keys = sorted(unique_tasks, key=lambda item: (item[0], item[1]))
    return [unique_tasks[key] for key in ordered_keys]


def _build_paper_payload(result: Mapping[str, Any], context: str) -> PaperPayload:
    """Extract the paper payload from an expanded result item."""

    paper_source = result.get("paper")
    paper_context = context
    if paper_source is None:
        paper_source = result
    else:
        paper_context = f"{context}['paper']"
    _require_mapping(paper_source, context=f"{paper_context} must be a mapping.")

    payload: dict[str, Any] = {
        "paper_id": _require_string(paper_source, "paper_id", paper_context),
        "title": _require_string(paper_source, "title", paper_context),
        "abstract": _require_string(paper_source, "abstract", paper_context),
        "authors": _require_string_list(paper_source, "authors", paper_context),
        "year": _require_int(paper_source, "year", paper_context),
        "categories": _require_string_list(paper_source, "categories", paper_context),
    }
    for optional_string in ("s2_paper_id", "url", "venue"):
        value = paper_source.get(optional_string)
        if isinstance(value, str):
            payload[optional_string] = value
    for optional_list in ("fields_of_study", "subfields", "author_ids"):
        value = paper_source.get(optional_list)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            payload[optional_list] = list(value)
    for optional_int in ("citation_count", "reference_count"):
        value = paper_source.get(optional_int)
        if isinstance(value, int) and not isinstance(value, bool):
            payload[optional_int] = value
    return payload


def _require_mapping(value: Any, context: str) -> None:
    """Raise if a value is not a mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{context} Got {type(value).__name__}.")


def _require_list(
    mapping: Mapping[str, Any], key: str, context: str
) -> list[Any]:
    """Read a required list field from a mapping."""

    value = mapping.get(key)
    if not isinstance(value, list):
        raise TypeError(
            f"{context}['{key}'] must be a list, got {type(value).__name__}."
        )
    return value


def _require_string(
    mapping: Mapping[str, Any], key: str, context: str
) -> str:
    """Read a required string field from a mapping."""

    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context}['{key}'] must be a string, got {value!r}.")
    return value


def _require_string_list(
    mapping: Mapping[str, Any], key: str, context: str
) -> list[str]:
    """Read a required list[str] field from a mapping."""

    value = mapping.get(key)
    if not isinstance(value, list):
        raise TypeError(
            f"{context}['{key}'] must be a list of strings, got {type(value).__name__}."
        )
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context}['{key}'] must contain only strings.")
    return list(value)


def _require_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    """Read a required integer field from a mapping."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}['{key}'] must be an integer, got {value!r}.")
    return value


def _require_query_type(value: Any, context: str) -> QueryType:
    """Validate the query type label."""

    if value not in {"broad", "specific"}:
        raise ValueError(f"{context} must be 'broad' or 'specific', got {value!r}.")
    return value
