"""Load and validate evaluation run submissions.

The evaluator normalizes both legacy snake_case submissions and newer
Semantic Scholar-style camelCase submissions to the internal shape:
``run_id``, ``query_id``, and canonical DB ``corpus_id`` stored as
``paper_id``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

EXPECTED_RESULTS_PER_QUERY = 10
EXPECTED_RANKS = list(range(1, EXPECTED_RESULTS_PER_QUERY + 1))


def load_and_validate_run_submission(
    run_path: str,
    queries: dict[str, dict],
    paper_metadata: dict[str, dict],
) -> list[dict]:
    """Load a JSONL run submission and validate it against benchmark metadata.

    Args:
        run_path: Path to the submitted JSONL file.
        queries: Mapping of canonical query IDs to query metadata.
        paper_metadata: Mapping of canonical paper IDs to paper metadata.

    Returns:
        A normalized list of submission rows sorted by ``query_id``. Each row
        contains a consistent ``run_id`` and a ``results`` list sorted by rank.

    Raises:
        ValueError: If the file cannot be read or the submission is invalid.
    """
    expected_query_ids = set(queries)
    paper_id_aliases = _build_paper_id_aliases(paper_metadata)
    seen_query_lines: dict[str, int] = {}
    normalized_rows: list[dict] = []
    expected_run_id: str | None = None
    path = Path(run_path)

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    raise ValueError(f"Invalid JSONL in {run_path!r}: line {line_number} is blank.")

                try:
                    raw_row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {run_path!r}: {exc.msg}."
                    ) from exc

                normalized_row = _validate_submission_row(
                    raw_row=raw_row,
                    line_number=line_number,
                    expected_query_ids=expected_query_ids,
                    paper_id_aliases=paper_id_aliases,
                )

                run_id = normalized_row["run_id"]
                query_id = normalized_row["query_id"]

                if expected_run_id is None:
                    expected_run_id = run_id
                elif run_id != expected_run_id:
                    raise ValueError(
                        f"Inconsistent run_id on line {line_number}: "
                        f"expected {expected_run_id!r}, got {run_id!r}."
                    )

                if query_id in seen_query_lines:
                    raise ValueError(
                        f"Duplicate query_id {query_id!r}: first seen on line "
                        f"{seen_query_lines[query_id]}, repeated on line {line_number}."
                    )

                seen_query_lines[query_id] = line_number
                normalized_rows.append(normalized_row)
    except OSError as exc:
        message = exc.strerror or str(exc)
        raise ValueError(f"Could not read submission file {run_path!r}: {message}.") from exc

    missing_query_ids = sorted(expected_query_ids - set(seen_query_lines))
    if missing_query_ids:
        missing = ", ".join(repr(query_id) for query_id in missing_query_ids)
        raise ValueError(f"Submission is missing results for query_id values: {missing}.")

    normalized_rows.sort(key=lambda row: row["query_id"])
    return normalized_rows


def _validate_submission_row(
    raw_row: Any,
    line_number: int,
    expected_query_ids: set[str],
    paper_id_aliases: dict[str, str],
) -> dict:
    """Validate and normalize one JSONL submission row."""
    if not isinstance(raw_row, dict):
        raise ValueError(f"Line {line_number} must contain a JSON object.")

    run_id = _require_string_field(raw_row, ("run_id", "runId"), line_number)
    if not run_id.strip():
        raise ValueError(f"Line {line_number} field 'run_id' must not be empty.")

    query_id = _require_string_field(raw_row, ("query_id", "queryId"), line_number)
    if query_id not in expected_query_ids:
        raise ValueError(f"Unknown query_id {query_id!r} on line {line_number}.")

    if "results" not in raw_row:
        raise ValueError(f"Line {line_number} is missing required field 'results'.")

    raw_results = raw_row["results"]
    if not isinstance(raw_results, list):
        raise ValueError(f"Line {line_number} field 'results' must be a list.")
    if len(raw_results) != EXPECTED_RESULTS_PER_QUERY:
        raise ValueError(
            f"Line {line_number} query_id {query_id!r} must contain exactly "
            f"{EXPECTED_RESULTS_PER_QUERY} results; got {len(raw_results)}."
        )

    normalized_results = [
        _validate_result_item(
            raw_result=raw_result,
            line_number=line_number,
            item_index=item_index,
            query_id=query_id,
            paper_id_aliases=paper_id_aliases,
        )
        for item_index, raw_result in enumerate(raw_results, start=1)
    ]

    observed_ranks = sorted(result["rank"] for result in normalized_results)
    if observed_ranks != EXPECTED_RANKS:
        raise ValueError(
            f"Line {line_number} query_id {query_id!r} must use ranks 1 through "
            f"{EXPECTED_RESULTS_PER_QUERY} exactly once; got {observed_ranks}."
        )

    paper_ids = [result["paper_id"] for result in normalized_results]
    duplicate_paper_ids = _find_duplicates(paper_ids)
    if duplicate_paper_ids:
        duplicates = ", ".join(repr(paper_id) for paper_id in duplicate_paper_ids)
        raise ValueError(
            f"Line {line_number} query_id {query_id!r} repeats paper_id values: {duplicates}."
        )

    normalized_results.sort(key=lambda result: result["rank"])
    return {
        "run_id": run_id,
        "query_id": query_id,
        "results": normalized_results,
    }


def _validate_result_item(
    raw_result: Any,
    line_number: int,
    item_index: int,
    query_id: str,
    paper_id_aliases: dict[str, str],
) -> dict:
    """Validate and normalize one ranked result item."""
    location = f"line {line_number} query_id {query_id!r} result #{item_index}"

    if not isinstance(raw_result, dict):
        raise ValueError(f"{location} must be a JSON object.")

    if "rank" not in raw_result:
        raise ValueError(f"{location} is missing required field 'rank'.")
    rank = raw_result["rank"]
    if type(rank) is not int:
        raise ValueError(f"{location} field 'rank' must be an int.")

    submitted_paper_id = _require_string_field(
        raw_result,
        ("paper_id", "paperId", "corpus_id", "corpusId"),
        line_number,
        location=location,
    )
    if submitted_paper_id not in paper_id_aliases:
        raise ValueError(f"{location} references unknown paper_id {submitted_paper_id!r}.")
    paper_id = paper_id_aliases[submitted_paper_id]

    score: float | None = None
    if "score" in raw_result:
        raw_score = raw_result["score"]
        if type(raw_score) not in (int, float):
            raise ValueError(f"{location} field 'score' must be an int or float when provided.")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"{location} field 'score' must be finite when provided.")

    return {
        "rank": rank,
        "paper_id": paper_id,
        "submitted_paper_id": submitted_paper_id,
        "score": score,
    }


def _require_string_field(
    raw_row: dict[str, Any],
    field_names: str | tuple[str, ...],
    line_number: int,
    *,
    location: str | None = None,
) -> str:
    """Return a required string field from a row."""
    accepted_names = (field_names,) if isinstance(field_names, str) else field_names
    for field_name in accepted_names:
        if field_name not in raw_row:
            continue
        value = raw_row[field_name]
        if not isinstance(value, str):
            where = location or f"Line {line_number}"
            raise ValueError(f"{where} field {field_name!r} must be a string.")
        return value

    where = location or f"Line {line_number}"
    accepted = " or ".join(repr(name) for name in accepted_names)
    raise ValueError(f"{where} is missing required field {accepted}.")


def _build_paper_id_aliases(paper_metadata: dict[str, dict]) -> dict[str, str]:
    """Build submitted-id to canonical ``corpus_id`` aliases."""
    aliases: dict[str, str] = {}

    for mapping_key, raw_metadata in paper_metadata.items():
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"paper_metadata[{mapping_key!r}] must be a dict.")

        canonical_id = _first_text_value(
            raw_metadata,
            ("corpus_id", "paper_id"),
            fallback=str(mapping_key),
        )
        candidate_aliases = {
            str(mapping_key),
            canonical_id,
            _first_text_value(raw_metadata, ("s2_paper_id", "paperId"), fallback=""),
        }

        for alias in candidate_aliases:
            if not alias:
                continue
            existing = aliases.get(alias)
            if existing is not None and existing != canonical_id:
                raise ValueError(
                    f"Paper id alias {alias!r} maps to both {existing!r} and {canonical_id!r}."
                )
            aliases[alias] = canonical_id

    return aliases


def _first_text_value(
    mapping: dict[str, Any],
    field_names: tuple[str, ...],
    *,
    fallback: str,
) -> str:
    """Return the first non-empty text value in a mapping."""
    for field_name in field_names:
        value = mapping.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _find_duplicates(values: list[str]) -> list[str]:
    """Return duplicate values in first-seen order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    duplicate_set: set[str] = set()

    for value in values:
        if value in seen and value not in duplicate_set:
            duplicates.append(value)
            duplicate_set.add(value)
        seen.add(value)

    return duplicates
