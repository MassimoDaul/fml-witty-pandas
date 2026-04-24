"""Load evaluator benchmark queries from a JSONL file.

The evaluator uses a stable internal shape even though the repo currently has
two public query contracts:

* legacy evaluator rows: ``query_id``, ``query_text``, ``query_type``
* Semantic Scholar-style rows: ``queryId``, ``query``, ``year``, ``fields``

The evaluator accepts legacy ``time_filter`` and Semantic Scholar ``year``
fields, but they are retained only as source metadata. Time filtering is not a
standalone evaluation metric.

Public benchmark files should stay small and model-facing. When ``query_type``
is absent, the current 100-query benchmark convention is used:
``q_001``-``q_050`` are broad and ``q_051``-``q_100`` are specific.
"""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
import re
from typing import Any, Literal

QueryType = Literal["broad", "specific"]

__all__ = ["load_eval_queries"]


def load_eval_queries(path: str) -> dict[str, dict]:
    """Load and normalize evaluator query metadata keyed by ``query_id``.

    Args:
        path: Path to an internal evaluator JSONL query file.

    Returns:
        A mapping from ``query_id`` to a normalized query record.

    Raises:
        ValueError: If the file contains malformed JSON or invalid query rows.
    """
    source = Path(path)
    queries: dict[str, dict] = {}

    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue

                try:
                    raw_row = json.loads(raw_line)
                except JSONDecodeError as exc:
                    raise ValueError(
                        f"{source}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc

                normalized = _normalize_query_row(
                    raw_row,
                    source=source,
                    line_number=line_number,
                )
                query_id = normalized["query_id"]
                if query_id in queries:
                    raise ValueError(
                        f"{source}:{line_number}: duplicate query_id {query_id!r}"
                    )
                queries[query_id] = normalized
    except OSError as exc:
        message = exc.strerror or str(exc)
        raise ValueError(f"Could not read query file {path!r}: {message}.") from exc

    return queries


def _normalize_query_row(
    raw_row: Any,
    *,
    source: Path,
    line_number: int,
) -> dict[str, object]:
    """Validate and normalize one evaluator query row."""
    if not isinstance(raw_row, dict):
        raise ValueError(f"{source}:{line_number}: expected a JSON object")

    query_id = _normalize_required_text_field(
        raw_row,
        ("query_id", "queryId"),
        canonical_name="query_id",
        source=source,
        line_number=line_number,
    )
    query_text = _normalize_required_text_field(
        raw_row,
        ("query_text", "query"),
        canonical_name="query_text",
        source=source,
        line_number=line_number,
    )
    query_type = _normalize_query_type(
        raw_row.get("query_type", raw_row.get("queryType")),
        query_id=query_id,
        source=source,
        line_number=line_number,
    )
    fields = _normalize_optional_text(raw_row.get("fields"))

    return {
        "query_id": query_id,
        "query_text": query_text,
        "query_type": query_type,
        "source_fields": fields,
        "source_year": raw_row.get("year"),
    }


def _normalize_required_text_field(
    raw_row: dict[str, Any],
    field_names: tuple[str, ...],
    *,
    canonical_name: str,
    source: Path,
    line_number: int,
) -> str:
    """Return the first present text field from a set of accepted names."""
    for field_name in field_names:
        if field_name in raw_row:
            return _normalize_required_text(
                raw_row.get(field_name),
                field_name=field_name,
                source=source,
                line_number=line_number,
            )

    accepted = " or ".join(repr(name) for name in field_names)
    raise ValueError(f"{source}:{line_number}: {canonical_name} must be provided as {accepted}")


def _normalize_required_text(
    value: object,
    *,
    field_name: str,
    source: Path,
    line_number: int,
) -> str:
    """Return a stripped, non-empty text field."""
    if not isinstance(value, str):
        raise ValueError(f"{source}:{line_number}: {field_name} must be a string")

    text = value.strip()
    if not text:
        raise ValueError(f"{source}:{line_number}: {field_name} must not be blank")
    return text


def _normalize_optional_text(value: object) -> str | None:
    """Return a stripped optional text field."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_query_type(
    value: object,
    *,
    query_id: str,
    source: Path,
    line_number: int,
) -> QueryType:
    """Validate the query type label."""
    if value is None:
        return _infer_query_type(query_id, source=source, line_number=line_number)
    if value not in {"broad", "specific"}:
        raise ValueError(
            f"{source}:{line_number}: query_type must be 'broad' or 'specific'"
        )
    return value


def _infer_query_type(
    query_id: str,
    *,
    source: Path,
    line_number: int,
) -> QueryType:
    """Infer query type for the current 100-query benchmark convention."""
    match = re.fullmatch(r"q[_-]?(\d+)", query_id)
    if match is None:
        raise ValueError(
            f"{source}:{line_number}: query_type is missing and could not be inferred "
            f"from query_id {query_id!r}"
        )

    query_number = int(match.group(1))
    if 1 <= query_number <= 50:
        return "broad"
    if 51 <= query_number <= 100:
        return "specific"

    raise ValueError(
        f"{source}:{line_number}: query_type is missing and query_id {query_id!r} "
        "is outside the supported q_001-q_100 inference range"
    )

