"""Expand validated run rows with query fields and canonical paper metadata."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


QueryType = Literal["broad", "specific"]


class ExpandedResult(TypedDict):
    """Expanded metadata for one ranked paper."""

    rank: int
    paper_id: str
    submitted_paper_id: str
    score: float | None
    s2_paper_id: str
    url: str
    title: str
    abstract: str
    authors: list[str]
    author_ids: list[str]
    year: int
    categories: list[str]
    fields_of_study: list[str]
    subfields: list[str]
    venue: str
    citation_count: int
    reference_count: int


class ExpandedQueryRow(TypedDict):
    """Expanded submission row used by later judging and scoring steps."""

    run_id: str
    query_id: str
    query_text: str
    query_type: QueryType
    results: list[ExpandedResult]


def _require_mapping(mapping: Any, *, label: str) -> dict[str, Any]:
    """Return a dictionary-like object or raise a clear error."""
    if not isinstance(mapping, dict):
        raise TypeError(f"{label} must be a dict, got {type(mapping).__name__}.")
    return mapping


def _require_field(mapping: dict[str, Any], field_name: str, *, label: str) -> Any:
    """Read a required field from a mapping with a descriptive error."""
    if field_name not in mapping:
        raise KeyError(f"Missing required field {field_name!r} in {label}.")
    return mapping[field_name]


def _copy_string_list(value: Any, *, field_name: str, item_label: str) -> list[str]:
    """Return a detached list of strings for JSON output."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(
            f"{field_name} in {item_label} must be a list, got {type(value).__name__}."
        )
    return [str(item) for item in value]


def _optional_text(value: Any) -> str:
    """Return an optional text value as a JSON-serializable string."""
    if value is None:
        return ""
    return str(value)


def _optional_int(value: Any, *, default: int = 0) -> int:
    """Return an optional integer value with a default."""
    if value is None:
        return default
    return int(value)


def expand_run_with_metadata(
    validated_run: list[dict],
    queries: dict[str, dict],
    paper_metadata: dict[str, dict],
) -> list[dict]:
    """Join a validated submission run with query fields and paper metadata.

    Args:
        validated_run: Validated submission rows, one per query.
        queries: Query metadata keyed by `query_id`.
        paper_metadata: Canonical paper metadata keyed by `paper_id`.

    Returns:
        A new list of expanded query rows suitable for JSON serialization and
        downstream judge-packet construction.

    Raises:
        KeyError: If a referenced `query_id`, `paper_id`, or required metadata
            field cannot be found.
        TypeError: If an input row has the wrong container type.
        ValueError: If `query_type` is not one of the supported values.
    """
    expanded_rows: list[ExpandedQueryRow] = []

    for row_index, row in enumerate(validated_run):
        row_mapping = _require_mapping(row, label=f"validated_run[{row_index}]")

        query_id = str(_require_field(row_mapping, "query_id", label=f"validated_run[{row_index}]"))
        try:
            query_info = _require_mapping(
                queries[query_id],
                label=f"queries[{query_id!r}]",
            )
        except KeyError as exc:
            raise KeyError(
                f"Unknown query_id {query_id!r} referenced in validated_run[{row_index}]."
            ) from exc

        query_type = _require_field(
            query_info,
            "query_type",
            label=f"queries[{query_id!r}]",
        )
        if query_type not in ("broad", "specific"):
            raise ValueError(
                f"Invalid query_type {query_type!r} for query_id {query_id!r}; "
                "expected 'broad' or 'specific'."
            )

        expanded_results: list[ExpandedResult] = []
        results = _require_field(row_mapping, "results", label=f"validated_run[{row_index}]")
        if not isinstance(results, list):
            raise TypeError(
                f"results in validated_run[{row_index}] must be a list, "
                f"got {type(results).__name__}."
            )

        for result_index, result in enumerate(results):
            result_mapping = _require_mapping(
                result,
                label=f"validated_run[{row_index}]['results'][{result_index}]",
            )

            paper_id = str(
                _require_field(
                    result_mapping,
                    "paper_id",
                    label=f"validated_run[{row_index}]['results'][{result_index}]",
                )
            )
            try:
                metadata = _require_mapping(
                    paper_metadata[paper_id],
                    label=f"paper_metadata[{paper_id!r}]",
                )
            except KeyError as exc:
                raise KeyError(
                    "Unknown paper_id "
                    f"{paper_id!r} referenced in validated_run[{row_index}] "
                    f"result {result_index} for query_id {query_id!r}."
                ) from exc

            score = result_mapping.get("score")
            fields_of_study = metadata.get("fields_of_study", metadata.get("categories", []))
            expanded_results.append(
                {
                    "rank": int(
                        _require_field(
                            result_mapping,
                            "rank",
                            label=f"validated_run[{row_index}]['results'][{result_index}]",
                        )
                    ),
                    "paper_id": paper_id,
                    "submitted_paper_id": _optional_text(
                        result_mapping.get("submitted_paper_id", paper_id)
                    ),
                    "score": None if score is None else float(score),
                    "s2_paper_id": _optional_text(metadata.get("s2_paper_id")),
                    "url": _optional_text(metadata.get("url")),
                    "title": str(
                        _require_field(
                            metadata,
                            "title",
                            label=f"paper_metadata[{paper_id!r}]",
                        )
                    ),
                    "abstract": str(
                        _require_field(
                            metadata,
                            "abstract",
                            label=f"paper_metadata[{paper_id!r}]",
                        )
                    ),
                    "authors": _copy_string_list(
                        _require_field(
                            metadata,
                            "authors",
                            label=f"paper_metadata[{paper_id!r}]",
                        ),
                        field_name="authors",
                        item_label=f"paper_metadata[{paper_id!r}]",
                    ),
                    "author_ids": _copy_string_list(
                        metadata.get("author_ids", []),
                        field_name="author_ids",
                        item_label=f"paper_metadata[{paper_id!r}]",
                    ),
                    "year": int(
                        _require_field(
                            metadata,
                            "year",
                            label=f"paper_metadata[{paper_id!r}]",
                        )
                    ),
                    "categories": _copy_string_list(
                        _require_field(
                            metadata,
                            "categories",
                            label=f"paper_metadata[{paper_id!r}]",
                        ),
                        field_name="categories",
                        item_label=f"paper_metadata[{paper_id!r}]",
                    ),
                    "fields_of_study": _copy_string_list(
                        fields_of_study,
                        field_name="fields_of_study",
                        item_label=f"paper_metadata[{paper_id!r}]",
                    ),
                    "subfields": _copy_string_list(
                        metadata.get("subfields", []),
                        field_name="subfields",
                        item_label=f"paper_metadata[{paper_id!r}]",
                    ),
                    "venue": _optional_text(metadata.get("venue")),
                    "citation_count": _optional_int(metadata.get("citation_count")),
                    "reference_count": _optional_int(metadata.get("reference_count")),
                }
            )

        expanded_rows.append(
            {
                "run_id": str(
                    _require_field(
                        row_mapping,
                        "run_id",
                        label=f"validated_run[{row_index}]",
                    )
                ),
                "query_id": query_id,
                "query_text": str(
                    _require_field(
                        query_info,
                        "query_text",
                        label=f"queries[{query_id!r}]",
                    )
                ),
                "query_type": query_type,
                "results": expanded_results,
            }
        )

    return expanded_rows
