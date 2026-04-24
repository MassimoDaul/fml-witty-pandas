"""Load frozen paper metadata from JSONL or CSV into a normalized mapping."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path

__all__ = ["load_paper_metadata"]

REQUIRED_FIELDS = ("paper_id", "title", "abstract", "authors", "year")


def load_paper_metadata(path: str) -> dict[str, dict]:
    """
    Load a frozen paper metadata file into a mapping keyed by ``paper_id``.

    Supported input formats are JSONL and CSV. Each paper is normalized to:

    {
        "paper_id": str,
        "title": str,
        "abstract": str,
        "authors": list[str],
        "year": int,
        "categories": list[str],
    }

    Args:
        path: Path to a ``.jsonl`` or ``.csv`` metadata file.

    Returns:
        A dictionary keyed by normalized ``paper_id`` values.

    Raises:
        ValueError: If the file format is unsupported or any row is invalid.
    """
    source = Path(path)
    records: dict[str, dict] = {}
    seen_rows: dict[str, int] = {}

    for row_number, raw_record in _iter_records(source):
        record: dict = _normalize_record(raw_record, source=source, row_number=row_number)
        paper_id = record["paper_id"]
        if paper_id in records:
            first_row = seen_rows[paper_id]
            raise ValueError(
                f"{source}:{row_number}: duplicate paper_id {paper_id!r} "
                f"(first seen at row {first_row})"
            )

        records[paper_id] = record
        seen_rows[paper_id] = row_number

    return records


def _iter_records(source: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield raw records paired with a 1-based row number."""
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_records(source)
        return
    if suffix == ".csv":
        yield from _iter_csv_records(source)
        return

    raise ValueError(
        f"Unsupported metadata file format {source.suffix!r} for {source}. "
        "Expected a .jsonl or .csv file."
    )


def _iter_jsonl_records(source: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")

            yield line_number, record


def _iter_csv_records(source: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield row dictionaries from a CSV file, skipping blank rows."""
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: CSV file is missing a header row")

        reader.fieldnames = [name.strip() if name is not None else "" for name in reader.fieldnames]
        missing_fields = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(
                f"{source}: CSV header missing required fields: {', '.join(missing_fields)}"
            )

        for row in reader:
            if None in row and row[None]:
                raise ValueError(f"{source}:{reader.line_num}: too many columns in CSV row")
            if _row_is_blank(row):
                continue
            yield reader.line_num, row


def _normalize_record(
    raw_record: Mapping[str, object],
    *,
    source: Path,
    row_number: int,
) -> dict[str, object]:
    """Normalize one raw metadata row and validate required fields."""
    missing_fields = [field for field in REQUIRED_FIELDS if field not in raw_record]
    if missing_fields:
        raise ValueError(
            f"{source}:{row_number}: missing required fields: {', '.join(missing_fields)}"
        )

    return {
        "paper_id": _normalize_text(
            raw_record.get("paper_id"),
            field_name="paper_id",
            source=source,
            row_number=row_number,
            allow_blank=False,
        ),
        "title": _normalize_text(
            raw_record.get("title"),
            field_name="title",
            source=source,
            row_number=row_number,
            allow_blank=False,
        ),
        "abstract": _normalize_text(
            raw_record.get("abstract"),
            field_name="abstract",
            source=source,
            row_number=row_number,
            allow_blank=False,
        ),
        "authors": _normalize_list(
            raw_record.get("authors"),
            field_name="authors",
            source=source,
            row_number=row_number,
        ),
        "year": _normalize_year(
            raw_record.get("year"),
            source=source,
            row_number=row_number,
        ),
        "categories": _normalize_list(
            raw_record.get("categories"),
            field_name="categories",
            source=source,
            row_number=row_number,
        ),
    }


def _normalize_text(
    value: object,
    *,
    field_name: str,
    source: Path,
    row_number: int,
    allow_blank: bool,
) -> str:
    """Convert a scalar field to a stripped string."""
    if value is None:
        if allow_blank:
            return ""
        raise ValueError(f"{source}:{row_number}: {field_name} is required")

    text = str(value).strip()
    if text or allow_blank:
        return text

    raise ValueError(f"{source}:{row_number}: {field_name} must not be blank")


def _normalize_year(value: object, *, source: Path, row_number: int) -> int:
    """Convert a raw year value to an integer."""
    if value is None:
        raise ValueError(f"{source}:{row_number}: year is required")

    if isinstance(value, bool):
        raise ValueError(f"{source}:{row_number}: year must be an integer")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{source}:{row_number}: year must be an integer")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{source}:{row_number}: year must not be blank")

    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{source}:{row_number}: invalid year value {text!r}") from exc


def _normalize_list(
    value: object,
    *,
    field_name: str,
    source: Path,
    row_number: int,
) -> list[str]:
    """Normalize list-like metadata fields such as authors and categories."""
    if value is None:
        return []

    items: object = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        parsed = _maybe_parse_json_array(text)
        if parsed is not None:
            items = parsed
        elif ";" in text:
            items = text.split(";")
        elif "," in text:
            items = text.split(",")
        else:
            items = [text]

    if isinstance(items, list):
        return _clean_list_items(items)

    raise ValueError(
        f"{source}:{row_number}: {field_name} must be a list or a string representation of a list"
    )


def _maybe_parse_json_array(text: str) -> list[object] | None:
    """Parse a JSON array string, returning None when the value is not a JSON array."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list):
        return parsed
    return None


def _clean_list_items(items: list[object]) -> list[str]:
    """Convert list items to stripped strings and drop blanks."""
    cleaned: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _row_is_blank(row: Mapping[str, object]) -> bool:
    """Return True when every value in a CSV row is empty."""
    return all((value is None) or (str(value).strip() == "") for key, value in row.items() if key is not None)
