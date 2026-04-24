"""Load canonical paper metadata from the Postgres corpus database."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

__all__ = ["load_postgres_paper_metadata"]


def load_postgres_paper_metadata(
    conn_string: str | None = None,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load paper metadata keyed by canonical DB ``corpus_id``.

    ``paper_ids`` may contain canonical ``corpus_id`` values or Semantic Scholar
    ``s2_paper_id`` aliases. When omitted, all papers are loaded. Loading all
    25k rows is intentionally acceptable for the evaluator because it keeps run
    validation simple and catches unknown paper IDs before judging.
    """
    load_dotenv()
    connection_string = conn_string or os.environ.get("POSTGRES_CONN_STRING")
    if not connection_string:
        raise ValueError(
            "Postgres metadata source requires POSTGRES_CONN_STRING or --postgres-conn-string."
        )

    try:
        import psycopg2
        import psycopg2.extras
    except ImportError as exc:
        raise ValueError("Postgres metadata loading requires psycopg2-binary.") from exc

    conn = psycopg2.connect(connection_string)
    try:
        paper_columns = _fetch_table_columns(conn, "papers")
        has_author_table = _table_exists(conn, "paper_authors")
        query, params = _build_metadata_query(
            paper_columns,
            has_author_table=has_author_table,
            paper_ids=paper_ids,
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = _normalize_db_row(dict(row))
        paper_id = record["paper_id"]
        if paper_id in records:
            raise ValueError(f"Duplicate corpus_id {paper_id!r} returned from Postgres metadata query.")
        records[paper_id] = record

    return records


def _fetch_table_columns(conn: Any, table_name: str) -> set[str]:
    """Return column names for a public table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        return {row[0] for row in cur.fetchall()}


def _table_exists(conn: Any, table_name: str) -> bool:
    """Return whether a public table exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            (table_name,),
        )
        return bool(cur.fetchone()[0])


def _build_metadata_query(
    paper_columns: set[str],
    *,
    has_author_table: bool,
    paper_ids: list[str] | None,
) -> tuple[str, tuple[Any, ...]]:
    """Build a metadata query using only columns available in the live DB."""
    required = {"corpus_id", "title", "abstract", "year"}
    missing = sorted(required - paper_columns)
    if missing:
        raise ValueError(f"Postgres papers table is missing required columns: {', '.join(missing)}")

    select_exprs = [
        "p.corpus_id",
        _column_expr(paper_columns, "s2_paper_id", "NULL::text"),
        _column_expr(paper_columns, "url", "NULL::text"),
        "p.title",
        "p.abstract",
        "p.year",
        _column_expr(paper_columns, "venue", "NULL::text"),
        _column_expr(paper_columns, "citation_count", "0::integer"),
        _column_expr(paper_columns, "reference_count", "0::integer"),
        _column_expr(paper_columns, "fields_of_study", "ARRAY[]::text[]"),
        _column_expr(paper_columns, "subfields", "ARRAY[]::text[]"),
        _column_expr(paper_columns, "author_ids", "ARRAY[]::text[]"),
    ]
    group_exprs = [
        expr.split(" AS ", 1)[0]
        for expr in select_exprs
        if expr.startswith("p.")
    ]

    if has_author_table:
        select_exprs.append(
            """
            COALESCE(
                array_agg(pa.author_name ORDER BY pa.author_name)
                    FILTER (WHERE pa.author_name IS NOT NULL AND btrim(pa.author_name) <> ''),
                ARRAY[]::text[]
            ) AS author_names
            """
        )
        join_sql = "LEFT JOIN paper_authors pa ON pa.corpus_id = p.corpus_id"
    else:
        select_exprs.append("ARRAY[]::text[] AS author_names")
        join_sql = ""

    where_sql = ""
    params: tuple[Any, ...] = ()
    if paper_ids:
        if "s2_paper_id" in paper_columns:
            where_sql = "WHERE p.corpus_id = ANY(%s) OR p.s2_paper_id = ANY(%s)"
            params = (paper_ids, paper_ids)
        else:
            where_sql = "WHERE p.corpus_id = ANY(%s)"
            params = (paper_ids,)

    query = f"""
        SELECT
            {", ".join(select_exprs)}
        FROM papers p
        {join_sql}
        {where_sql}
        GROUP BY {", ".join(group_exprs)}
        ORDER BY p.corpus_id
    """
    return query, params


def _column_expr(columns: set[str], column_name: str, fallback_sql: str) -> str:
    """Return a SELECT expression for an optional paper column."""
    if column_name in columns:
        return f"p.{column_name}"
    return f"{fallback_sql} AS {column_name}"


def _normalize_db_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a database row into the evaluator metadata contract."""
    corpus_id = _required_text(row.get("corpus_id"), "corpus_id")
    title = _required_text(row.get("title"), "title")
    abstract = _required_text(row.get("abstract"), "abstract")
    year = _required_int(row.get("year"), "year")
    fields = _clean_string_list(row.get("fields_of_study"))
    subfields = _clean_string_list(row.get("subfields"))

    return {
        "paper_id": corpus_id,
        "corpus_id": corpus_id,
        "s2_paper_id": _optional_text(row.get("s2_paper_id")),
        "url": _optional_text(row.get("url")) or "",
        "title": title,
        "abstract": abstract,
        "authors": _clean_string_list(row.get("author_names")),
        "author_ids": _clean_string_list(row.get("author_ids")),
        "year": year,
        "venue": _optional_text(row.get("venue")) or "",
        "citation_count": _optional_int(row.get("citation_count"), default=0),
        "reference_count": _optional_int(row.get("reference_count"), default=0),
        "fields_of_study": fields,
        "subfields": subfields,
        "categories": fields,
    }


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f"Postgres metadata row has missing {field_name!r}.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_int(value: Any, field_name: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Postgres metadata row has missing integer {field_name!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Postgres metadata row has invalid integer {field_name!r}: {value!r}.") from exc


def _optional_int(value: Any, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_string_list(value: Any) -> list[str]:
    """Return a deduped list of non-empty strings while preserving order."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned
