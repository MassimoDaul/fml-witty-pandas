"""Write CSV and Markdown evaluation reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

__all__ = ["write_evaluation_reports"]

_PER_QUERY_FIELDS = (
    "run_id",
    "query_id",
    "query_type",
    "precision_at_10",
    "ndcg_at_10",
    "mean_relevance_at_10",
    "mean_relevance_norm_at_10",
    "high_relevance_rate_at_10",
    "weak_or_irrelevant_rate_at_10",
    "relevant_count_at_10",
    "high_relevance_count_at_10",
    "reciprocal_rank_at_10",
    "high_relevance_reciprocal_rank_at_10",
    "exploration_range_raw",
    "exploration_range_norm",
    "broad_summary_score",
    "field_jaccard_mean_at_10",
    "subfield_jaccard_mean_at_10",
    "field_diversity_at_10",
    "subfield_diversity_at_10",
    "field_subfield_diversity_at_10",
    "venue_pairwise_match_rate_at_10",
    "venue_diversity_count_at_10",
    "mean_citation_count_at_10",
    "mean_reference_count_at_10",
    "pooled_candidate_count",
    "pooled_relevant_count",
    "pooled_relevant_recall_at_10",
    "pooled_high_relevance_recall_at_10",
    "pooled_coverage_at_10",
    "unique_paper_count_at_10",
    "unique_relevant_count_at_10",
    "overlap_jaccard_mean_at_10",
)

_SUMMARY_FIELDS = (
    "query_count",
    "precision_at_10",
    "ndcg_at_10",
    "mean_relevance_at_10",
    "mean_relevance_norm_at_10",
    "high_relevance_rate_at_10",
    "weak_or_irrelevant_rate_at_10",
    "reciprocal_rank_at_10",
    "exploration_range_raw",
    "exploration_range_norm",
    "broad_summary_score",
    "field_subfield_diversity_at_10",
    "venue_pairwise_match_rate_at_10",
    "pooled_relevant_recall_at_10",
    "pooled_coverage_at_10",
    "unique_paper_count_at_10",
    "unique_relevant_count_at_10",
    "overlap_jaccard_mean_at_10",
)


def write_evaluation_reports(
    per_query_by_run: dict[str, list[dict[str, Any]]],
    aggregate_by_run: dict[str, dict[str, Any]],
    output_dir: str,
    *,
    comparison_summary: dict[str, Any] | None = None,
    win_rate_by_query: dict[str, dict[str, Any]] | None = None,
    win_rate_summary: dict[str, Any] | None = None,
    intrinsic_summary: dict[str, Any] | None = None,
) -> None:
    """Write per-query, overall, and query-type reports to ``output_dir``.

    Args:
        per_query_by_run: Per-query metric rows keyed by ``run_id``.
        aggregate_by_run: Aggregate summaries keyed by ``run_id``.
        output_dir: Directory where exported files will be written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_per_query_csv(
        per_query_by_run,
        output_path / "per_query_metrics.csv",
    )
    _write_overall_summary_csv(
        aggregate_by_run,
        output_path / "overall_summary.csv",
    )
    _write_query_type_summary_csv(
        aggregate_by_run,
        output_path / "query_type_summary.csv",
    )
    _write_summary_markdown(
        aggregate_by_run,
        output_path / "summary.md",
        comparison_summary=comparison_summary or {},
        win_rate_summary=win_rate_summary or {},
    )
    if comparison_summary:
        _write_comparison_summary(comparison_summary, output_path)
    if win_rate_by_query:
        _write_win_rate_per_query(win_rate_by_query, output_path / "win_rate_per_query.csv")
    if win_rate_summary:
        _write_win_rate_summary(win_rate_summary, output_path)
    if intrinsic_summary:
        _write_intrinsic_summary(intrinsic_summary, output_path)


def _write_per_query_csv(
    per_query_by_run: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Write one row per query per run."""
    rows: list[dict[str, Any]] = []
    for run_id in sorted(per_query_by_run):
        for row in sorted(per_query_by_run[run_id], key=lambda item: str(item.get("query_id", ""))):
            rows.append({field_name: _serialize_value(row.get(field_name)) for field_name in _PER_QUERY_FIELDS})

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PER_QUERY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_overall_summary_csv(
    aggregate_by_run: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write one overall summary row per run."""
    fieldnames = ["run_id", *_SUMMARY_FIELDS]
    rows: list[dict[str, Any]] = []

    for run_id in sorted(aggregate_by_run):
        overall = _require_summary_mapping(
            aggregate_by_run[run_id],
            "overall",
            run_id=run_id,
        )
        row = {"run_id": run_id}
        for field_name in _SUMMARY_FIELDS:
            row[field_name] = _serialize_value(overall.get(field_name))
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_query_type_summary_csv(
    aggregate_by_run: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write one query-type summary row per run."""
    fieldnames = ["run_id", "query_type", *_SUMMARY_FIELDS]
    rows: list[dict[str, Any]] = []

    for run_id in sorted(aggregate_by_run):
        by_query_type = _require_nested_mapping(
            aggregate_by_run[run_id],
            "by_query_type",
            run_id=run_id,
        )
        for query_type in ("broad", "specific"):
            query_type_summary = by_query_type.get(query_type)
            if not isinstance(query_type_summary, dict):
                raise ValueError(
                    f"aggregate_by_run[{run_id!r}]['by_query_type'][{query_type!r}] must be a dict"
                )

            row = {"run_id": run_id, "query_type": query_type}
            for field_name in _SUMMARY_FIELDS:
                row[field_name] = _serialize_value(query_type_summary.get(field_name))
            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(
    aggregate_by_run: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    comparison_summary: dict[str, Any],
    win_rate_summary: dict[str, Any],
) -> None:
    """Write a short human-readable Markdown summary."""
    lines: list[str] = ["# Evaluation Summary", ""]

    lines.append("## Runs Evaluated")
    if aggregate_by_run:
        for run_id in sorted(aggregate_by_run):
            lines.append(f"- `{run_id}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Overall Summary")
    lines.extend(
        _markdown_table(
            headers=["run_id", *_SUMMARY_FIELDS],
            rows=[
                [
                    run_id,
                    *[
                        _display_value(
                            _require_summary_mapping(
                                aggregate_by_run[run_id],
                                "overall",
                                run_id=run_id,
                            ).get(field_name)
                        )
                        for field_name in _SUMMARY_FIELDS
                    ],
                ]
                for run_id in sorted(aggregate_by_run)
            ],
        )
    )
    lines.append("")

    lines.append("## Query Type Summary")
    lines.extend(
        _markdown_table(
            headers=["run_id", "query_type", *_SUMMARY_FIELDS],
            rows=[
                [
                    run_id,
                    query_type,
                    *[
                        _display_value(
                            _require_nested_mapping(
                                aggregate_by_run[run_id],
                                "by_query_type",
                                run_id=run_id,
                            )[query_type].get(field_name)
                        )
                        for field_name in _SUMMARY_FIELDS
                    ],
                ]
                for run_id in sorted(aggregate_by_run)
                for query_type in ("broad", "specific")
            ],
        )
    )
    lines.append("")
    lines.append(
        "Best-by-bucket conclusions are allowed; different runs may be strongest for broad versus specific queries."
    )
    lines.append("")

    if comparison_summary:
        lines.append("## Corpus Coverage")
        lines.append(
            f"- Unique papers returned across all runs: "
            f"{_display_value(comparison_summary.get('overall_unique_papers_returned'))}"
        )
        lines.append(
            f"- Corpus coverage rate: "
            f"{_display_value(comparison_summary.get('overall_corpus_coverage_rate'))}"
        )
        lines.append("")

    if win_rate_summary:
        lines.append("## Win Rate")
        by_run = win_rate_summary.get("by_run", {})
        if isinstance(by_run, dict):
            lines.extend(
                _markdown_table(
                    headers=[
                        "run_id",
                        "strict_wins",
                        "strict_win_rate",
                        "fractional_wins",
                        "fractional_win_rate",
                    ],
                    rows=[
                        [
                            run_id,
                            _display_value(row.get("strict_wins")),
                            _display_value(row.get("strict_win_rate")),
                            _display_value(row.get("fractional_wins")),
                            _display_value(row.get("fractional_win_rate")),
                        ]
                        for run_id, row in sorted(by_run.items())
                        if isinstance(row, dict)
                    ],
                )
            )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_comparison_summary(comparison_summary: dict[str, Any], output_path: Path) -> None:
    """Write corpus coverage summary files."""
    (output_path / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    by_run = comparison_summary.get("by_run", {})
    if not isinstance(by_run, dict):
        return

    fieldnames = [
        "run_id",
        "unique_papers_returned",
        "corpus_size",
        "corpus_coverage_rate",
    ]
    rows = []
    for run_id, row in sorted(by_run.items()):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "run_id": run_id,
                "unique_papers_returned": _serialize_value(row.get("unique_papers_returned")),
                "corpus_size": _serialize_value(row.get("corpus_size")),
                "corpus_coverage_rate": _serialize_value(row.get("corpus_coverage_rate")),
            }
        )

    with (output_path / "corpus_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_win_rate_per_query(
    win_rate_by_query: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write one row per query-level comparison judgment."""
    fieldnames = [
        "query_id",
        "outcome",
        "winner_run_ids",
        "confidence",
        "short_rationale",
    ]
    rows = []
    for query_id, row in sorted(win_rate_by_query.items()):
        rows.append(
            {
                "query_id": query_id,
                "outcome": _serialize_value(row.get("outcome")),
                "winner_run_ids": ";".join(str(item) for item in row.get("winner_run_ids", [])),
                "confidence": _serialize_value(row.get("confidence")),
                "short_rationale": _serialize_value(row.get("short_rationale")),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_win_rate_summary(win_rate_summary: dict[str, Any], output_path: Path) -> None:
    """Write win-rate summary and pairwise preference reports."""
    (output_path / "win_rate_summary.json").write_text(
        json.dumps(win_rate_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    by_run = win_rate_summary.get("by_run", {})
    if isinstance(by_run, dict):
        fieldnames = [
            "run_id",
            "strict_wins",
            "strict_win_rate",
            "fractional_wins",
            "fractional_win_rate",
        ]
        rows = []
        for run_id, row in sorted(by_run.items()):
            if not isinstance(row, dict):
                continue
            rows.append({"run_id": run_id, **{field: _serialize_value(row.get(field)) for field in fieldnames[1:]}})
        with (output_path / "win_rate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    pairwise = win_rate_summary.get("pairwise", {})
    if isinstance(pairwise, dict):
        fieldnames = ["run_id", "opponent_run_id", "wins", "losses", "ties", "no_preference"]
        rows = []
        for run_id, opponents in sorted(pairwise.items()):
            if not isinstance(opponents, dict):
                continue
            for opponent_run_id, counts in sorted(opponents.items()):
                if not isinstance(counts, dict):
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "opponent_run_id": opponent_run_id,
                        "wins": _serialize_value(counts.get("wins")),
                        "losses": _serialize_value(counts.get("losses")),
                        "ties": _serialize_value(counts.get("ties")),
                        "no_preference": _serialize_value(counts.get("no_preference")),
                    }
                )
        with (output_path / "pairwise_preferences.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _write_intrinsic_summary(intrinsic_summary: dict[str, Any], output_path: Path) -> None:
    """Write DB-backed intrinsic embedding diagnostics."""
    (output_path / "intrinsic_metrics.json").write_text(
        json.dumps(intrinsic_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    columns = intrinsic_summary.get("columns", {})
    if not isinstance(columns, dict):
        return

    metric_names = sorted(
        {
            metric_name
            for metrics in columns.values()
            if isinstance(metrics, dict)
            for metric_name in metrics
        }
    )
    fieldnames = ["embedding_column", *metric_names]
    rows = []
    for column, metrics in sorted(columns.items()):
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "embedding_column": column,
                **{metric_name: _serialize_value(metrics.get(metric_name)) for metric_name in metric_names},
            }
        )

    with (output_path / "intrinsic_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a simple GitHub-flavored Markdown table."""
    if not rows:
        return ["No data available."]

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return [header_line, separator_line, *body_lines]


def _serialize_value(value: object) -> object:
    """Format values for CSV output without mutating internal representations."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _display_value(value: object) -> str:
    """Format values for Markdown tables."""
    serialized = _serialize_value(value)
    if serialized == "":
        return "N/A"
    return str(serialized)


def _require_summary_mapping(
    aggregate_row: dict[str, Any],
    field_name: str,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Read a required summary mapping from one aggregate row."""
    summary = aggregate_row.get(field_name)
    if not isinstance(summary, dict):
        raise ValueError(f"aggregate_by_run[{run_id!r}][{field_name!r}] must be a dict")
    return summary


def _require_nested_mapping(
    aggregate_row: dict[str, Any],
    field_name: str,
    *,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Read a required nested mapping from one aggregate row."""
    nested = aggregate_row.get(field_name)
    if not isinstance(nested, dict):
        raise ValueError(f"aggregate_by_run[{run_id!r}][{field_name!r}] must be a dict")
    return nested
