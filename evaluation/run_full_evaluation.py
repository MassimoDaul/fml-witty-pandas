"""Run the full evaluation workflow and export reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.aggregate_metrics import aggregate_run_metrics
from evaluation.build_paper_tasks import build_unique_paper_judging_tasks
from evaluation.compute_query_metrics import compute_query_metrics_for_run
from evaluation.expand_run_with_metadata import expand_run_with_metadata
from evaluation.intrinsic_metrics import (
    DEFAULT_INTRINSIC_COLUMNS,
    compute_intrinsic_embedding_metrics,
)
from evaluation.load_and_validate_submission import load_and_validate_run_submission
from evaluation.load_paper_metadata import load_paper_metadata
from evaluation.load_postgres_paper_metadata import load_postgres_paper_metadata
from evaluation.load_queries import load_eval_queries
from evaluation.openai_judges import DEFAULT_OPENAI_JUDGE_MODEL, build_openai_judge_fns
from evaluation.paper_judging import score_paper_relevance_with_cache
from evaluation.pooled_metrics import add_pooled_metrics
from evaluation.range_judging import score_broad_exploration_range_with_cache
from evaluation.win_rate_judging import (
    score_query_win_rate_with_cache,
    summarize_win_rates,
)
from evaluation.write_reports import write_evaluation_reports

JudgeFn = Callable[[dict[str, Any]], dict[str, Any]]

__all__ = ["run_full_evaluation"]


def run_full_evaluation(
    query_path: str,
    paper_metadata_path: str | None,
    run_paths: list[str],
    paper_cache_path: str,
    range_cache_path: str,
    output_dir: str,
    paper_judge_fn: JudgeFn,
    range_judge_fn: JudgeFn,
    *,
    metadata_source: str = "file",
    postgres_conn_string: str | None = None,
    corpus_size: int | None = None,
    win_rate_cache_path: str | None = None,
    win_rate_judge_fn: JudgeFn | None = None,
    intrinsic_columns: list[str] | None = None,
    intrinsic_sample_limit: int | None = None,
    intrinsic_k: int = 10,
    intrinsic_pair_k: int = 20,
    intrinsic_nprobe: int = 25,
) -> dict[str, Any]:
    """Run the end-to-end benchmark workflow for one or more submitted runs.

    Args:
        query_path: Path to the evaluator query JSONL file.
        paper_metadata_path: Path to the frozen paper metadata file when using
            ``metadata_source='file'``.
        run_paths: Paths to one or more submitted run JSONL files.
        paper_cache_path: JSONL cache path for paper relevance judgments.
        range_cache_path: JSONL cache path for broad-query Exploration Range judgments.
        output_dir: Directory where reports will be written.
        paper_judge_fn: Callback used for paper-level judging.
        range_judge_fn: Callback used for broad-query list-level judging.
        metadata_source: ``file`` or ``postgres``.
        postgres_conn_string: Optional Postgres connection string override.
        corpus_size: Optional corpus size for coverage reporting.
        win_rate_cache_path: Optional JSONL cache for query-level comparison judgments.
        win_rate_judge_fn: Optional callback for query-level comparison judging.
        intrinsic_columns: Optional embedding columns for DB-backed intrinsic metrics.
        intrinsic_sample_limit: Optional sample limit for intrinsic diagnostics.
        intrinsic_k: k for metadata coherence intrinsic metrics.
        intrinsic_pair_k: k for eval_pairs intrinsic metrics.
        intrinsic_nprobe: pgvector IVF probe count for intrinsic diagnostics.

    Returns:
        Evaluation summaries and comparison reports.
    """
    queries = load_eval_queries(query_path)
    paper_metadata = _load_metadata(
        metadata_source=metadata_source,
        paper_metadata_path=paper_metadata_path,
        postgres_conn_string=postgres_conn_string,
    )
    if corpus_size is None:
        corpus_size = len(paper_metadata)

    expanded_by_run: dict[str, list[dict[str, Any]]] = {}
    expanded_runs: list[list[dict[str, Any]]] = []

    for run_path in run_paths:
        validated_run = load_and_validate_run_submission(
            run_path,
            queries,
            paper_metadata,
        )
        expanded_run = expand_run_with_metadata(
            validated_run,
            queries,
            paper_metadata,
        )
        run_id = _extract_run_id(expanded_run, run_path=run_path)
        if run_id in expanded_by_run:
            raise ValueError(f"Duplicate run_id {run_id!r} across run submissions.")
        expanded_by_run[run_id] = expanded_run
        expanded_runs.append(expanded_run)

    paper_tasks = build_unique_paper_judging_tasks(expanded_runs)
    paper_judgments = score_paper_relevance_with_cache(
        paper_tasks,
        paper_cache_path,
        paper_judge_fn,
    )
    range_judgments = score_broad_exploration_range_with_cache(
        expanded_runs,
        range_cache_path,
        range_judge_fn,
    )

    per_query_by_run: dict[str, list[dict[str, Any]]] = {}

    for run_id, expanded_run in expanded_by_run.items():
        per_query_rows = compute_query_metrics_for_run(
            expanded_run,
            paper_judgments,
            range_judgments,
        )
        per_query_by_run[run_id] = per_query_rows

    comparison_summary = add_pooled_metrics(
        per_query_by_run,
        expanded_by_run,
        paper_judgments,
        corpus_size=corpus_size,
    )

    aggregate_by_run: dict[str, dict[str, Any]] = {}
    for run_id, per_query_rows in per_query_by_run.items():
        aggregate_by_run[run_id] = aggregate_run_metrics(per_query_rows)

    win_rate_by_query: dict[str, dict[str, Any]] = {}
    win_rate_summary: dict[str, Any] = {}
    if win_rate_cache_path and win_rate_judge_fn:
        win_rate_by_query = score_query_win_rate_with_cache(
            expanded_by_run,
            per_query_by_run,
            win_rate_cache_path,
            win_rate_judge_fn,
        )
        win_rate_summary = summarize_win_rates(
            win_rate_by_query,
            sorted(expanded_by_run),
        )

    intrinsic_summary: dict[str, Any] = {}
    if intrinsic_columns:
        if metadata_source != "postgres":
            raise ValueError("Intrinsic metrics require --metadata-source=postgres.")
        intrinsic_summary = compute_intrinsic_embedding_metrics(
            postgres_conn_string,
            embedding_columns=intrinsic_columns,
            k=intrinsic_k,
            pair_k=intrinsic_pair_k,
            nprobe=intrinsic_nprobe,
            sample_limit=intrinsic_sample_limit,
        )

    write_evaluation_reports(
        per_query_by_run,
        aggregate_by_run,
        output_dir,
        comparison_summary=comparison_summary,
        win_rate_by_query=win_rate_by_query,
        win_rate_summary=win_rate_summary,
        intrinsic_summary=intrinsic_summary,
    )
    return {
        "aggregate_by_run": aggregate_by_run,
        "comparison_summary": comparison_summary,
        "win_rate_summary": win_rate_summary,
        "intrinsic_summary": intrinsic_summary,
    }


def _load_metadata(
    *,
    metadata_source: str,
    paper_metadata_path: str | None,
    postgres_conn_string: str | None,
) -> dict[str, dict[str, Any]]:
    """Load paper metadata from the configured source."""
    if metadata_source == "postgres":
        return load_postgres_paper_metadata(postgres_conn_string)
    if metadata_source == "file":
        if not paper_metadata_path:
            raise ValueError("--paper-metadata is required when --metadata-source=file.")
        return load_paper_metadata(paper_metadata_path)
    raise ValueError("metadata_source must be 'file' or 'postgres'.")


def _extract_run_id(expanded_run: list[dict[str, Any]], *, run_path: str) -> str:
    """Return the consistent run identifier from an expanded run."""
    if not expanded_run:
        raise ValueError(f"Run submission {run_path!r} produced no rows.")

    run_ids = {row.get("run_id") for row in expanded_run}
    if len(run_ids) != 1:
        raise ValueError(
            f"Run submission {run_path!r} must contain exactly one run_id, got {run_ids!r}."
        )

    run_id = next(iter(run_ids))
    if not isinstance(run_id, str):
        raise ValueError(f"Run submission {run_path!r} has a non-string run_id.")
    return run_id


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the full evaluation workflow.")
    parser.add_argument("--queries", required=True, help="Path to evaluator query JSONL.")
    parser.add_argument(
        "--paper-metadata",
        required=False,
        help="Path to the frozen paper metadata file.",
    )
    parser.add_argument(
        "--metadata-source",
        choices=("file", "postgres"),
        default="file",
        help="Where paper metadata should be loaded from.",
    )
    parser.add_argument(
        "--postgres-conn-string",
        help="Optional Postgres connection string override. Defaults to POSTGRES_CONN_STRING.",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        help="Optional corpus size override for coverage reporting.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run submission JSONL files.",
    )
    parser.add_argument(
        "--paper-cache",
        required=True,
        help="Path to the paper relevance cache JSONL file.",
    )
    parser.add_argument(
        "--range-cache",
        required=True,
        help="Path to the Exploration Range cache JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for exported CSV and Markdown reports.",
    )
    parser.add_argument(
        "--win-rate-cache",
        help="Optional query-level win-rate judgment cache JSONL file.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=("placeholder", "openai"),
        default="placeholder",
        help="LLM judge provider for CLI execution.",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help=f"OpenAI judge model. Defaults to OPENAI_JUDGE_MODEL or {DEFAULT_OPENAI_JUDGE_MODEL}.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        help="Optional reasoning effort for compatible OpenAI models.",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=1500,
        help="Max output tokens for each OpenAI judge call.",
    )
    parser.add_argument(
        "--intrinsic-columns",
        nargs="*",
        help=(
            "Optional embedding columns for DB-backed intrinsic metrics. "
            f"Defaults to {', '.join(DEFAULT_INTRINSIC_COLUMNS)} when the flag is used without values."
        ),
    )
    parser.add_argument(
        "--intrinsic-sample-limit",
        type=int,
        help="Optional sample limit for intrinsic metrics.",
    )
    parser.add_argument("--intrinsic-k", type=int, default=10)
    parser.add_argument("--intrinsic-pair-k", type=int, default=20)
    parser.add_argument("--intrinsic-nprobe", type=int, default=25)
    return parser


def _placeholder_paper_judge(_: dict[str, Any]) -> dict[str, Any]:
    """Placeholder paper judge callback for the CLI."""
    raise NotImplementedError(
        "Paper judge callback is not wired in yet. "
        "Replace _placeholder_paper_judge with your real LLM judge function."
    )


def _placeholder_range_judge(_: dict[str, Any]) -> dict[str, Any]:
    """Placeholder range judge callback for the CLI."""
    raise NotImplementedError(
        "Range judge callback is not wired in yet. "
        "Replace _placeholder_range_judge with your real LLM judge function."
    )


def _placeholder_win_rate_judge(_: dict[str, Any]) -> dict[str, Any]:
    """Placeholder query-level comparison judge callback for the CLI."""
    raise NotImplementedError(
        "Win-rate judge callback is not wired in yet. "
        "Replace _placeholder_win_rate_judge with your real LLM comparison judge."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    paper_judge_fn, range_judge_fn, win_rate_judge_fn = _build_judge_callbacks(args)

    try:
        aggregate_by_run = run_full_evaluation(
            query_path=args.queries,
            paper_metadata_path=args.paper_metadata,
            run_paths=args.runs,
            paper_cache_path=args.paper_cache,
            range_cache_path=args.range_cache,
            output_dir=args.output_dir,
            paper_judge_fn=paper_judge_fn,
            range_judge_fn=range_judge_fn,
            metadata_source=args.metadata_source,
            postgres_conn_string=args.postgres_conn_string,
            corpus_size=args.corpus_size,
            win_rate_cache_path=args.win_rate_cache,
            win_rate_judge_fn=win_rate_judge_fn if args.win_rate_cache else None,
            intrinsic_columns=(
                list(DEFAULT_INTRINSIC_COLUMNS)
                if args.intrinsic_columns == []
                else args.intrinsic_columns
            ),
            intrinsic_sample_limit=args.intrinsic_sample_limit,
            intrinsic_k=args.intrinsic_k,
            intrinsic_pair_k=args.intrinsic_pair_k,
            intrinsic_nprobe=args.intrinsic_nprobe,
        )
    except (ValueError, NotImplementedError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(json.dumps(aggregate_by_run, indent=2, sort_keys=True))
    return 0


def _build_judge_callbacks(args: argparse.Namespace) -> tuple[JudgeFn, JudgeFn, JudgeFn | None]:
    """Build judge callbacks for the CLI."""
    if args.judge_provider == "openai":
        paper_judge_fn, range_judge_fn, win_rate_judge_fn = build_openai_judge_fns(
            model=args.openai_model,
            api_key_env=args.openai_api_key_env,
            reasoning_effort=args.openai_reasoning_effort,
            max_output_tokens=args.openai_max_output_tokens,
        )
        return paper_judge_fn, range_judge_fn, win_rate_judge_fn

    return (
        _placeholder_paper_judge,
        _placeholder_range_judge,
        _placeholder_win_rate_judge,
    )


if __name__ == "__main__":
    raise SystemExit(main())
