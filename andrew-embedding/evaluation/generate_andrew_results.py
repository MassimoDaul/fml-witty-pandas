"""
Generate andrew-results.jsonl by running all benchmark queries through the
Andrew embedding search pipeline.

Usage:
    python andrew-embedding/evaluation/generate_andrew_results.py
    python andrew-embedding/evaluation/generate_andrew_results.py \
        --queries evaluation/benchmark_queries.jsonl \
        --output submissions/andrew-results.jsonl --k 10 --nprobe 25
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from tqdm import tqdm

_HERE      = Path(__file__).resolve().parent          # andrew-embedding/evaluation/
_REPO_ROOT = _HERE.parent.parent                      # project root
sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "andrew_query", _HERE.parent / "query.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
search = _mod.search

RUN_ID = "andrew"


def load_queries(path: str) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate andrew-results.jsonl from benchmark queries.")
    parser.add_argument("--queries", default=str(_REPO_ROOT / "evaluation" / "benchmark_queries.jsonl"))
    parser.add_argument("--output",  default=str(_REPO_ROOT / "evaluation" / "andrew-results.jsonl"))
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--nprobe",  type=int, default=25)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as out_f:
        for q in tqdm(queries, desc="Querying"):
            query_id   = q["queryId"]
            query_text = q["query"]

            hits = search(query_text, k=args.k, nprobe=args.nprobe)

            results = [
                {
                    "rank":    rank,
                    "paperId": hit["corpus_id"],
                    "score":   round(1.0 - hit["dist"], 6),
                }
                for rank, hit in enumerate(hits, 1)
            ]

            record = {"runId": RUN_ID, "queryId": query_id, "results": results}
            out_f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(queries)} rows to {out_path}")


if __name__ == "__main__":
    main()
