"""
Entry point for the arXiv ingestion pipeline.

Commands:
    python run.py setup-db                          Create table in Supabase
    python run.py download                          Download Kaggle dataset
    python run.py ingest [--qty N] [--offset N]     Embed and upload papers
    python run.py build-index                       Build HNSW index post-ingest
    python run.py verify                            Print row counts

Defaults: --qty 10000 --offset 0
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Ensure the papers/ directory is on the path so ingest.* imports work
sys.path.insert(0, str(Path(__file__).parent))

from ingest.config import DEFAULT_QTY, DEFAULT_OFFSET, DATA_DIR, ARXIV_JSON


def cmd_setup_db(args) -> None:
    from ingest.db import get_connection, setup_schema
    conn = get_connection()
    setup_schema(conn)
    conn.close()


def cmd_download(args) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARXIV_JSON.exists():
        print(f"Dataset already exists at {ARXIV_JSON}")
        print("Delete it manually if you want to re-download.")
        return
    print("Downloading arXiv metadata from Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "Cornell-University/arxiv",
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        check=True,
    )
    if ARXIV_JSON.exists():
        size_gb = ARXIV_JSON.stat().st_size / 1e9
        print(f"Download complete. File size: {size_gb:.1f} GB")
    else:
        print("Warning: expected file not found after download. Check Kaggle output above.")


def cmd_ingest(args) -> None:
    from ingest.pipeline import run
    run(qty=args.qty, offset=args.offset)


def cmd_build_index(args) -> None:
    from ingest.db import get_connection, build_hnsw_index
    conn = get_connection()
    build_hnsw_index(conn)
    conn.close()


def cmd_verify(args) -> None:
    from ingest.db import get_connection, get_counts
    conn = get_connection()
    counts = get_counts(conn)
    conn.close()
    print(f"Total papers:        {counts['total']:,}")
    print(f"Missing embeddings:  {counts['missing_embeddings']:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="arXiv ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup-db", help="Create table in Supabase (run once)")
    sub.add_parser("download", help="Download Kaggle dataset to data/")

    ingest_p = sub.add_parser("ingest", help="Embed and upload papers")
    ingest_p.add_argument(
        "--qty",
        type=int,
        default=DEFAULT_QTY,
        help=f"Number of papers to process (0 = all). Default: {DEFAULT_QTY}",
    )
    ingest_p.add_argument(
        "--offset",
        type=int,
        default=DEFAULT_OFFSET,
        help=f"Line offset in the dataset to start from. Default: {DEFAULT_OFFSET}",
    )

    sub.add_parser("build-index", help="Build HNSW index after ingest is complete")
    sub.add_parser("verify", help="Print row counts from DB")

    args = parser.parse_args()

    dispatch = {
        "setup-db": cmd_setup_db,
        "download": cmd_download,
        "ingest": cmd_ingest,
        "build-index": cmd_build_index,
        "verify": cmd_verify,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
