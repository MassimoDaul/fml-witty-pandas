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


def cmd_backfill_authors(args) -> None:
    from tqdm import tqdm
    from ingest.db import get_connection, upsert_authors
    from ingest.config import DB_BATCH

    conn = get_connection()
    print("Loading authors from papers table...")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT arxiv_id, authors FROM papers WHERE authors IS NOT NULL AND array_length(authors, 1) > 0"
        )
        rows = cur.fetchall()

    print(f"Found {len(rows):,} papers with authors. Building pairs...")

    pairs: list[tuple[str, str]] = []
    for arxiv_id, authors in rows:
        for name in authors:
            if name:
                pairs.append((name, arxiv_id))

    print(f"Total author-paper pairs: {len(pairs):,}")

    total_affected = 0
    with tqdm(total=len(pairs), unit="pair", desc="Upserting authors") as pbar:
        for i in range(0, len(pairs), DB_BATCH):
            batch = pairs[i : i + DB_BATCH]
            affected = upsert_authors(conn, batch)
            total_affected += affected
            pbar.update(len(batch))

    conn.close()
    print(f"\nDone. Author rows affected: {total_affected:,}")


def cmd_verify(args) -> None:
    from ingest.db import get_connection, get_counts, get_author_counts
    conn = get_connection()
    counts = get_counts(conn)
    author_counts = get_author_counts(conn)
    conn.close()
    print(f"Total papers:             {counts['total']:,}")
    print(f"Missing embeddings:       {counts['missing_embeddings']:,}")
    print(f"Total authors:            {author_counts['total_authors']:,}")
    print(f"Author-paper links:       {author_counts['total_author_paper_links']:,}")


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
    sub.add_parser("backfill-authors", help="Populate authors table from existing papers (run once)")

    args = parser.parse_args()

    dispatch = {
        "setup-db": cmd_setup_db,
        "download": cmd_download,
        "ingest": cmd_ingest,
        "build-index": cmd_build_index,
        "verify": cmd_verify,
        "backfill-authors": cmd_backfill_authors,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
