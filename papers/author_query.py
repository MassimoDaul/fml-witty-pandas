"""
Author lookup against the authors table.

Usage:
    python author_query.py "Yann LeCun"
    python author_query.py "LeCun" --k 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingest.config import POSTGRES_CONN_STRING

import psycopg2
from pgvector.psycopg2 import register_vector

DEFAULT_K = 10


def search_by_name(name: str, k: int = DEFAULT_K, conn=None) -> list[dict]:
    """
    Return up to k authors whose name matches (case-insensitive substring).
    Each result dict has: name, paper_count, papers (list of {arxiv_id, title, published}).
    """
    close_after = conn is None
    if conn is None:
        conn = psycopg2.connect(POSTGRES_CONN_STRING)
        register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                a.name,
                array_length(a.papers, 1) AS paper_count,
                json_agg(
                    json_build_object(
                        'arxiv_id', p.arxiv_id,
                        'title', p.title,
                        'published', p.published
                    )
                    ORDER BY p.published DESC NULLS LAST
                ) AS papers
            FROM authors a
            LEFT JOIN papers p ON p.arxiv_id = ANY(a.papers)
            WHERE a.name ILIKE %s
            GROUP BY a.name, a.papers
            ORDER BY array_length(a.papers, 1) DESC, a.name
            LIMIT %s
            """,
            (f"%{name}%", k),
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

    if close_after:
        conn.close()

    return [dict(zip(cols, row)) for row in rows]


def papers_by_author(name: str, conn=None) -> list[dict]:
    """
    Return full paper records for papers written by the given author (exact name match).
    Each result dict has: arxiv_id, title, abstract, categories, authors, published.
    """
    close_after = conn is None
    if conn is None:
        conn = psycopg2.connect(POSTGRES_CONN_STRING)
        register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.arxiv_id, p.title, p.abstract, p.categories, p.authors, p.published
            FROM papers p
            JOIN authors a ON p.arxiv_id = ANY(a.papers)
            WHERE a.name = %s
            ORDER BY p.published DESC NULLS LAST
            """,
            (name,),
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

    if close_after:
        conn.close()

    return [dict(zip(cols, row)) for row in rows]


def print_results(author_name: str, papers: list[dict]) -> None:
    if not papers:
        print(f"No papers found for author: {author_name!r}")
        return

    print(f"\nPapers by {author_name!r} ({len(papers)} found):")
    for i, p in enumerate(papers, 1):
        categories = ", ".join(p["categories"]) if p["categories"] else "-"
        print(f"\n{'─' * 72}")
        print(f"#{i}  {p['arxiv_id']}  |  {p['published'] or '-'}")
        print(f"    {p['title']}")
        print(f"    {categories}")
    print(f"\n{'─' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Author lookup")
    parser.add_argument("name", help="Author name (substring match)")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Max authors to return from name search (default: {DEFAULT_K})",
    )
    args = parser.parse_args()

    matches = search_by_name(args.name, k=args.k)
    if not matches:
        print(f"No authors found matching {args.name!r}")
        return

    if len(matches) == 1:
        author = matches[0]
        papers = papers_by_author(author["name"])
        print_results(author["name"], papers)
    else:
        print(f"Found {len(matches)} matching authors:")
        for i, a in enumerate(matches, 1):
            print(f"  {i}. {a['name']}  ({len(a['papers'])} papers)")


if __name__ == "__main__":
    main()
