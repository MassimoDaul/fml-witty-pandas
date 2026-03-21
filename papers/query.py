"""
K-nearest neighbor search over embedded arXiv papers.

Usage:
    python query.py "machine learning for protein folding"
    python query.py "transformer attention mechanisms" --k 20
    python query.py "quantum computing algorithms" --k 5 --no-abstract
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingest.config import POSTGRES_CONN_STRING
from ingest.embed import load_model
from ingest.clean import clean_text

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DEFAULT_K = 10

_model: SentenceTransformer | None = None


def embed_query(text: str) -> list[float]:
    global _model
    if _model is None:
        _model = load_model()
    # Queries use "search_query:" prefix (asymmetric pair with "search_document:")
    prompt = f"search_query: {clean_text(text)}"
    vector = _model.encode(prompt, normalize_embeddings=True)
    return vector.tolist()


def search(query: str, k: int = DEFAULT_K, conn=None) -> list[dict]:
    """
    Return the k most similar papers to query.

    Each result dict has:
        arxiv_id, title, abstract, categories, authors, published, similarity
    """
    close_after = conn is None
    if conn is None:
        conn = psycopg2.connect(POSTGRES_CONN_STRING)
        register_vector(conn)

    embedding = embed_query(query)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                arxiv_id,
                title,
                abstract,
                categories,
                authors,
                published,
                1 - (embedding <=> %s::vector) AS similarity
            FROM papers
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, k),
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

    if close_after:
        conn.close()

    return [dict(zip(cols, row)) for row in rows]


def print_results(results: list[dict], show_abstract: bool = True) -> None:
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        categories = ", ".join(r["categories"]) if r["categories"] else "-"
        authors = ", ".join(r["authors"]) if r["authors"] else "-"
        if r["authors"] and len(r["authors"]) > 3:
            authors += f" +{len(r['authors']) - 3} more"

        print(f"\n{'─' * 72}")
        print(f"#{i}  [{r['similarity']:.4f}]  {r['arxiv_id']}")
        print(f"    {r['title']}")
        print(f"    {authors}")
        print(f"    {categories}  |  {r['published'] or '-'}")
        if show_abstract:
            abstract = r["abstract"]
            if len(abstract) > 300:
                abstract = abstract[:300].rstrip() + "..."
            print(f"\n    {abstract}")

    print(f"\n{'─' * 72}")
    print(f"{len(results)} results")


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN search over arXiv papers")
    parser.add_argument("query", help="Research topic or question")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of nearest neighbors to return (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--no-abstract",
        action="store_true",
        help="Hide abstract snippets in output",
    )
    args = parser.parse_args()

    results = search(args.query, k=args.k)
    print_results(results, show_abstract=not args.no_abstract)


if __name__ == "__main__":
    main()
