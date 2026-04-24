"""
K-nearest neighbor search over S2 papers using the nomic embedding column.

Usage:
    python query.py "machine learning for protein folding"
    python query.py "transformer attention mechanisms" --k 20
    python query.py "quantum computing algorithms" --k 5 --no-abstract
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from ingest.config import POSTGRES_CONN_STRING
from ingest.clean import clean_text
from ingest.embed import load_model

DEFAULT_K = 10

_model: SentenceTransformer | None = None


class PaperYear:
    """Wraps an integer year so templates can use .year, slicing, and str()."""
    def __init__(self, year: int):
        self.year = year
        self._s = str(year)

    def __str__(self):     return self._s
    def __bool__(self):    return True
    def __getitem__(self, idx): return self._s[idx]
    def toordinal(self):   return self.year * 365


def embed_query(text: str) -> list[float]:
    global _model
    if _model is None:
        _model = load_model()
    prompt = f"search_query: {clean_text(text)}"
    return _model.encode(prompt, normalize_embeddings=True).tolist()


def embed_document(title: str, abstract: str) -> list[float]:
    global _model
    if _model is None:
        _model = load_model()
    prompt = f"search_document: {clean_text(title)}. {clean_text(abstract)}"
    return _model.encode(prompt, normalize_embeddings=True).tolist()


def _row_to_paper(row) -> dict:
    corpus_id, s2_paper_id, url, title, abstract, fields, year, similarity = row
    return {
        "arxiv_id":   corpus_id,
        "s2_paper_id": s2_paper_id,
        "url":        url or "",
        "title":      title or "",
        "abstract":   abstract or "",
        "authors":    [],
        "categories": fields or [],
        "published":  PaperYear(year) if year else None,
        "similarity": float(similarity),
    }


def search(
    query: str,
    k: int = DEFAULT_K,
    conn=None,
    categories: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict]:
    close_after = conn is None
    if conn is None:
        conn = psycopg2.connect(POSTGRES_CONN_STRING)
        register_vector(conn)

    vec = np.array(embed_query(query), dtype=np.float32)

    where_clauses = ["nomic IS NOT NULL"]
    where_params: list = []
    if categories:
        where_clauses.append("fields_of_study && %s::text[]")
        where_params.append(categories)
    if year_from is not None:
        where_clauses.append("year >= %s")
        where_params.append(year_from)
    if year_to is not None:
        where_clauses.append("year <= %s")
        where_params.append(year_to)

    where_sql = "WHERE " + " AND ".join(where_clauses)

    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        cur.execute(
            f"""
            SELECT corpus_id, s2_paper_id, url, title, abstract, fields_of_study, year,
                   1 - (nomic <=> %s::vector) AS similarity
            FROM papers
            {where_sql}
            ORDER BY nomic <=> %s::vector
            LIMIT %s
            """,
            [vec] + where_params + [vec, k],
        )
        rows = cur.fetchall()

    if close_after:
        conn.close()

    return [_row_to_paper(r) for r in rows]


def related_search(
    title: str,
    abstract: str,
    k: int = 8,
    exclude_id: str = "",
    conn=None,
) -> list[dict]:
    close_after = conn is None
    if conn is None:
        conn = psycopg2.connect(POSTGRES_CONN_STRING)
        register_vector(conn)

    vec = np.array(embed_document(title, abstract), dtype=np.float32)

    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        cur.execute(
            """
            SELECT corpus_id, s2_paper_id, url, title, abstract, fields_of_study, year,
                   1 - (nomic <=> %s::vector) AS similarity
            FROM papers
            WHERE nomic IS NOT NULL AND corpus_id != %s
            ORDER BY nomic <=> %s::vector
            LIMIT %s
            """,
            (vec, exclude_id, vec, k),
        )
        rows = cur.fetchall()

    if close_after:
        conn.close()

    return [_row_to_paper(r) for r in rows]


def print_results(results: list[dict], show_abstract: bool = True) -> None:
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        fields = ", ".join(r["categories"]) if r["categories"] else "-"
        print(f"\n{'─' * 72}")
        print(f"#{i}  [{r['similarity']:.4f}]  {r['arxiv_id']}")
        print(f"    {r['title']}")
        print(f"    {r['published'] or '-'}  |  {fields}")
        if show_abstract:
            abstract = r["abstract"]
            if len(abstract) > 300:
                abstract = abstract[:300].rstrip() + "..."
            print(f"\n    {abstract}")

    print(f"\n{'─' * 72}")
    print(f"{len(results)} results")


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN search over S2 papers")
    parser.add_argument("query", help="Research topic or question")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--no-abstract", action="store_true")
    args = parser.parse_args()

    results = search(args.query, k=args.k)
    print_results(results, show_abstract=not args.no_abstract)


if __name__ == "__main__":
    main()
