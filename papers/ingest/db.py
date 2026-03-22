from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from .config import POSTGRES_CONN_STRING


def get_connection():
    conn = psycopg2.connect(
        POSTGRES_CONN_STRING,
        options="-c statement_timeout=0",  # needed for long index builds
    )
    register_vector(conn)
    return conn


def setup_schema(conn) -> None:
    sql_path = Path(__file__).parent.parent / "setup_db.sql"
    sql = sql_path.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("Schema ready.")


def upsert_papers(conn, rows: list[dict]) -> int:
    """
    Insert rows, skip on conflict. Returns number of rows inserted.
    Each row dict must have: arxiv_id, title, abstract, categories,
    authors, published, updated, embedding.
    """
    data = [
        (
            r["arxiv_id"],
            r["title"],
            r["abstract"],
            r["categories"],
            r["authors"],
            r["published"],
            r["updated"],
            r["embedding"],
        )
        for r in rows
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO papers
                (arxiv_id, title, abstract, categories, authors, published, updated, embedding)
            VALUES %s
            ON CONFLICT (arxiv_id) DO NOTHING
            """,
            data,
        )
        inserted = cur.rowcount
    conn.commit()
    return inserted


def build_hnsw_index(conn) -> None:
    print("Building HNSW index — this may take several minutes...")
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS papers_embedding_hnsw_idx
                ON papers
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 128)
            """
        )
    conn.commit()
    print("Index built.")


def upsert_authors(conn, pairs: list[tuple[str, str]]) -> int:
    """
    Insert/update authors from (name, arxiv_id) pairs.
    On conflict, appends new arxiv_ids and deduplicates the papers array.
    Returns number of rows affected.
    """
    if not pairs:
        return 0

    # Group by author name
    by_name: dict[str, list[str]] = {}
    for name, arxiv_id in pairs:
        by_name.setdefault(name, []).append(arxiv_id)

    data = [(name, ids) for name, ids in by_name.items()]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO authors (name, papers)
            VALUES %s
            ON CONFLICT (name) DO UPDATE
                SET papers = (
                    SELECT array_agg(DISTINCT x)
                    FROM unnest(authors.papers || EXCLUDED.papers) x
                )
            """,
            data,
        )
        affected = cur.rowcount
    conn.commit()
    return affected


def get_author_counts(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM authors")
        total_authors = cur.fetchone()[0]
        cur.execute("SELECT COALESCE(SUM(array_length(papers, 1)), 0) FROM authors")
        total_links = cur.fetchone()[0]
    return {"total_authors": total_authors, "total_author_paper_links": total_links}


def get_counts(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM papers")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM papers WHERE embedding IS NULL")
        missing_embeddings = cur.fetchone()[0]
    return {"total": total, "missing_embeddings": missing_embeddings}
