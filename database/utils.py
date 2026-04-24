"""
database/utils.py

Shared utilities for embedding contributors: read/write embeddings,
manage IVF indexes, and run similarity search.

Requires: psycopg2-binary, pgvector, numpy, python-dotenv
Env var:  POSTGRES_CONN_STRING
"""

import os
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PGConnection

load_dotenv()

# Maps column name → (stored_type, dimension).
EMBEDDING_COLS: dict[str, tuple[str, int]] = {
    "nomic":            ("vector", 384),
    "massimo_title":    ("vector", 384),
    "massimo_abstract": ("vector", 384),
    "massimo_metadata": ("vector", 384),
    "andrew":           ("vector", 128),
    "audrey":           ("vector", 384),
    "autoresearch":     ("vector", 128),
    "autoresearch_new": ("vector", 128),
}

# IVF tuning defaults for ~25k papers (lists = rows / 1000).
NLIST: int = 25
_INSERT_BATCH: int = 500


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> PGConnection:
    """Return a new psycopg2 connection with pgvector types registered."""
    conn = psycopg2.connect(os.environ["POSTGRES_CONN_STRING"])
    register_vector(conn)
    return conn


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_col(column: str) -> None:
    if column not in EMBEDDING_COLS:
        raise ValueError(f"Unknown embedding column {column!r}. Valid columns: {sorted(EMBEDDING_COLS)}")

def _validate_int(name: str, value: int, lo: int, hi: int) -> None:
    if not isinstance(value, int) or not (lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}], got {value!r}")


# ── Embedding access ──────────────────────────────────────────────────────────

def fetch_embeddings(
    conn: PGConnection,
    column: str,
    corpus_ids: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Return {corpus_id: embedding_array} for the given column.

    Args:
        conn:       Active database connection from get_connection().
        column:     One of EMBEDDING_COLS.
        corpus_ids: Subset of corpus IDs to fetch. Fetches all non-NULL rows
                    when None.

    Returns:
        Dict mapping corpus_id to a float32 numpy array. Shape is (384,) for
        384-dim columns and (128,) for andrew. NULL rows are excluded.
    """
    _validate_col(column)
    with conn.cursor() as cur:
        if corpus_ids is None:
            cur.execute(
                f"SELECT corpus_id, {column} FROM papers WHERE {column} IS NOT NULL"
            )
        else:
            cur.execute(
                f"SELECT corpus_id, {column} FROM papers "
                f"WHERE corpus_id = ANY(%s) AND {column} IS NOT NULL",
                (corpus_ids,),
            )
        return {row[0]: np.array(row[1].tolist(), dtype=np.float32) for row in cur.fetchall()}


def get_unembedded(conn: PGConnection, column: str) -> list[str]:
    """
    Return corpus_ids where the given embedding column is NULL.

    Use this to discover which papers still need embeddings before running
    your model. Re-running after upsert_embeddings should return an empty list.

    Args:
        conn:   Active database connection from get_connection().
        column: One of EMBEDDING_COLS.

    Returns:
        List of corpus_id strings with no embedding in this column.
    """
    _validate_col(column)
    with conn.cursor() as cur:
        cur.execute(f"SELECT corpus_id FROM papers WHERE {column} IS NULL")
        return [row[0] for row in cur.fetchall()]


# ── Embedding writes ──────────────────────────────────────────────────────────

def upsert_embeddings(
    conn: PGConnection,
    column: str,
    id_vector_pairs: list[tuple[str, np.ndarray]],
    batch_size: int = _INSERT_BATCH,
) -> None:
    """
    Write embeddings to the given column for each (corpus_id, vector) pair.

    Always call drop_ivf_indexes(conn, column) before this and
    build_ivf_indexes(conn, column) after a bulk write. IVF centroids are
    computed once at index-build time; writing rows without rebuilding leaves
    the index's cluster structure stale.

    Args:
        conn:           Active database connection from get_connection().
        column:         One of EMBEDDING_COLS.
        id_vector_pairs: List of (corpus_id, 768-dim float32 array) tuples.
        batch_size:     Rows per UPDATE batch (default 500).
    """
    _validate_col(column)
    with conn.cursor() as cur:
        for i in range(0, len(id_vector_pairs), batch_size):
            batch = id_vector_pairs[i : i + batch_size]
            psycopg2.extras.execute_batch(
                cur,
                f"UPDATE papers SET {column} = %s WHERE corpus_id = %s",
                [(np.array(vec, dtype=np.float32), cid) for cid, vec in batch],
            )
    conn.commit()


# ── Index management ──────────────────────────────────────────────────────────

def build_ivf_indexes(conn: PGConnection, column: str, nlist: int = NLIST) -> None:
    """
    Create an IVF Flat index on the given embedding column using vector cosine distance.

    Call this AFTER the column's embeddings are fully written. Rows with a NULL
    embedding are excluded from the index automatically. Building on partial
    data produces poor centroids that do not improve as more rows are added —
    always wait until the column is complete.

    Args:
        conn:   Active database connection from get_connection().
        column: One of EMBEDDING_COLS.
        nlist:  Number of IVF cluster centroids. Default 25 is tuned for ~25k papers.
    """
    _validate_col(column)
    _validate_int("nlist", nlist, 1, 32768)
    col_expr = f"{column} vector_cosine_ops"
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '32MB'")
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS papers_{column}_ivf
            ON papers
            USING ivfflat ({col_expr})
            WITH (lists = {nlist})
        """)
    conn.commit()


def drop_ivf_indexes(conn: PGConnection, column: str) -> None:
    """
    Drop the IVF index on the given embedding column.

    Call this before bulk-writing embeddings. The standard workflow is:
        drop_ivf_indexes → upsert_embeddings → build_ivf_indexes
    """
    _validate_col(column)
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS papers_{column}_ivf")
    conn.commit()


def list_indexes(conn: PGConnection) -> list[dict[str, str]]:
    """
    Return info on which embedding columns currently have IVF indexes.

    Args:
        conn: Active database connection from get_connection().

    Returns:
        List of {"index": index_name, "size": human-readable disk size}.
        Empty list if no IVF indexes exist.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                indexname,
                pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
            FROM pg_indexes
            WHERE tablename = 'papers'
              AND indexname LIKE '%_ivf'
            ORDER BY indexname
        """)
        return [{"index": row[0], "size": row[1]} for row in cur.fetchall()]


# ── Search ────────────────────────────────────────────────────────────────────

def search_similar(
    conn: PGConnection,
    column: str,
    query_vec: np.ndarray,
    k: int = 10,
    nprobe: int = 10,
) -> list[dict]:
    """
    Return the k nearest papers to query_vec by approximate cosine distance.

    Requires an IVF index on the column (see build_ivf_indexes). Without the
    index the query falls back to an exact sequential scan, which is slow.

    Args:
        conn:      Active database connection from get_connection().
        column:    One of EMBEDDING_COLS.
        query_vec: Query embedding matching the column's dimension (384 for
                   halfvec columns, 128 for andrew). Should be L2-normalised.
        k:         Number of nearest neighbours to return.
        nprobe:    IVF clusters to probe per query. Higher = better recall at
                   the cost of latency. At nlist=25 on 25k vectors:
                     nprobe=10 → ~95% recall   (recommended default)
                     nprobe=25 → ~99% recall   (exact at nlist=25)

    Returns:
        List of {"corpus_id": str, "title": str, "dist": float} sorted by
        ascending cosine distance (0.0 = identical, 2.0 = opposite).
    """
    _validate_col(column)
    _validate_int("k", k, 1, 10_000)
    _validate_int("nprobe", nprobe, 1, 32768)
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {nprobe}")
        cur.execute(
            f"""
            SELECT corpus_id, title, {column} <=> %s AS dist
            FROM papers
            WHERE {column} IS NOT NULL
            ORDER BY dist
            LIMIT %s
            """,
            (np.array(query_vec, dtype=np.float32), k),
        )
        return [
            {"corpus_id": row[0], "title": row[1], "dist": float(row[2])}
            for row in cur.fetchall()
        ]