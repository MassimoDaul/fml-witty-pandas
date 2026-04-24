"""
database/enrich.py

Enriches existing papers with author and reference data from S2,
then builds precomputed eval_pairs for the evaluation layer.

Creates three new tables:
  paper_authors    (corpus_id, author_id, author_name)
  paper_references (corpus_id, ref_corpus_id)   -- ref need not be in papers
  eval_pairs       (query_id, target_id, pair_type, weight)

eval_pairs stores both directions (A→B and B→A) so eval lookups are
a simple WHERE query_id = X with no OR across columns.

Run from project root:
    python database/enrich.py
"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

GRAPH_API = "https://api.semanticscholar.org/graph/v1"
API_KEY   = os.environ["S2_API_KEY"]
HEADERS   = {"x-api-key": API_KEY}

BATCH_SIZE   = 500    # corpus IDs per Graph API request
API_WORKERS  = 5      # concurrent requests
API_SLEEP    = 0.15   # seconds between requests per worker
INSERT_BATCH = 2_000  # rows per DB insert (larger ok for non-vector tables)
MIN_COUPLING = 5      # min shared references to create a coupling pair

ENRICH_FIELDS = "authors,references.externalIds"

# ── DB ────────────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_CONN_STRING"])


def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_authors (
                corpus_id   TEXT NOT NULL REFERENCES papers(corpus_id) ON DELETE CASCADE,
                author_id   TEXT NOT NULL,
                author_name TEXT,
                PRIMARY KEY (corpus_id, author_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_references (
                corpus_id     TEXT NOT NULL REFERENCES papers(corpus_id) ON DELETE CASCADE,
                ref_corpus_id TEXT NOT NULL,
                PRIMARY KEY (corpus_id, ref_corpus_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_pairs (
                query_id  TEXT NOT NULL REFERENCES papers(corpus_id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES papers(corpus_id) ON DELETE CASCADE,
                pair_type TEXT    NOT NULL,  -- 'author' | 'coupling'
                weight    FLOAT   NOT NULL,
                PRIMARY KEY (query_id, target_id, pair_type)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_authors_author_id "
            "ON paper_authors (author_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_refs_ref_corpus_id "
            "ON paper_references (ref_corpus_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_pairs_query_id "
            "ON eval_pairs (query_id)"
        )
    conn.commit()
    print("Tables ready.")


# ── BACKOFF ───────────────────────────────────────────────────────────────────

_BACKOFF_BASE = 2.0
_BACKOFF_MAX  = 64.0
_MAX_RETRIES  = 7


def _post_with_backoff(**kwargs) -> requests.Response:
    for attempt in range(_MAX_RETRIES):
        resp = requests.post(**kwargs)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429 or resp.status_code >= 500:
            wait_s = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX) + random.random()
            print(f"  {resp.status_code} — retrying in {wait_s:.1f}s "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait_s)
        else:
            resp.raise_for_status()
    raise RuntimeError(f"S2 API still returning errors after {_MAX_RETRIES} retries")


# ── FETCH ─────────────────────────────────────────────────────────────────────

def _fetch_batch(args: tuple) -> tuple[list[dict], list[dict]]:
    batch_ids, batch_corpus_ids = args
    resp = _post_with_backoff(
        url=f"{GRAPH_API}/paper/batch",
        headers=HEADERS,
        params={"fields": ENRICH_FIELDS},
        json={"ids": batch_ids},
        timeout=60,
    )
    time.sleep(API_SLEEP)

    author_rows: list[dict] = []
    ref_rows:    list[dict] = []

    for cid, rec in zip(batch_corpus_ids, resp.json()):
        if rec is None:
            continue
        for author in (rec.get("authors") or []):
            aid = str(author.get("authorId") or "").strip()
            if aid:
                author_rows.append({
                    "corpus_id":   cid,
                    "author_id":   aid,
                    "author_name": (author.get("name") or "").strip() or None,
                })
        for ref in (rec.get("references") or []):
            ext     = ref.get("externalIds") or {}
            ref_cid = str(ext.get("CorpusId") or "").strip()
            if ref_cid:
                ref_rows.append({"corpus_id": cid, "ref_corpus_id": ref_cid})

    return author_rows, ref_rows


def fetch_and_store(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT corpus_id FROM papers ORDER BY corpus_id")
        corpus_ids = [row[0] for row in cur.fetchall()]

    n_batches = -(-len(corpus_ids) // BATCH_SIZE)
    print(f"\nFetching enrichment for {len(corpus_ids):,} papers "
          f"({n_batches} requests, {API_WORKERS} workers)...")

    batches = [
        (
            [f"CorpusId:{cid}" for cid in corpus_ids[i:i + BATCH_SIZE]],
            corpus_ids[i:i + BATCH_SIZE],
        )
        for i in range(0, len(corpus_ids), BATCH_SIZE)
    ]

    all_authors: list[dict] = []
    all_refs:    list[dict] = []

    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures = [pool.submit(_fetch_batch, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetching"):
            a_rows, r_rows = fut.result()
            all_authors.extend(a_rows)
            all_refs.extend(r_rows)

    print(f"  authors fetched:    {len(all_authors):,}")
    print(f"  references fetched: {len(all_refs):,}")

    _bulk_insert(conn, "paper_authors",    ["corpus_id", "author_id", "author_name"], all_authors)
    _bulk_insert(conn, "paper_references", ["corpus_id", "ref_corpus_id"],            all_refs)


def sync_author_ids(conn):
    """Denormalize paper_authors → papers.author_ids for fast GIN lookups and graph construction."""
    print("Syncing author_ids to papers table...")
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE papers p
            SET author_ids = sub.ids
            FROM (
                SELECT corpus_id, array_agg(author_id) AS ids
                FROM paper_authors
                GROUP BY corpus_id
            ) sub
            WHERE p.corpus_id = sub.corpus_id
        """)
        updated = cur.rowcount
    conn.commit()
    print(f"  updated: {updated:,} rows")


def _bulk_insert(conn, table: str, columns: list[str], rows: list[dict]):
    if not rows:
        return
    print(f"Inserting {len(rows):,} rows into {table}...")
    col_str = ", ".join(columns)
    with conn.cursor() as cur:
        for i in tqdm(range(0, len(rows), INSERT_BATCH), desc=f"  {table}"):
            batch = rows[i:i + INSERT_BATCH]
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {table} ({col_str}) VALUES %s ON CONFLICT DO NOTHING",
                [[r[c] for c in columns] for r in batch],
            )
    conn.commit()


# ── EVAL PAIRS ────────────────────────────────────────────────────────────────

def build_eval_pairs(conn):
    """
    Populates eval_pairs in both directions (A→B and B→A) so eval.py can
    do a simple WHERE query_id = X lookup.

    Author pairs:   weight = number of shared authors (almost always 1)
    Coupling pairs: weight = number of shared references (>= MIN_COUPLING)

    The coupling join self-joins paper_references on ref_corpus_id — the shared
    reference does not need to be in the papers table, only the two papers do.
    """
    print("\nBuilding author pairs...")
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO eval_pairs (query_id, target_id, pair_type, weight)
            SELECT
                a1.corpus_id,
                a2.corpus_id,
                'author',
                COUNT(*)::float
            FROM paper_authors a1
            JOIN paper_authors a2
              ON  a1.author_id  = a2.author_id
              AND a1.corpus_id != a2.corpus_id
            GROUP BY a1.corpus_id, a2.corpus_id
            ON CONFLICT (query_id, target_id, pair_type) DO NOTHING
        """)
        author_count = cur.rowcount
    conn.commit()
    print(f"  inserted: {author_count:,}")

    print(f"Building coupling pairs (MIN_COUPLING={MIN_COUPLING})...")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM eval_pairs WHERE pair_type = 'coupling'")
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO eval_pairs (query_id, target_id, pair_type, weight)
            SELECT
                r1.corpus_id,
                r2.corpus_id,
                'coupling',
                COUNT(*)::float
            FROM paper_references r1
            JOIN paper_references r2
              ON  r1.ref_corpus_id  = r2.ref_corpus_id
              AND r1.corpus_id     != r2.corpus_id
            GROUP BY r1.corpus_id, r2.corpus_id
            HAVING COUNT(*) >= {MIN_COUPLING}
            ON CONFLICT (query_id, target_id, pair_type) DO NOTHING
        """)
        coupling_count = cur.rowcount
    conn.commit()
    print(f"  inserted: {coupling_count:,}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pair_type, COUNT(*), AVG(weight), MAX(weight)
            FROM eval_pairs
            GROUP BY pair_type
            ORDER BY pair_type
        """)
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM paper_authors")
        n_authors = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM paper_references")
        n_refs = cur.fetchone()[0]

    print(f"\n{'─'*50}")
    print(f"paper_authors:    {n_authors:>10,} rows")
    print(f"paper_references: {n_refs:>10,} rows")
    print(f"\neval_pairs:")
    print(f"  {'type':12s}  {'count':>10s}  {'avg_weight':>12s}  {'max_weight':>10s}")
    for pair_type, count, avg_w, max_w in rows:
        print(f"  {pair_type:12s}  {count:>10,}  {avg_w:>12.2f}  {max_w:>10.0f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    conn = get_connection()
    try:
        create_tables(conn)
        fetch_and_store(conn)
        sync_author_ids(conn)
        build_eval_pairs(conn)
        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
