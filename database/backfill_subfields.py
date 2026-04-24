"""
database/backfill_subfields.py

Backfills the subfields column (s2-fos-model categories only) for papers
already in the database. Safe to re-run — skips papers where subfields
is already populated.

Run from project root:
    python database/backfill_subfields.py
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

GRAPH_API    = "https://api.semanticscholar.org/graph/v1"
API_KEY      = os.environ["S2_API_KEY"]
HEADERS      = {"x-api-key": API_KEY}

BATCH_SIZE   = 500
API_WORKERS  = 5
API_SLEEP    = 0.15
UPDATE_BATCH = 500

_BACKOFF_BASE = 2.0
_BACKOFF_MAX  = 64.0
_MAX_RETRIES  = 7


def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_CONN_STRING"])


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
    raise RuntimeError(f"S2 API still failing after {_MAX_RETRIES} retries")


def _fetch_batch(args: tuple) -> list[tuple[str, list[str]]]:
    """Returns [(corpus_id, [subfield, ...]), ...] for one batch."""
    batch_ids, batch_corpus_ids = args
    resp = _post_with_backoff(
        url=f"{GRAPH_API}/paper/batch",
        headers=HEADERS,
        params={"fields": "s2FieldsOfStudy"},
        json={"ids": batch_ids},
        timeout=60,
    )
    time.sleep(API_SLEEP)
    results = []
    for cid, rec in zip(batch_corpus_ids, resp.json()):
        if rec is None:
            results.append((cid, []))
            continue
        subfields = [
            f["category"]
            for f in (rec.get("s2FieldsOfStudy") or [])
            if f.get("source") == "s2-fos-model"
        ]
        results.append((cid, subfields))
    return results


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT corpus_id FROM papers
                WHERE subfields IS NULL
                ORDER BY corpus_id
            """)
            corpus_ids = [row[0] for row in cur.fetchall()]

        print(f"{len(corpus_ids):,} papers need subfields backfill")
        if not corpus_ids:
            return

        batches = [
            (
                [f"CorpusId:{cid}" for cid in corpus_ids[i:i + BATCH_SIZE]],
                corpus_ids[i:i + BATCH_SIZE],
            )
            for i in range(0, len(corpus_ids), BATCH_SIZE)
        ]

        all_updates: list[tuple[list[str], str]] = []

        with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
            futures = [pool.submit(_fetch_batch, b) for b in batches]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="fetching"):
                for cid, subfields in fut.result():
                    all_updates.append((subfields, cid))

        print(f"Writing {len(all_updates):,} updates...")
        with conn.cursor() as cur:
            for i in tqdm(range(0, len(all_updates), UPDATE_BATCH), desc="updating"):
                batch = all_updates[i:i + UPDATE_BATCH]
                psycopg2.extras.execute_batch(
                    cur,
                    "UPDATE papers SET subfields = %s WHERE corpus_id = %s",
                    batch,
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM papers WHERE subfields IS NOT NULL AND array_length(subfields, 1) > 0")
            filled = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM papers WHERE subfields = '{}'")
            empty = cur.fetchone()[0]

        print(f"\nDone.")
        print(f"  with subfields:    {filled:,}")
        print(f"  empty (no s2-fos): {empty:,}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
