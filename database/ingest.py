"""
database/ingest.py

Two-pass ingest from a single S2 abstract shard:
  Pass 1 — stream one shard, stop at TARGET_N papers
  Pass 2 — batch Graph API for metadata (~50 requests)

Run from project root:
    python database/ingest.py
"""

import gzip
import json
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

API_BASE  = "https://api.semanticscholar.org/datasets/v1"
GRAPH_API = "https://api.semanticscholar.org/graph/v1"
API_KEY   = os.environ["S2_API_KEY"]
HEADERS   = {"x-api-key": API_KEY}

RELEASE_ID = "2026-03-10"

TARGET_N     = 25_000
BATCH_SIZE   = 500   # corpus IDs per Graph API request
API_WORKERS  = 5     # concurrent Graph API requests
API_SLEEP    = 0.15  # seconds between requests per worker
INSERT_BATCH = 500   # rows per DB insert

METADATA_FIELDS = "title,year,venue,s2FieldsOfStudy,citationCount,referenceCount,url"

# ── DB ────────────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_CONN_STRING"])


# ── STREAMING HELPER ──────────────────────────────────────────────────────────

def stream_gz_jsonl(url: str):
    with requests.get(url, headers=HEADERS, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            for line in gz:
                if line:
                    yield json.loads(line.decode("utf-8"))


# ── PASS 1: single shard ──────────────────────────────────────────────────────

def collect_abstracts(shard_urls: list[str]) -> dict[str, str]:
    url = random.choice(shard_urls)
    print(f"\nPass 1 — streaming one shard (target {TARGET_N:,})")
    abstract_map: dict[str, str] = {}
    for rec in tqdm(stream_gz_jsonl(url), desc="scanning"):
        cid      = str(rec.get("corpusid") or "").strip()
        abstract = str(rec.get("abstract") or "").strip()
        if cid and abstract:
            abstract_map[cid] = abstract
            if len(abstract_map) >= TARGET_N:
                break
    print(f"Abstracts collected: {len(abstract_map):,}")
    return abstract_map


# ── PASS 2: batch Graph API for metadata (~50 requests) ──────────────────────

_BACKOFF_BASE = 2.0
_BACKOFF_MAX  = 64.0
_MAX_RETRIES  = 7


def _post_with_backoff(**kwargs) -> requests.Response:
    """POST with exponential backoff + jitter on 429 or 5xx."""
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


def _fetch_batch(args: tuple) -> list[dict]:
    batch_ids, batch_corpus_ids = args
    resp = _post_with_backoff(
        url=f"{GRAPH_API}/paper/batch",
        headers=HEADERS,
        params={"fields": METADATA_FIELDS},
        json={"ids": batch_ids},
        timeout=60,
    )
    time.sleep(API_SLEEP)
    rows = []
    for cid, rec in zip(batch_corpus_ids, resp.json()):
        if rec is None:
            continue
        rows.append({
            "corpus_id":       cid,
            "s2_paper_id":     rec.get("paperId") or None,
            "url":             rec.get("url") or None,
            "title":           rec.get("title") or "",
            "year":            rec.get("year"),
            "venue":           rec.get("venue") or None,
            "fields_of_study": [f["category"] for f in (rec.get("s2FieldsOfStudy") or [])],
            "citation_count":  rec.get("citationCount") or 0,
            "reference_count": rec.get("referenceCount") or 0,
        })
    return rows


def collect_metadata(corpus_ids: list[str]) -> list[dict]:
    n_batches = -(-len(corpus_ids) // BATCH_SIZE)
    print(f"\nPass 2 — Graph API ({len(corpus_ids):,} papers, "
          f"{n_batches} requests, {API_WORKERS} workers)")
    batches = [
        (
            [f"CorpusId:{cid}" for cid in corpus_ids[i:i + BATCH_SIZE]],
            corpus_ids[i:i + BATCH_SIZE],
        )
        for i in range(0, len(corpus_ids), BATCH_SIZE)
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures = [pool.submit(_fetch_batch, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fetching"):
            rows.extend(fut.result())
    print(f"Metadata fetched: {len(rows):,}")
    return rows


# ── DB INSERT ─────────────────────────────────────────────────────────────────

def insert_papers(papers: list[dict], abstract_map: dict[str, str]):
    papers = [p for p in papers if abstract_map.get(p["corpus_id"])]
    print(f"\nInserting {len(papers):,} papers...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for i in tqdm(range(0, len(papers), INSERT_BATCH), desc="inserting"):
                batch = papers[i:i + INSERT_BATCH]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO papers
                        (corpus_id, s2_paper_id, url, title, abstract,
                         year, venue, citation_count, reference_count, fields_of_study)
                    VALUES %s
                    ON CONFLICT (corpus_id) DO NOTHING
                    """,
                    [
                        (
                            p["corpus_id"],
                            p["s2_paper_id"],
                            p["url"],
                            p["title"],
                            abstract_map.get(p["corpus_id"], ""),
                            p["year"],
                            p["venue"],
                            p["citation_count"],
                            p["reference_count"],
                            p["fields_of_study"],
                        )
                        for p in batch
                    ],
                )
        conn.commit()
        print(f"Done. {len(papers):,} rows inserted.")
    finally:
        conn.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    r = requests.get(
        f"{API_BASE}/release/{RELEASE_ID}/dataset/abstracts",
        headers=HEADERS, timeout=60,
    )
    r.raise_for_status()
    abstract_shards: list[str] = r.json()["files"]

    abstract_map = collect_abstracts(abstract_shards)
    papers       = collect_metadata(list(abstract_map.keys()))
    insert_papers(papers, abstract_map)


if __name__ == "__main__":
    main()
