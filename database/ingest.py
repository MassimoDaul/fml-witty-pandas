"""
database/ingest.py

Ingest from S2 Bulk Search API:
  - Pages through bulk search results (up to 1 000 per page) using token-based pagination
  - Filters CS papers with abstracts from MIN_YEAR+
  - Stops after TARGET_N papers

Run from project root:
    python database/ingest.py
"""

import os

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

BULK_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
API_KEY         = os.environ["S2_API_KEY"]
HEADERS         = {"x-api-key": API_KEY}

MIN_YEAR     = 2021
TARGET_N     = 25_000
INSERT_BATCH = 500

FIELDS = ",".join([
    "paperId",
    "externalIds",
    "url",
    "title",
    "abstract",
    "year",
    "venue",
    "journal",
    "s2FieldsOfStudy",
    "citationCount",
    "referenceCount",
])

# ── DB ────────────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_CONN_STRING"])


# ── COLLECT ───────────────────────────────────────────────────────────────────

def collect_papers() -> list[dict]:
    print(f"\nFetching CS papers from S2 Bulk Search API (year >= {MIN_YEAR})...")

    papers: list[dict] = []
    token:  str | None = None
    page   = 0

    params = {
        "query":               "",
        "fields":              FIELDS,
        "fieldsOfStudy":       "Computer Science",
        "publicationDateOrYear": f"{MIN_YEAR}:",
        "limit":               1000,
    }

    with tqdm(total=TARGET_N, desc="papers", unit="paper") as bar:
        while len(papers) < TARGET_N:
            if token:
                params["token"] = token
            elif "token" in params:
                del params["token"]

            resp = requests.get(BULK_SEARCH_API, headers=HEADERS, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            records = data.get("data") or []
            page += 1

            for rec in records:
                if len(papers) >= TARGET_N:
                    break

                abstract = (rec.get("abstract") or "").strip()
                if not abstract:
                    continue

                ext_ids   = rec.get("externalIds") or {}
                corpus_id = str(ext_ids.get("CorpusId") or "").strip()
                if not corpus_id:
                    continue

                year = rec.get("year")
                if not year:
                    continue

                journal = rec.get("journal") or {}
                venue   = rec.get("venue") or journal.get("name") or None

                papers.append({
                    "corpus_id":       corpus_id,
                    "s2_paper_id":     rec.get("paperId") or None,
                    "url":             rec.get("url") or (
                        f"https://www.semanticscholar.org/paper/{rec['paperId']}"
                        if rec.get("paperId") else None
                    ),
                    "title":           str(rec.get("title") or ""),
                    "abstract":        abstract,
                    "year":            int(year),
                    "venue":           venue,
                    "fields_of_study": [e["category"] for e in (rec.get("s2FieldsOfStudy") or [])],
                    "subfields":       [e["category"] for e in (rec.get("s2FieldsOfStudy") or [])
                                        if e.get("source") == "s2-fos-model"],
                    "citation_count":  rec.get("citationCount") or 0,
                    "reference_count": rec.get("referenceCount") or 0,
                })
                bar.update(1)

            token = data.get("token")
            print(f"  page {page}  |  got {len(records)} records  |  total so far {len(papers):,}")

            if not token:
                print("  No more pages from API.")
                break

    print(f"\nDone collecting: {len(papers):,} papers")
    return papers


# ── INSERT ────────────────────────────────────────────────────────────────────

def insert_papers(papers: list[dict]):
    print(f"\nInserting {len(papers):,} papers into Postgres...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for i in tqdm(range(0, len(papers), INSERT_BATCH), desc="inserting", unit="batch"):
                batch = papers[i:i + INSERT_BATCH]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO papers
                        (corpus_id, s2_paper_id, url, title, abstract,
                         year, venue, citation_count, reference_count, fields_of_study, subfields)
                    VALUES %s
                    ON CONFLICT (corpus_id) DO NOTHING
                    """,
                    [
                        (
                            p["corpus_id"],
                            p["s2_paper_id"],
                            p["url"],
                            p["title"],
                            p["abstract"],
                            p["year"],
                            p["venue"],
                            p["citation_count"],
                            p["reference_count"],
                            p["fields_of_study"],
                            p["subfields"],
                        )
                        for p in batch
                    ],
                )
        conn.commit()
        print(f"Inserted {len(papers):,} rows (duplicates skipped via ON CONFLICT).")
    finally:
        conn.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    papers = collect_papers()
    insert_papers(papers)


if __name__ == "__main__":
    main()
