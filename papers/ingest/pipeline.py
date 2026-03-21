import json
from datetime import date, datetime
from tqdm import tqdm

from .clean import build_embed_input
from .embed import embed_texts, load_model
from .checkpoint import read_checkpoint, write_checkpoint
from .db import get_connection, upsert_papers
from .config import ARXIV_JSON, DB_BATCH


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_date(version_created: str) -> date | None:
    """Parse arXiv version date string like 'Mon, 2 Jan 2023 12:00:00 GMT'."""
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            return datetime.strptime(version_created, fmt).date()
        except ValueError:
            continue
    return None


def _parse_record(raw: dict) -> dict | None:
    title = (raw.get("title") or "").strip()
    abstract = (raw.get("abstract") or "").strip()
    if not title or not abstract:
        return None

    categories = (raw.get("categories") or "").split()
    if not categories:
        return None

    authors = []
    for parts in raw.get("authors_parsed") or []:
        # parts: [last, first, suffix]
        name = " ".join(p for p in [parts[1], parts[0]] if p)
        if name:
            authors.append(name)

    versions = raw.get("versions") or []
    published = _parse_date(versions[0]["created"]) if versions else None

    updated = None
    raw_updated = raw.get("update_date")
    if raw_updated:
        try:
            updated = date.fromisoformat(raw_updated)
        except ValueError:
            pass

    return {
        "arxiv_id": raw["id"],
        "title": title,
        "abstract": abstract,
        "categories": categories,
        "authors": authors or None,
        "published": published,
        "updated": updated,
        "embed_input": build_embed_input(title, abstract),
        "embedding": None,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(qty: int, offset: int) -> None:
    if not ARXIV_JSON.exists():
        raise FileNotFoundError(
            f"Dataset not found at {ARXIV_JSON}\n"
            "Run:  python run.py download"
        )

    # Load model before opening DB so startup messages appear upfront
    load_model()
    conn = get_connection()

    start_line = read_checkpoint(offset)
    end_line = (offset + qty) if qty > 0 else None  # None = no limit

    total_display = qty if qty > 0 else None
    already_done = max(0, start_line - offset)

    total_inserted = 0
    total_skipped = 0
    pending: list[dict] = []

    def flush(up_to_line: int) -> None:
        nonlocal total_inserted
        texts = [r["embed_input"] for r in pending]
        embeddings = embed_texts(texts)
        for record, emb in zip(pending, embeddings):
            record["embedding"] = emb
        inserted = upsert_papers(conn, pending)
        total_inserted += inserted
        write_checkpoint(offset, up_to_line)
        pending.clear()

    print(f"\nStarting ingest  offset={offset}  qty={'all' if qty == 0 else qty}")
    if start_line > offset:
        print(f"  (resuming — skipping first {start_line - offset} lines)")

    with open(ARXIV_JSON, "r", encoding="utf-8") as f:
        # Seek to start_line by iterating (file is too large for random access)
        for _ in range(start_line):
            f.readline()

        with tqdm(
            total=total_display,
            initial=already_done,
            unit="paper",
            desc="Ingesting",
            dynamic_ncols=True,
        ) as pbar:
            for abs_line, line in enumerate(f, start=start_line):
                if end_line is not None and abs_line >= end_line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    total_skipped += 1
                    continue

                record = _parse_record(raw)
                if record is None:
                    total_skipped += 1
                    continue

                pending.append(record)
                pbar.update(1)

                if len(pending) >= DB_BATCH:
                    flush(abs_line + 1)

            # Flush any remaining records
            if pending:
                final_line = (end_line or abs_line)
                flush(final_line)

    conn.close()
    print(f"\nDone.  Inserted: {total_inserted}  Skipped/invalid: {total_skipped}")
