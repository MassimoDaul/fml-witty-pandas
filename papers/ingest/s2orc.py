"""
S2ORC full-text utilities via the Semantic Scholar Datasets API.

The Datasets API provides presigned S3 URLs for each shard of the S2ORC
dataset.  Each shard is a gzip-compressed JSONL file where every line is:

    {
        "corpusid": 12345678,
        "externalids": {"DOI": "10.1234/example", "ArXiv": "...", ...},
        "content": {
            "source": "pdf",
            "text": "Introduction\n\nIn this paper..."
        }
    }

Reference: https://api.semanticscholar.org/datasets/v1
"""

import gzip
import json
import logging

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DATASETS_BASE = "https://api.semanticscholar.org/datasets/v1"
_DATASET_NAME = "s2orc"

# Cache shard URLs within a process so repeated flush() calls don't re-hit the API
_shard_url_cache: list[str] = []


def _get_shard_urls_cached(api_key: str) -> list[str]:
    if not _shard_url_cache:
        _shard_url_cache.extend(get_shard_urls(api_key))
    return _shard_url_cache


def get_shard_urls(api_key: str) -> list[str]:
    """
    Return the list of presigned S3 shard URLs for the latest S2ORC release.
    """
    resp = requests.get(
        f"{_DATASETS_BASE}/release/latest",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    release_id = resp.json()["release_id"]
    logger.info("Latest S2ORC release: %s", release_id)

    resp = requests.get(
        f"{_DATASETS_BASE}/release/{release_id}/dataset/{_DATASET_NAME}",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    files = resp.json().get("files", [])
    logger.info("Found %d S2ORC shards", len(files))
    return files


def iter_shard(url: str):
    """
    Stream a single gzip-compressed JSONL shard from a presigned S3 URL.
    Yields parsed JSON objects one at a time without buffering the full shard.
    The presigned URL is self-authenticating — no extra headers required.
    """
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            for raw_line in gz:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line in shard")


def fetch_s2orc_texts(records: list[dict], api_key: str) -> None:
    """
    Populate body_text in-place for records that have a DOI, using pre-extracted
    S2ORC text (no PDF download or parsing required).

    Streams S2ORC shards from the Datasets API and matches records by DOI.
    Stops as soon as all DOIs in the batch have been found.
    Records without a DOI, or not present in S2ORC, are left with body_text=None.
    """
    # Build DOI → record index map for this batch
    doi_to_idx: dict[str, int] = {}
    for i, r in enumerate(records):
        doi = r.get("doi")
        if doi:
            doi_to_idx[doi] = i

    if not doi_to_idx:
        return

    remaining = set(doi_to_idx)
    shard_urls = _get_shard_urls_cached(api_key)

    with tqdm(
        total=len(shard_urls),
        unit="shard",
        desc="  S2ORC full text",
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for url in shard_urls:
            if not remaining:
                break
            try:
                for rec in iter_shard(url):
                    doi = (rec.get("externalids") or {}).get("DOI")
                    if not doi or doi not in remaining:
                        continue
                    text = ((rec.get("content") or {}).get("text") or "").strip() or None
                    records[doi_to_idx[doi]]["body_text"] = text
                    remaining.discard(doi)
                    if not remaining:
                        break
            except Exception as exc:
                logger.warning("Error reading shard: %s", exc)
            pbar.update(1)

    found = len(doi_to_idx) - len(remaining)
    logger.info("S2ORC batch: %d/%d matched", found, len(doi_to_idx))
