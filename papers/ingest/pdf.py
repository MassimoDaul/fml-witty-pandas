import logging
import fitz  # pymupdf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "research-paper-ingest/1.0 (academic use)"}
_TIMEOUT = 15
_MAX_WORKERS = 8


def _fetch_one(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("content-type", "").lower():
            logger.warning("Non-PDF content-type at %s", url)
            return None
        doc = fitz.open(stream=resp.content, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        result = text.strip() or None
        if result:
            logger.debug("Fetched PDF (%d chars): %s", len(result), url)
        else:
            logger.warning("Empty text extracted from PDF: %s", url)
        return result
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching PDF: %s", url)
    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP %s fetching PDF: %s", e.response.status_code, url)
    except Exception as e:
        logger.warning("Failed to fetch/parse PDF: %s — %s", url, e)
    return None


def fetch_pdf_texts(records: list[dict]) -> None:
    """
    Populate body_text in-place for records that have a pdf_url.
    Fetches PDFs concurrently with a tqdm progress bar.
    Records without a pdf_url are left as-is.
    """
    targets = [(i, r["pdf_url"]) for i, r in enumerate(records) if r.get("pdf_url")]
    if not targets:
        return

    fetched = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, url): i for i, url in targets}
        with tqdm(
            total=len(futures),
            unit="pdf",
            desc="  Fetching PDFs",
            leave=False,
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                i = futures[future]
                result = future.result()
                records[i]["body_text"] = result
                if result:
                    fetched += 1
                pbar.update(1)

    logger.info(
        "PDF batch: %d/%d fetched successfully (%d had no URL)",
        fetched,
        len(targets),
        len(records) - len(targets),
    )
