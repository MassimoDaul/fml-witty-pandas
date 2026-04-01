import re
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from html import escape
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

sys.path.insert(0, str(Path(__file__).parent.parent / "papers"))

import httpx
import psycopg2
import psycopg2.extras
from cachetools import TTLCache
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool

from author_query import papers_by_author, search_by_name
from ingest.config import POSTGRES_CONN_STRING
from query import embed_query, related_search, search

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["url_encode"] = lambda s: quote(str(s), safe="")

DEFAULT_RESULTS_K = 12
MAX_RESULTS_K = 50
DEFAULT_MODE = "papers"
DEFAULT_SORT = "relevance"
PAPER_SORTS = {"relevance", "newest", "oldest"}


@contextmanager
def db_conn(app):
    conn = app.state.pool.getconn()
    register_vector(conn)
    try:
        yield conn
    finally:
        app.state.pool.putconn(conn)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = ThreadedConnectionPool(1, 10, POSTGRES_CONN_STRING)
    app.state.pool = pool
    app.state.s2_cache = TTLCache(maxsize=64, ttl=3600)
    app.state.s2_cache_lock = threading.Lock()
    embed_query("warmup")
    yield
    pool.closeall()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def normalize_mode(value: str) -> str:
    return "authors" if value == "authors" else DEFAULT_MODE


def normalize_sort(value: str) -> str:
    return value if value in PAPER_SORTS else DEFAULT_SORT


def clamp_k(value: str | int | None) -> int:
    try:
        parsed = int(value) if value is not None else DEFAULT_RESULTS_K
    except (TypeError, ValueError):
        parsed = DEFAULT_RESULTS_K
    return max(1, min(parsed, MAX_RESULTS_K))


def clean_year(value: str | None) -> tuple[str, int | None]:
    raw = (value or "").strip()
    if not raw:
        return "", None
    try:
        parsed = int(raw)
    except ValueError:
        return raw, None
    if parsed < 1900 or parsed > 2099:
        return raw, None
    return str(parsed), parsed


def sort_paper_results(results: list[dict], sort: str) -> list[dict]:
    if sort == "relevance":
        return results

    def published_ordinal(item: dict) -> int:
        published = item.get("published")
        if hasattr(published, "toordinal"):
            return published.toordinal()
        return -1

    reverse = sort == "newest"
    return sorted(results, key=published_ordinal, reverse=reverse)


def build_current_url(request: Request) -> str:
    query = request.url.query
    return f"{request.url.path}?{query}" if query else request.url.path


def safe_internal_path(path: str, fallback: str) -> str:
    if path and path.startswith("/") and not path.startswith("//"):
        return path
    return fallback


def build_results_url(
    *,
    q: str = "",
    mode: str = DEFAULT_MODE,
    sort: str = DEFAULT_SORT,
    k: int = DEFAULT_RESULTS_K,
    categories: str = "",
    year_from: str = "",
    year_to: str = "",
) -> str:
    params = {
        "q": q,
        "mode": normalize_mode(mode),
        "sort": normalize_sort(sort),
        "k": clamp_k(k),
    }
    if categories.strip():
        params["categories"] = categories.strip()
    if year_from.strip():
        params["year_from"] = year_from.strip()
    if year_to.strip():
        params["year_to"] = year_to.strip()
    query_string = urlencode(params)
    return f"/results?{query_string}" if query_string else "/results"


def build_results_context(
    request: Request,
    *,
    q: str = "",
    mode: str = DEFAULT_MODE,
    sort: str = DEFAULT_SORT,
    k: str | int | None = DEFAULT_RESULTS_K,
    categories: str = "",
    year_from: str = "",
    year_to: str = "",
) -> dict:
    query_text = q.strip()
    mode = normalize_mode(mode)
    sort = normalize_sort(sort)
    k_value = clamp_k(k)
    categories_text = categories.strip()
    year_from_text, year_from_value = clean_year(year_from)
    year_to_text, year_to_value = clean_year(year_to)

    if year_from_value is not None and year_to_value is not None and year_from_value > year_to_value:
        year_from_value, year_to_value = year_to_value, year_from_value
        year_from_text, year_to_text = str(year_from_value), str(year_to_value)

    results: list[dict] = []
    error_message = ""

    if query_text:
        try:
            with db_conn(request.app) as conn:
                if mode == "authors":
                    results = search_by_name(query_text, k=k_value, conn=conn)
                else:
                    category_filters = [c for c in re.split(r"[,\s]+", categories_text) if c] or None
                    results = search(
                        query_text,
                        k=k_value,
                        conn=conn,
                        categories=category_filters,
                        year_from=year_from_value,
                        year_to=year_to_value,
                    )
                    results = sort_paper_results(results, sort)
        except Exception as exc:
            error_message = str(exc)

    current_url = build_current_url(request)

    return {
        "query": query_text,
        "mode": mode,
        "sort": sort,
        "k": k_value,
        "categories": categories_text,
        "year_from": year_from_text,
        "year_to": year_to_text,
        "results": results,
        "result_count": len(results),
        "has_query": bool(query_text),
        "error_message": error_message,
        "current_url": current_url,
        "clear_filters_url": build_results_url(
            q=query_text,
            mode=mode,
            sort=sort,
            k=k_value,
        ),
        "switch_mode_url": build_results_url(
            q=query_text,
            mode="authors" if mode == "papers" else "papers",
            sort=DEFAULT_SORT,
            k=k_value,
        ),
        "page_body_class": "page-results" if request.url.path == "/results" else "page-search",
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "mode": DEFAULT_MODE,
            "query": "",
        },
    )


@app.get("/results", response_class=HTMLResponse)
def results_page(
    request: Request,
    q: str = Query(default=""),
    mode: str = Query(default=DEFAULT_MODE),
    sort: str = Query(default=DEFAULT_SORT),
    k: str = Query(default=str(DEFAULT_RESULTS_K)),
    categories: str = Query(default=""),
    year_from: str = Query(default=""),
    year_to: str = Query(default=""),
):
    context = build_results_context(
        request,
        q=q,
        mode=mode,
        sort=sort,
        k=k,
        categories=categories,
        year_from=year_from,
        year_to=year_to,
    )
    return templates.TemplateResponse(request, "results.html", context)


@app.post("/search")
def legacy_search(
    query: str = Form(...),
    k: str = Form(default=str(DEFAULT_RESULTS_K)),
    type: str = Form(default=DEFAULT_MODE),
    categories: str = Form(default=""),
    year_from: str = Form(default=""),
    year_to: str = Form(default=""),
):
    redirect_url = build_results_url(
        q=query.strip(),
        mode=type,
        k=k,
        categories=categories,
        year_from=year_from,
        year_to=year_to,
    )
    return RedirectResponse(url=redirect_url, status_code=303)


@app.get("/author", response_class=HTMLResponse)
def author_detail(
    request: Request,
    name: str = Query(...),
    return_to: str = Query(default=""),
):
    with db_conn(request.app) as conn:
        papers = papers_by_author(name, conn=conn)
    if not papers:
        return HTMLResponse("<p>Author not found.</p>", status_code=404)

    fallback = build_results_url(q=name, mode="authors")
    return templates.TemplateResponse(
        request,
        "author.html",
        {
            "name": name,
            "papers": papers,
            "back_href": safe_internal_path(return_to, fallback),
            "current_url": build_current_url(request),
        },
    )


@app.get("/paper/{arxiv_id}/related", response_class=HTMLResponse)
def paper_related(request: Request, arxiv_id: str):
    with db_conn(request.app) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title, abstract FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            row = cur.fetchone()
        if not row:
            return HTMLResponse("<p>Paper not found.</p>")
        title, abstract = row
        try:
            results = related_search(title, abstract, k=5, exclude_id=arxiv_id, conn=conn)
        except Exception as exc:
            return HTMLResponse(
                (
                    '<section class="state-panel state-panel--error">'
                    '<p class="eyebrow">Related papers unavailable</p>'
                    "<h3>Semantic Scholar is not available right now.</h3>"
                    f"<p>{escape(str(exc))}</p>"
                    "</section>"
                )
            )
    return templates.TemplateResponse(
        request,
        "_related_results.html",
        {
            "results": results,
            "return_to": f"/paper/{arxiv_id}",
        },
    )


@app.get("/paper/{arxiv_id}", response_class=HTMLResponse)
def paper_detail(
    request: Request,
    arxiv_id: str,
    return_to: str = Query(default=""),
):
    with db_conn(request.app) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT arxiv_id, title, abstract, categories, authors, published "
                "FROM papers WHERE arxiv_id = %s",
                (arxiv_id,),
            )
            row = cur.fetchone()
    if not row:
        return HTMLResponse("<p>Paper not found in local database.</p>", status_code=404)

    paper = {
        "arxiv_id": row["arxiv_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "categories": row["categories"] or [],
        "authors": row["authors"] or [],
        "published": str(row["published"]) if row["published"] else None,
    }
    fallback = "/results"
    return templates.TemplateResponse(
        request,
        "paper.html",
        {
            "paper": paper,
            "back_href": safe_internal_path(return_to, fallback),
        },
    )


S2_FIELDS = (
    "externalIds,"
    "references.title,references.externalIds,references.year,references.authors,references.citationCount,"
    "citations.title,citations.externalIds,citations.year,citations.authors,citations.citationCount"
)


def _fetch_s2_paper(arxiv_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Fetch a paper from Semantic Scholar with exponential-backoff retry on 429.
    Returns (body_dict, error_string) — exactly one will be non-None."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}"
    try:
        resp = None
        for attempt in range(4):
            resp = httpx.get(url, params={"fields": S2_FIELDS}, timeout=15.0)
            if resp.status_code != 429:
                break
            wait = int(resp.headers.get("Retry-After", 0)) or (2 ** attempt)
            time.sleep(wait)
    except httpx.RequestError:
        return None, "Could not reach Semantic Scholar"

    if resp.status_code == 404:
        return None, "Paper not found in Semantic Scholar"
    if resp.status_code == 429:
        return None, "Semantic Scholar rate limit hit — wait a moment and try again"
    if resp.status_code != 200:
        return None, f"Semantic Scholar returned {resp.status_code}"
    return resp.json(), None


def _parse_s2_papers(items: list, is_citation: bool) -> list[dict]:
    """Convert a list of raw S2 paper dicts into normalised node dicts."""
    seen = set()
    result = []
    for paper in items:
        ext_ids = paper.get("externalIds") or {}
        arxiv_ref = ext_ids.get("ArXiv")
        node_id = arxiv_ref or paper.get("paperId") or ""
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        result.append({
            "id": node_id,
            "arxiv_id": arxiv_ref,
            "s2_id": paper.get("paperId"),
            "ext_ids": {k: v for k, v in ext_ids.items() if k != "ArXiv"},
            "title": paper.get("title") or "Unknown",
            "year": paper.get("year"),
            "authors": [a.get("name", "") for a in (paper.get("authors") or [])[:3]],
            "citation_count": paper.get("citationCount"),
            "is_citation": is_citation,
        })
    return result


def _build_graph(center_id: str, refs: list, cites: list, in_db: set) -> dict:
    """Assemble nodes and links for the reference graph."""
    nodes = [{"id": center_id, "arxiv_id": center_id, "is_center": True, "is_citation": False, "in_db": True}]
    links = []
    for n in refs + cites:
        nodes.append({**n, "in_db": n["arxiv_id"] in in_db if n["arxiv_id"] else False, "is_center": False})
        if n["is_citation"]:
            links.append({"source": n["id"], "target": center_id, "is_citation": True})
        else:
            links.append({"source": center_id, "target": n["id"], "is_citation": False})
    return {"nodes": nodes, "links": links}


@app.get("/api/paper/{arxiv_id}/references")
def paper_references(request: Request, arxiv_id: str):
    with request.app.state.s2_cache_lock:
        cached = request.app.state.s2_cache.get(arxiv_id)
    if cached:
        return JSONResponse(cached)

    body, error = _fetch_s2_paper(arxiv_id)
    if error:
        return JSONResponse({"nodes": [], "links": [], "error": error})

    refs = _parse_s2_papers(body.get("references") or [], False)
    cites = _parse_s2_papers(body.get("citations") or [], True)

    all_arxiv_ids = [n["arxiv_id"] for n in refs + cites if n["arxiv_id"]]
    in_db = set()
    if all_arxiv_ids:
        with db_conn(request.app) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)", (all_arxiv_ids,))
                in_db = {row[0] for row in cur.fetchall()}

    response_data = {
        **_build_graph(arxiv_id, refs, cites, in_db),
        "paper_ids": body.get("externalIds") or {},
        "s2_paper_id": body.get("paperId"),
    }
    with request.app.state.s2_cache_lock:
        request.app.state.s2_cache[arxiv_id] = response_data
    return JSONResponse(response_data)
