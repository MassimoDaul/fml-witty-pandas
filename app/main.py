import re
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).parent.parent / "papers"))

import httpx
import psycopg2
import psycopg2.extras
from cachetools import TTLCache
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool

from author_query import papers_by_author, search_by_name
from ingest.config import POSTGRES_CONN_STRING
from query import embed_query, related_search, search

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["url_encode"] = lambda s: quote(str(s), safe="")


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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
def do_search(
    request: Request,
    query: str = Form(...),
    k: int = Form(10),
    type: str = Form("papers"),
    categories: str = Form(default=""),
    year_from: str = Form(default=""),
    year_to: str = Form(default=""),
):
    k = max(1, min(k, 50))
    try:
        with db_conn(request.app) as conn:
            if type == "authors":
                results = search_by_name(query, k=k, conn=conn)
                return templates.TemplateResponse(
                    "_author_results.html", {"request": request, "results": results}
                )
            cats = [c for c in re.split(r"[,\s]+", categories.strip()) if c] or None
            yf = int(year_from.strip()) if year_from.strip() else None
            yt = int(year_to.strip()) if year_to.strip() else None
            results = search(query, k=k, conn=conn, categories=cats, year_from=yf, year_to=yt)
    except Exception as e:
        return HTMLResponse(f'<p style="color:red">Error: {e}</p>', status_code=500)
    return templates.TemplateResponse(
        "_results.html", {"request": request, "results": results}
    )


@app.get("/author", response_class=HTMLResponse)
def author_detail(request: Request, name: str = Query(...)):
    with db_conn(request.app) as conn:
        papers = papers_by_author(name, conn=conn)
    if not papers:
        return HTMLResponse("<p>Author not found.</p>", status_code=404)
    return templates.TemplateResponse(
        "author.html", {"request": request, "name": name, "papers": papers}
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
        except Exception as e:
            return HTMLResponse(f'<p style="color:red">Error: {e}</p>')
    return templates.TemplateResponse("_results.html", {"request": request, "results": results})


@app.get("/paper/{arxiv_id}", response_class=HTMLResponse)
def paper_detail(request: Request, arxiv_id: str):
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
    return templates.TemplateResponse("paper.html", {"request": request, "paper": paper})


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
