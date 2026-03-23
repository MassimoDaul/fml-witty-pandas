import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "papers"))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from query import search, embed_query, related_search
from author_query import search_by_name
from ingest.config import POSTGRES_CONN_STRING
import psycopg2
from pgvector.psycopg2 import register_vector
import httpx
import time
import threading
from cachetools import TTLCache

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = psycopg2.connect(POSTGRES_CONN_STRING)
    register_vector(conn)
    app.state.conn = conn
    app.state.s2_cache = TTLCache(maxsize=64, ttl=3600) # small cache with 1 hour ttl
    app.state.s2_cache_lock = threading.Lock()
    embed_query("warmup")
    yield
    conn.close()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
def do_search(request: Request, query: str = Form(...), k: int = Form(10), type: str = Form("papers")):
    k = max(1, min(k, 50))
    try:
        if type == "authors":
            results = search_by_name(query, k=k, conn=request.app.state.conn)
            return templates.TemplateResponse(
                "_author_results.html", {"request": request, "results": results}
            )
        results = search(query, k=k, conn=request.app.state.conn)
    except Exception as e:
        return HTMLResponse(f'<p style="color:red">Error: {e}</p>', status_code=500)
    return templates.TemplateResponse(
        "_results.html", {"request": request, "results": results}
    )


@app.get("/paper/{arxiv_id}/related", response_class=HTMLResponse)
def paper_related(request: Request, arxiv_id: str):
    conn = request.app.state.conn
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
    conn = request.app.state.conn
    with conn.cursor() as cur:
        cur.execute(
            "SELECT arxiv_id, title, abstract, categories, authors, published "
            "FROM papers WHERE arxiv_id = %s",
            (arxiv_id,),
        )
        row = cur.fetchone()
    if not row:
        return HTMLResponse("<p>Paper not found in local database.</p>", status_code=404)
    paper = {
        "arxiv_id": row[0],
        "title": row[1],
        "abstract": row[2],
        "categories": row[3] or [],
        "authors": row[4] or [],
        "published": str(row[5]) if row[5] else None,
    }
    return templates.TemplateResponse("paper.html", {"request": request, "paper": paper})


@app.get("/api/paper/{arxiv_id}/references")
def paper_references(request: Request, arxiv_id: str):
    with request.app.state.s2_cache_lock:
        cached = request.app.state.s2_cache.get(arxiv_id)
    if cached:
        return JSONResponse(cached)

    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}"
    fields = (
        "externalIds,"
        "references.title,references.externalIds,references.year,references.authors,references.citationCount,"
        "citations.title,citations.externalIds,citations.year,citations.authors,citations.citationCount"
    )
    try:
        resp = None
        for attempt in range(4):
            resp = httpx.get(url, params={"fields": fields}, timeout=15.0)
            if resp.status_code != 429:
                break
            wait = int(resp.headers.get("Retry-After", 0)) or (2 ** attempt)
            time.sleep(wait)
    except httpx.RequestError:
        return JSONResponse({"nodes": [], "links": [], "error": "Could not reach Semantic Scholar"})

    if resp.status_code == 404:
        return JSONResponse({"nodes": [], "links": [], "error": "Paper not found in Semantic Scholar"})
    if resp.status_code == 429:
        return JSONResponse({"nodes": [], "links": [], "error": "Semantic Scholar rate limit hit — wait a moment and try again"})
    if resp.status_code != 200:
        return JSONResponse({"nodes": [], "links": [], "error": f"Semantic Scholar returned {resp.status_code}"})

    body = resp.json()

    def parse_papers(items, is_citation):
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

    refs = parse_papers(body.get("references") or [], False)
    cites = parse_papers(body.get("citations") or [], True)
    all_nodes = refs + cites

    conn = request.app.state.conn
    all_arxiv_ids = [n["arxiv_id"] for n in all_nodes if n["arxiv_id"]]
    in_db = set()
    if all_arxiv_ids:
        with conn.cursor() as cur:
            cur.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)", (all_arxiv_ids,))
            in_db = {row[0] for row in cur.fetchall()}

    nodes = [{"id": arxiv_id, "arxiv_id": arxiv_id, "is_center": True, "is_citation": False, "in_db": True}]
    links = []
    for n in all_nodes:
        nodes.append({**n, "in_db": n["arxiv_id"] in in_db if n["arxiv_id"] else False, "is_center": False})
        if n["is_citation"]:
            links.append({"source": n["id"], "target": arxiv_id, "is_citation": True})
        else:
            links.append({"source": arxiv_id, "target": n["id"], "is_citation": False})

    response_data = {
        "nodes": nodes,
        "links": links,
        "paper_ids": body.get("externalIds") or {},
        "s2_paper_id": body.get("paperId"),
    }
    with request.app.state.s2_cache_lock:
        request.app.state.s2_cache[arxiv_id] = response_data
    return JSONResponse(response_data)
