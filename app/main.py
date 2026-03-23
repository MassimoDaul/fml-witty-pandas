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

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = psycopg2.connect(POSTGRES_CONN_STRING)
    register_vector(conn)
    app.state.conn = conn
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
    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}/references"
    try:
        resp = httpx.get(url, params={"fields": "title,externalIds,year,authors,citationCount"}, timeout=15.0)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            time.sleep(wait)
            resp = httpx.get(url, params={"fields": "title,externalIds,year,authors,citationCount"}, timeout=15.0)
    except httpx.RequestError:
        return JSONResponse({"nodes": [], "links": [], "error": "Could not reach Semantic Scholar"})

    if resp.status_code == 404:
        return JSONResponse({"nodes": [], "links": [], "error": "Paper not found in Semantic Scholar"})
    if resp.status_code == 429:
        return JSONResponse({"nodes": [], "links": [], "error": "Semantic Scholar rate limit hit — wait a moment and try again"})
    if resp.status_code != 200:
        return JSONResponse({"nodes": [], "links": [], "error": f"Semantic Scholar returned {resp.status_code}"})

    data = resp.json().get("data", [])

    seen = set()
    refs = []
    for item in data:
        cited = item.get("citedPaper") or {}
        ext_ids = cited.get("externalIds") or {}
        arxiv_ref = ext_ids.get("ArXiv")
        node_id = arxiv_ref or cited.get("paperId") or ""
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        refs.append({
            "id": node_id,
            "arxiv_id": arxiv_ref,
            "title": cited.get("title") or "Unknown",
            "year": cited.get("year"),
            "authors": [a.get("name", "") for a in (cited.get("authors") or [])[:3]],
            "citation_count": cited.get("citationCount"),
        })

    conn = request.app.state.conn
    ref_arxiv_ids = [r["arxiv_id"] for r in refs if r["arxiv_id"]]
    in_db = set()
    if ref_arxiv_ids:
        with conn.cursor() as cur:
            cur.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)", (ref_arxiv_ids,))
            in_db = {row[0] for row in cur.fetchall()}

    nodes = [{"id": arxiv_id, "arxiv_id": arxiv_id, "is_center": True, "in_db": True}]
    links = []
    for r in refs:
        nodes.append({**r, "in_db": r["arxiv_id"] in in_db if r["arxiv_id"] else False, "is_center": False})
        links.append({"source": arxiv_id, "target": r["id"]})

    return JSONResponse({"nodes": nodes, "links": links})
