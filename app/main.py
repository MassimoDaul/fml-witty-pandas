import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "papers"))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from query import search, embed_query
from ingest.config import POSTGRES_CONN_STRING
import psycopg2
from pgvector.psycopg2 import register_vector

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
def do_search(request: Request, query: str = Form(...), k: int = Form(10)):
    k = max(1, min(k, 50))
    try:
        results = search(query, k=k, conn=request.app.state.conn)
    except Exception as e:
        return HTMLResponse(f'<p style="color:red">Error: {e}</p>', status_code=500)
    return templates.TemplateResponse(
        "_results.html", {"request": request, "results": results}
    )
