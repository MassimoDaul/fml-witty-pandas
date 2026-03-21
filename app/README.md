# App

Minimal web interface for the paper search pipeline.

## Stack

| Component | Choice |
|---|---|
| Server | FastAPI + Uvicorn |
| Frontend | HTMX |
| Templates | Jinja2 |

## Routes

| Route | Description |
|---|---|
| `GET /` | Search form |
| `POST /search` | Runs `papers/query.py::search()`, returns results partial |

## Setup

From the repo root, activate the shared venv and install dependencies:

```bash
python -m venv venv

venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Ensure `.env` at the repo root contains `POSTGRES_CONN_STRING`.

## Usage

```bash
uvicorn app.main:app --reload
```

Then open `http://localhost:8000`.

**Note:** app startup is slow while the embedding model loads. Subsequent requests are fast.
