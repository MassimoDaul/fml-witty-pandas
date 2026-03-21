from pathlib import Path
from dotenv import load_dotenv
import os

# .env is two levels up from this file (papers/ingest/config.py -> root)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
ARXIV_JSON = DATA_DIR / "arxiv-metadata-oai-snapshot.json"

POSTGRES_CONN_STRING = os.environ["POSTGRES_CONN_STRING"]

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# Texts per encode() call — sentence-transformers handles internal mini-batching
EMBED_BATCH = 64

# Rows per DB upsert
DB_BATCH = 200

DEFAULT_QTY = 10_000
DEFAULT_OFFSET = 0
