"""
database/init.py

Creates the papers table. Run once before ingest.

    python database/init.py
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_CONN_STRING"])


def init_db():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    corpus_id        TEXT PRIMARY KEY,
                    s2_paper_id      TEXT UNIQUE,
                    url              TEXT,
                    title            TEXT NOT NULL,
                    abstract         TEXT,
                    comments         TEXT,
                    conclusion       TEXT,
                    year             INT,
                    venue            TEXT,
                    citation_count   INT DEFAULT 0,
                    reference_count  INT DEFAULT 0,
                    fields_of_study  TEXT[],
                    subfields        TEXT[],
                    author_ids       TEXT[],        -- GIN index directly on papers
                    nomic            vector(384),
                    massimo_title    vector(384),
                    massimo_abstract vector(384),
                    massimo_metadata vector(384),
                    andrew           vector(128),
                    autoresearch     vector(128),
                    autoresearch_new vector(128),
                    audrey           vector(384),
                    inserted_at      TIMESTAMPTZ DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_papers_author_ids
                ON papers USING GIN (author_ids);
            """)
        conn.commit()
        print("Database initialized.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
