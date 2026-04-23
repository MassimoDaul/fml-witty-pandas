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
                    nomic            vector(768),   -- nomic(title + abstract)
                    massimo_title    vector(768),
                    massimo_abstract vector(768),
                    massimo_metadata vector(768),
                    andrew           vector(128),
                    audrey           vector(768),
                    inserted_at      TIMESTAMPTZ DEFAULT now()
                );
            """)
        conn.commit()
        print("Database initialized.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
