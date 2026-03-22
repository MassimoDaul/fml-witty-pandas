-- Run this once in the Supabase SQL editor (requires superuser for the extension)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    abstract    TEXT NOT NULL,
    categories  TEXT[]  NOT NULL,
    authors     TEXT[],
    published   DATE,
    updated     DATE,
    embedding   vector(768),
    inserted_at TIMESTAMPTZ DEFAULT now()
);

-- Build this index AFTER bulk insert, not before. See: python run.py build-index
-- CREATE INDEX papers_embedding_hnsw_idx
--     ON papers USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 128);

CREATE TABLE IF NOT EXISTS authors (
    name        TEXT PRIMARY KEY,
    papers      TEXT[] NOT NULL DEFAULT '{}',  -- arxiv_ids of their papers
    inserted_at TIMESTAMPTZ DEFAULT now()
);
