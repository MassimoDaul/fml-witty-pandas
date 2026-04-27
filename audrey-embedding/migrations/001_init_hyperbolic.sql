-- audrey-embedding migration 001: initialize hyperbolic columns.
--
-- Architecture: hyperbolic representation, Euclidean tangent for ANN, hyperbolic rerank.
--   audrey            halfvec(64)  tangent-space proxy (logmap0 of audrey_hyp); pgvector ANN runs against this
--   audrey_hyp        halfvec(64)  true Poincaré-ball coordinates; used for Python rerank
--   audrey_curvature  real         curvature c (default 1.0)
--
-- DESTRUCTIVE: drops the existing audrey halfvec column and recreates it at 64-dim.
-- The 64-dim choice is deliberate: hyperbolic space represents hierarchies efficiently
-- in low dimension, and matching the existing baseline width (~384) would obscure
-- the geometric signal. halfvec matches team convention for embedding columns.
--
-- Run once:
--   psql "$POSTGRES_CONN_STRING" -f audrey-embedding/migrations/001_init_hyperbolic.sql

BEGIN;

-- Guard: refuse to drop the existing audrey column if it has any data.
-- Wrapped in pg_class lookup so this is a no-op the first time it runs
-- (i.e. on a fresh DB where the column doesn't exist yet).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'papers' AND column_name = 'audrey'
    ) THEN
        IF (SELECT COUNT(*) FROM papers WHERE audrey IS NOT NULL) > 0 THEN
            RAISE EXCEPTION 'audrey column has data — aborting destructive migration';
        END IF;
    END IF;
END $$;

ALTER TABLE papers DROP COLUMN IF EXISTS audrey;
ALTER TABLE papers DROP COLUMN IF EXISTS audrey_hyp;
ALTER TABLE papers DROP COLUMN IF EXISTS audrey_curvature;

ALTER TABLE papers
    ADD COLUMN audrey           halfvec(64),
    ADD COLUMN audrey_hyp       halfvec(64),
    ADD COLUMN audrey_curvature real DEFAULT 1.0;

COMMIT;

-- Build the IVF index AFTER bulk-writing audrey embeddings, via:
--   python -c "from database.utils import get_connection, build_ivf_indexes; \
--              build_ivf_indexes(get_connection(), 'audrey')"
