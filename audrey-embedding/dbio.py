"""
audrey-embedding/dbio.py

Local DB helpers for the audrey-embedding module.

These exist because:
- pgvector-python returns numpy.ndarray for `vector` columns and HalfVector for
  `halfvec` columns; we need a uniform decoder.
- database/utils.py::build_ivf_indexes hardcodes vector_cosine_ops, but audrey
  is a halfvec column and needs halfvec_cosine_ops.

Kept inside audrey-embedding/ to avoid touching shared utility code.
"""

from __future__ import annotations

import numpy as np
from psycopg2.extensions import connection as PGConnection


def vec_to_np(v) -> np.ndarray:
    """Decode a pgvector cell (vector or halfvec) to float32 numpy array."""
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32)


def build_audrey_ivf_index(conn: PGConnection, nlist: int = 25) -> None:
    """
    Create the IVF cosine index on papers.audrey using halfvec_cosine_ops.

    audrey is halfvec(64); the team's database/utils.build_ivf_indexes assumes
    vector type and uses vector_cosine_ops, which fails on halfvec columns.
    """
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '32MB'")
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS papers_audrey_ivf
            ON papers
            USING ivfflat (audrey halfvec_cosine_ops)
            WITH (lists = {int(nlist)})
        """)
    conn.commit()
