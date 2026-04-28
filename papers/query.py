"""
K-nearest neighbor search over S2 papers.
Supports four embedding backends: nomic (default), andrew, massimo, audrey.

Usage:
    python query.py "machine learning for protein folding"
    python query.py "transformer attention mechanisms" --k 20 --embedding andrew
    python query.py "quantum computing" --k 5 --embedding audrey
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from ingest.config import POSTGRES_CONN_STRING
from ingest.clean import clean_text
from ingest.embed import load_model

_REPO_ROOT = Path(__file__).parent.parent

DEFAULT_K = 10
VALID_EMBEDDINGS = ("nomic", "andrew", "massimo", "audrey")

# Nomic-embed-text-v1.5 outputs 768-dim; stored embeddings use Matryoshka truncation to 384.
_NOMIC_DIM = 384

_model: SentenceTransformer | None = None
_andrew_proj = None
_audrey_head = None


class PaperYear:
    """Wraps an integer year so templates can use .year, slicing, and str()."""
    def __init__(self, year: int):
        self.year = year
        self._s = str(year)

    def __str__(self):          return self._s
    def __bool__(self):         return True
    def __getitem__(self, idx): return self._s[idx]
    def toordinal(self):        return self.year * 365


# ── Nomic encoder ──────────────────────────────────────────────────────────────

def _nomic_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _nomic_vec(text: str, task: str = "search_query") -> np.ndarray:
    """Encode text with Nomic, truncate to 384-dim (Matryoshka), and re-normalise."""
    raw = _nomic_model().encode(f"{task}: {clean_text(text)}", normalize_embeddings=True)
    vec = raw[:_NOMIC_DIM].astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_query(text: str) -> list[float]:
    return _nomic_vec(text, task="search_query").tolist()


def embed_document(title: str, abstract: str) -> list[float]:
    raw = _nomic_model().encode(
        f"search_document: {clean_text(title)}. {clean_text(abstract)}",
        normalize_embeddings=True,
    )
    vec = raw[:_NOMIC_DIM].astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


# ── Poincaré ball math (torch-only, no geoopt dependency) ─────────────────────

def _poincare_expmap0(v: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """Map tangent vector at origin to Poincaré ball."""
    import torch
    sqrt_c = c ** 0.5
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    return (torch.tanh(sqrt_c * norm) / (sqrt_c * norm)) * v


def _poincare_logmap0(x: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """Map Poincaré point back to tangent space at origin."""
    import torch
    sqrt_c = c ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    return (torch.atanh(sqrt_c * norm) / (sqrt_c * norm)) * x


def _poincare_project(x: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """Clamp x onto the open ball (numerical safety)."""
    import torch
    max_norm = (1.0 - 1e-3) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > max_norm, x / norm * max_norm, x)


def _mobius_add(x: "torch.Tensor", y: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = (1 + 2 * c * xy + c ** 2 * x2 * y2).clamp(min=1e-15)
    return num / denom


def _poincare_dist(x: "torch.Tensor", y: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """True hyperbolic (Poincaré ball) distance — used for Audrey rerank."""
    import torch
    sqrt_c = c ** 0.5
    add = _mobius_add(-x, y, c)
    return (2.0 / sqrt_c) * torch.atanh((sqrt_c * add.norm(dim=-1)).clamp(max=1 - 1e-7))


def _vec_to_np(v) -> np.ndarray:
    """Convert a pgvector halfvec/vector cell to float32 numpy array."""
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32)


# ── Andrew encoder ─────────────────────────────────────────────────────────────

def _get_andrew_proj():
    """Load and cache paper_proj Linear(384→128) from the Andrew checkpoint."""
    global _andrew_proj
    if _andrew_proj is not None:
        return _andrew_proj
    import torch
    ckpt_path = _REPO_ROOT / "andrew-embedding" / "weights" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Andrew checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    proj = torch.nn.Linear(384, 128, bias=True)
    proj.weight = torch.nn.Parameter(ckpt["model_state_dict"]["paper_proj.weight"])
    proj.bias   = torch.nn.Parameter(ckpt["model_state_dict"]["paper_proj.bias"])
    proj.eval()
    _andrew_proj = proj
    return _andrew_proj


def _embed_andrew(text: str) -> np.ndarray:
    """Encode query text into Andrew's 128-dim space via paper_proj."""
    import torch
    import torch.nn.functional as F
    nomic_t = torch.tensor(_nomic_vec(text)[:384], dtype=torch.float32).unsqueeze(0)
    proj = _get_andrew_proj()
    with torch.no_grad():
        out = F.normalize(proj(nomic_t).squeeze(0), dim=0)
    return out.numpy().astype(np.float32)


# ── Audrey encoder ─────────────────────────────────────────────────────────────

class _AudreyHead:
    """Minimal inference wrapper matching audrey-embedding/train.py::ProjectionHead."""

    def __init__(self, state_dict, in_dim: int, out_dim: int, c: float):
        import torch.nn as nn
        self.c = c
        hidden = state_dict["net.0.weight"].shape[0]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        net_state = {k[4:]: v for k, v in state_dict.items() if k.startswith("net.")}
        self.net.load_state_dict(net_state)
        self.net.eval()

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        import torch
        with torch.no_grad():
            v = self.net(x)
        return _poincare_project(_poincare_expmap0(v, self.c), self.c)


def _get_audrey_head() -> _AudreyHead:
    global _audrey_head
    if _audrey_head is not None:
        return _audrey_head
    import torch
    ckpt_path = _REPO_ROOT / "audrey-embedding" / "weights" / "projection_v1.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Audrey checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    _audrey_head = _AudreyHead(ckpt["state_dict"], ckpt["in_dim"], ckpt["out_dim"], ckpt["curvature"])
    return _audrey_head


def _encode_audrey(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_hyp, q_tan) as float32 arrays for Audrey two-stage search."""
    import torch
    feat = torch.from_numpy(_nomic_vec(text).astype(np.float32)).unsqueeze(0)
    head = _get_audrey_head()
    q_hyp = head.encode(feat)
    q_tan = _poincare_logmap0(q_hyp, head.c)
    return (
        q_hyp.squeeze(0).numpy().astype(np.float32),
        q_tan.squeeze(0).numpy().astype(np.float32),
    )


# ── Row helper ─────────────────────────────────────────────────────────────────

def _row_to_paper(row) -> dict:
    corpus_id, s2_paper_id, url, title, abstract, fields, year, similarity = row
    return {
        "arxiv_id":    corpus_id,
        "s2_paper_id": s2_paper_id,
        "url":         url or "",
        "title":       title or "",
        "abstract":    abstract or "",
        "authors":     [],
        "categories":  fields or [],
        "published":   PaperYear(year) if year else None,
        "similarity":  float(similarity),
    }


def _extra_where(categories, year_from, year_to) -> tuple[str, list]:
    """Build optional AND-clauses and params for category/year filters."""
    clauses, params = [], []
    if categories:
        clauses.append("fields_of_study && %s::text[]")
        params.append(categories)
    if year_from is not None:
        clauses.append("year >= %s")
        params.append(year_from)
    if year_to is not None:
        clauses.append("year <= %s")
        params.append(year_to)
    sql = (" AND " + " AND ".join(clauses)) if clauses else ""
    return sql, params


# ── Per-embedding search implementations ───────────────────────────────────────

def _search_nomic(query, k, conn, categories, year_from, year_to) -> list[dict]:
    vec = np.array(embed_query(query), dtype=np.float32)
    extra, extra_params = _extra_where(categories, year_from, year_to)
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        cur.execute(
            f"""
            SELECT corpus_id, s2_paper_id, url, title, abstract, fields_of_study, year,
                   1 - (nomic <=> %s::vector) AS similarity
            FROM papers
            WHERE nomic IS NOT NULL{extra}
            ORDER BY nomic <=> %s::vector
            LIMIT %s
            """,
            [vec] + extra_params + [vec, k],
        )
        return [_row_to_paper(r) for r in cur.fetchall()]


def _search_andrew(query, k, conn, categories, year_from, year_to) -> list[dict]:
    """GNN-enhanced search via paper_proj projection (128-dim)."""
    vec = _embed_andrew(query)
    extra, extra_params = _extra_where(categories, year_from, year_to)
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        cur.execute(
            f"""
            SELECT corpus_id, s2_paper_id, url, title, abstract, fields_of_study, year,
                   1 - (andrew <=> %s::vector) AS similarity
            FROM papers
            WHERE andrew IS NOT NULL{extra}
            ORDER BY andrew <=> %s::vector
            LIMIT %s
            """,
            [vec] + extra_params + [vec, k],
        )
        return [_row_to_paper(r) for r in cur.fetchall()]


def _search_massimo(query, k, conn, categories, year_from, year_to) -> list[dict]:
    """Weighted multi-segment search: 35% title + 50% abstract + 15% metadata."""
    vec = np.array(embed_query(query), dtype=np.float32)
    extra, extra_params = _extra_where(categories, year_from, year_to)
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        cur.execute(
            f"""
            SELECT corpus_id, s2_paper_id, url, title, abstract, fields_of_study, year,
                   (0.35 * (1 - (massimo_title    <=> %s::vector)) +
                    0.50 * (1 - (massimo_abstract  <=> %s::vector)) +
                    COALESCE(0.15 * (1 - (massimo_metadata <=> %s::vector)), 0.0)) AS similarity
            FROM papers
            WHERE massimo_title IS NOT NULL AND massimo_abstract IS NOT NULL{extra}
            ORDER BY (0.35 * (massimo_title   <=> %s::vector) +
                      0.50 * (massimo_abstract <=> %s::vector) +
                      COALESCE(0.15 * (massimo_metadata <=> %s::vector), 0.0)) ASC
            LIMIT %s
            """,
            [vec, vec, vec] + extra_params + [vec, vec, vec, k],
        )
        return [_row_to_paper(r) for r in cur.fetchall()]


def _search_audrey(query, k, conn, categories, year_from, year_to) -> list[dict]:
    """Two-stage hyperbolic search: tangent ANN → Poincaré distance rerank."""
    import torch
    q_hyp, q_tan = _encode_audrey(query)
    extra, extra_params = _extra_where(categories, year_from, year_to)
    candidates = max(k * 5, 100)
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        cur.execute(
            f"""
            SELECT corpus_id, s2_paper_id, url, title, abstract, fields_of_study, year, audrey_hyp
            FROM papers
            WHERE audrey IS NOT NULL AND audrey_hyp IS NOT NULL{extra}
            ORDER BY audrey <=> %s
            LIMIT %s
            """,
            extra_params + [np.array(q_tan, dtype=np.float32), candidates],
        )
        rows = cur.fetchall()

    if not rows:
        return []

    c = _get_audrey_head().c
    q_hyp_t = torch.from_numpy(q_hyp).unsqueeze(0)
    cand_hyp = torch.from_numpy(np.stack([_vec_to_np(r[7]) for r in rows]))
    d_hyp = _poincare_dist(q_hyp_t.expand_as(cand_hyp), cand_hyp, c=c).numpy()

    scored = sorted(zip(d_hyp.tolist(), rows), key=lambda x: x[0])
    results = []
    for dist, row in scored[:k]:
        corpus_id, s2_paper_id, url, title, abstract, fields, year, _ = row
        results.append({
            "arxiv_id":    corpus_id,
            "s2_paper_id": s2_paper_id,
            "url":         url or "",
            "title":       title or "",
            "abstract":    abstract or "",
            "authors":     [],
            "categories":  fields or [],
            "published":   PaperYear(year) if year else None,
            "similarity":  float(1.0 / (1.0 + dist)),
        })
    return results


# ── Public search API ──────────────────────────────────────────────────────────

def search(
    query: str,
    k: int = DEFAULT_K,
    conn=None,
    categories: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    embedding: str = "nomic",
) -> list[dict]:
    close_after = conn is None
    if conn is None:
        conn = psycopg2.connect(POSTGRES_CONN_STRING)
        register_vector(conn)

    try:
        if embedding == "andrew":
            return _search_andrew(query, k, conn, categories, year_from, year_to)
        elif embedding == "massimo":
            return _search_massimo(query, k, conn, categories, year_from, year_to)
        elif embedding == "audrey":
            return _search_audrey(query, k, conn, categories, year_from, year_to)
        else:
            return _search_nomic(query, k, conn, categories, year_from, year_to)
    finally:
        if close_after:
            conn.close()


def related_search(
    title: str,
    abstract: str,
    k: int = 8,
    exclude_id: str = "",
    conn=None,
) -> list[dict]:
    close_after = conn is None
    if conn is None:
        conn = psycopg2.connect(POSTGRES_CONN_STRING)
        register_vector(conn)

    vec = np.array(embed_document(title, abstract), dtype=np.float32)

    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        cur.execute(
            """
            SELECT corpus_id, s2_paper_id, url, title, abstract, fields_of_study, year,
                   1 - (nomic <=> %s::vector) AS similarity
            FROM papers
            WHERE nomic IS NOT NULL AND corpus_id != %s
            ORDER BY nomic <=> %s::vector
            LIMIT %s
            """,
            (vec, exclude_id, vec, k),
        )
        rows = cur.fetchall()

    if close_after:
        conn.close()

    return [_row_to_paper(r) for r in rows]


# ── CLI ────────────────────────────────────────────────────────────────────────

def print_results(results: list[dict], show_abstract: bool = True) -> None:
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        fields = ", ".join(r["categories"]) if r["categories"] else "-"
        print(f"\n{'─' * 72}")
        print(f"#{i}  [{r['similarity']:.4f}]  {r['arxiv_id']}")
        print(f"    {r['title']}")
        print(f"    {r['published'] or '-'}  |  {fields}")
        if show_abstract:
            abstract = r["abstract"]
            if len(abstract) > 300:
                abstract = abstract[:300].rstrip() + "..."
            print(f"\n    {abstract}")

    print(f"\n{'─' * 72}")
    print(f"{len(results)} results")


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN search over S2 papers")
    parser.add_argument("query", help="Research topic or question")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--embedding", choices=VALID_EMBEDDINGS, default="nomic")
    parser.add_argument("--no-abstract", action="store_true")
    args = parser.parse_args()

    results = search(args.query, k=args.k, embedding=args.embedding)
    print_results(results, show_abstract=not args.no_abstract)


if __name__ == "__main__":
    main()
