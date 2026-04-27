"""
audrey-embedding/manifold.py

Single source of truth for Poincaré-ball operations across train / embed / query.
Wraps geoopt so curvature lives in one place and the rest of the module never
touches geoopt directly.

Conventions
-----------
- The manifold is the Poincaré ball P^d = { x in R^d : ||x|| < 1/sqrt(c) }.
- Curvature c > 0; default c = 1.0.
- Tangent space at the origin is identified with R^d via expmap0 / logmap0.
- The "tangent proxy" we store in Postgres is logmap0(x_hyp): a Euclidean point
  pgvector can ANN-search. Cosine on the tangent is a recall-only heuristic; the
  truth metric is dist(x, y) below.
"""

from __future__ import annotations

import torch
from geoopt.manifolds import PoincareBall

DIM: int = 64
CURVATURE: float = 1.0


def ball(c: float = CURVATURE) -> PoincareBall:
    """Return a PoincareBall manifold object at curvature c."""
    return PoincareBall(c=c)


def project(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Clamp x onto the open ball (numerical safety)."""
    return ball(c).projx(x)


def expmap0(v: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Tangent-at-origin -> Poincaré ball."""
    return ball(c).expmap0(v)


def logmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Poincaré ball -> tangent at origin (the proxy stored in Postgres)."""
    return ball(c).logmap0(x)


def dist(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """True hyperbolic (Poincaré) distance. This is the rerank metric."""
    return ball(c).dist(x, y)
