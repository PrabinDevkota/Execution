"""Global FLOP-budget rank allocation via spectral protect scores (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from spectralite.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LayerAllocMeta:
    """Per-matrix geometry + spectral scores for allocation."""

    name: str
    in_features: int
    out_features: int
    q: int
    rho_eff: float
    compressibility: float
    stable_rank: float
    protect: float

    @property
    def dense_macs(self) -> int:
        return int(self.in_features * self.out_features)

    def lowrank_macs(self, rank: int) -> int:
        r = int(max(1, min(rank, self.q)))
        return int(r * (self.in_features + self.out_features))


def keep_ratio(metas: Sequence[LayerAllocMeta], ranks: dict[str, int]) -> float:
    """Analytic touched-layer keep ratio ``Σ r(m+n) / Σ mn``."""
    num = 0
    den = 0
    for meta in metas:
        den += meta.dense_macs
        num += meta.lowrank_macs(ranks[meta.name])
    return float(num) / float(max(den, 1))


def ranks_from_lambda(
    metas: Sequence[LayerAllocMeta],
    lam: float,
    *,
    min_rank: int = 1,
) -> dict[str, int]:
    """Map global scale ``λ`` → per-layer ranks via protect scores.

    ``r_ℓ = clip(round(λ · protect_ℓ · q_ℓ), min_rank, q_ℓ)``

    High ``protect`` (flat spectrum × stable-rank importance) retains more rank.
    """
    ranks: dict[str, int] = {}
    for meta in metas:
        raw = float(lam) * float(meta.protect) * float(meta.q)
        r = int(round(raw))
        r = max(int(min_rank), min(int(meta.q), r))
        ranks[meta.name] = r
    return ranks


def allocate_ranks_for_budget(
    metas: Sequence[LayerAllocMeta],
    target_keep_ratio: float,
    *,
    tol: float = 5e-4,
    max_iters: int = 40,
    min_rank: int = 1,
) -> dict[str, Any]:
    """Binary-search ``λ`` so analytic keep ratio ≈ ``target_keep_ratio``.

    Args:
        metas: Layer metadata with spectral ``protect`` scores.
        target_keep_ratio: Matched FLOP/param keep vs Phase-3 uniform ActSVD.
        tol: Absolute tolerance on keep ratio.
        max_iters: Binary-search iterations.
        min_rank: Floor rank per matrix.
    """
    if not metas:
        raise ValueError("No layers to allocate")
    if not 0.0 < target_keep_ratio <= 1.0:
        raise ValueError(f"target_keep_ratio must be in (0,1], got {target_keep_ratio}")

    # Bracket λ: λ=0 → min ranks; large λ → full rank.
    lo, hi = 0.0, 1.0
    while keep_ratio(metas, ranks_from_lambda(metas, hi, min_rank=min_rank)) < target_keep_ratio:
        hi *= 2.0
        if hi > 1e6:
            break

    best_ranks = ranks_from_lambda(metas, hi, min_rank=min_rank)
    best_keep = keep_ratio(metas, best_ranks)
    best_lam = hi

    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        ranks = ranks_from_lambda(metas, mid, min_rank=min_rank)
        got = keep_ratio(metas, ranks)
        if abs(got - target_keep_ratio) < abs(best_keep - target_keep_ratio):
            best_ranks, best_keep, best_lam = ranks, got, mid
        if abs(got - target_keep_ratio) <= tol:
            best_ranks, best_keep, best_lam = ranks, got, mid
            break
        if got < target_keep_ratio:
            lo = mid
        else:
            hi = mid

    logger.info(
        "Allocated ranks: target_keep=%.4f got=%.4f lambda=%.6f layers=%d",
        target_keep_ratio,
        best_keep,
        best_lam,
        len(metas),
    )
    return {
        "ranks": best_ranks,
        "lambda": best_lam,
        "target_keep_ratio": float(target_keep_ratio),
        "achieved_keep_ratio": float(best_keep),
        "num_layers": len(metas),
    }


def allocation_table(
    metas: Sequence[LayerAllocMeta],
    ranks: dict[str, int],
) -> list[dict[str, Any]]:
    """Serializable per-layer allocation rows for artifacts."""
    rows: list[dict[str, Any]] = []
    for meta in metas:
        r = int(ranks[meta.name])
        rows.append(
            {
                "name": meta.name,
                "in_features": meta.in_features,
                "out_features": meta.out_features,
                "q": meta.q,
                "rank": r,
                "rank_ratio": r / max(meta.q, 1),
                "rho_eff": meta.rho_eff,
                "compressibility": meta.compressibility,
                "stable_rank": meta.stable_rank,
                "protect": meta.protect,
                "dense_macs": meta.dense_macs,
                "lowrank_macs": meta.lowrank_macs(r),
            }
        )
    return rows
