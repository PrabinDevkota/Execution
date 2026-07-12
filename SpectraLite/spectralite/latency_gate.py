"""Latency feasibility gate: only factorize when ``r < κ · mn/(m+n)``."""

from __future__ import annotations

from typing import Any

from spectralite.svd_vanilla import flop_break_even_rank
from spectralite.utils import get_logger

logger = get_logger(__name__)


def latency_break_even(
    in_features: int,
    out_features: int,
    *,
    kappa_speed: float = 1.0,
) -> float:
    """Hardware-aware break-even rank ``κ_speed · mn/(m+n)``.

    ``κ_speed < 1`` is stricter (harder to pass → more layers stay dense).
    """
    if kappa_speed <= 0:
        raise ValueError(f"kappa_speed must be > 0, got {kappa_speed}")
    return float(kappa_speed) * flop_break_even_rank(in_features, out_features)


def passes_latency_gate(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    kappa_speed: float = 1.0,
) -> bool:
    """Return True iff low-rank factorization is FLOP/latency-feasible."""
    return float(rank) < latency_break_even(
        in_features, out_features, kappa_speed=kappa_speed
    )


def gate_decision(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    kappa_speed: float = 1.0,
) -> dict[str, Any]:
    """Structured gate decision for logging / artifacts."""
    be = flop_break_even_rank(in_features, out_features)
    thresh = latency_break_even(in_features, out_features, kappa_speed=kappa_speed)
    ok = float(rank) < thresh
    return {
        "rank": int(rank),
        "flop_break_even": float(be),
        "kappa_speed": float(kappa_speed),
        "latency_threshold": float(thresh),
        "passes_gate": bool(ok),
        "action": "compress" if ok else "keep_dense",
    }
