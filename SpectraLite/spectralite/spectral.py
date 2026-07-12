"""Roy–Vetterli effective rank / spectral compressibility (Phase 4)."""

from __future__ import annotations

import math
from typing import Any

import torch


def singular_value_probabilities(singular_values: torch.Tensor) -> torch.Tensor:
    """Normalize singular values to a discrete distribution ``p_i = σ_i / Σσ``.

    Matches Roy & Vetterli (EUSIPCO 2007) effective-rank definition.
    """
    s = singular_values.detach().double().clamp_min(0.0).flatten()
    total = float(s.sum().item())
    if total <= 0.0 or not math.isfinite(total):
        # Degenerate: uniform over available modes
        n = max(int(s.numel()), 1)
        return torch.full((n,), 1.0 / n, dtype=torch.float64)
    return s / total


def spectral_entropy(singular_values: torch.Tensor) -> float:
    """Shannon entropy ``H = -Σ p_i log p_i`` of the normalized spectrum."""
    p = singular_value_probabilities(singular_values)
    # Avoid 0 * log 0
    p = p[p > 0]
    if p.numel() == 0:
        return 0.0
    h = float((-p * torch.log(p)).sum().item())
    return h if math.isfinite(h) else 0.0


def effective_rank(singular_values: torch.Tensor) -> float:
    """Effective rank ``ρ_eff = exp(H) ∈ [1, q]`` (Roy & Vetterli)."""
    q = max(int(singular_values.numel()), 1)
    rho = math.exp(spectral_entropy(singular_values))
    if not math.isfinite(rho):
        return float(q)
    return float(min(max(rho, 1.0), float(q)))


def spectral_compressibility(singular_values: torch.Tensor) -> float:
    """``s = 1 - ρ_eff / q`` — high ⇒ steep decay ⇒ safe to truncate."""
    q = max(int(singular_values.numel()), 1)
    rho = effective_rank(singular_values)
    return float(1.0 - rho / float(q))


def stable_rank(singular_values: torch.Tensor) -> float:
    """Stable-rank importance proxy ``‖σ‖₂² / σ₁²`` (= ‖W̃‖_F²/σ₁²)."""
    s = singular_values.detach().double().clamp_min(0.0).flatten()
    if s.numel() == 0:
        return 1.0
    s1 = float(s[0].item())
    if s1 <= 0.0:
        return 1.0
    fro2 = float((s * s).sum().item())
    val = fro2 / (s1 * s1)
    return float(val) if math.isfinite(val) else 1.0


def protect_score(
    *,
    rho_eff: float,
    stable_rank_val: float,
    q: int,
    mode: str = "full",
) -> float:
    """Per-matrix protect score for rank allocation (Phase 4/7).

    Modes:
      - ``full``: (ρ_eff/q) * (stable_rank/q) — default SpectraLite
      - ``rho``: ρ_eff/q — spectral-only
      - ``stable_rank``: stable_rank/q — importance-only
      - ``uniform``: 1.0 — equal protect (allocator ≈ uniform keep)
    """
    q = max(int(q), 1)
    mode = (mode or "full").lower()
    if mode in {"uniform", "equal"}:
        return 1.0
    if mode in {"rho", "rho_only", "spectral"}:
        return float(rho_eff) / float(q)
    if mode in {"stable_rank", "sr", "stable"}:
        return float(stable_rank_val) / float(q)
    # full
    return (float(rho_eff) / float(q)) * (float(stable_rank_val) / float(q))


def spectrum_metrics(singular_values: torch.Tensor, *, protect_mode: str = "full") -> dict[str, Any]:
    """Bundle spectral statistics used by the Phase-4 allocator."""
    q = int(singular_values.numel())
    rho = effective_rank(singular_values)
    s_comp = spectral_compressibility(singular_values)
    a = stable_rank(singular_values)
    return {
        "q": q,
        "entropy": spectral_entropy(singular_values),
        "rho_eff": rho,
        "compressibility": s_comp,
        "stable_rank": a,
        "protect_mode": protect_mode,
        "protect": protect_score(rho_eff=rho, stable_rank_val=a, q=q, mode=protect_mode),
    }
