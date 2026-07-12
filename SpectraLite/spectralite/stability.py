"""Phase 5 stability safeguards: Ledoit–Wolf covariance + κ gating."""

from __future__ import annotations

from typing import Any, Optional

import torch

from spectralite.utils import get_logger

logger = get_logger(__name__)


def estimate_covariance_ledoit_wolf(
    activations: torch.Tensor,
    *,
    max_samples: int = 8192,
    ridge_floor: float = 1e-8,
    seed: int = 42,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Ledoit–Wolf shrunk covariance of activation rows (float64 CPU).

    Uses ``sklearn.covariance.ledoit_wolf`` (centered). A tiny diagonal floor
    keeps the estimate Cholesky-factorizable for whitening.
    """
    if activations.ndim != 2:
        raise ValueError("activations must be 2D [N, in_features]")
    n, d = activations.shape
    if n < 2:
        raise ValueError("Need at least 2 activation rows")

    x = activations.detach().float().cpu()
    if n > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_samples]
        x = x[idx]
        n_used = int(max_samples)
    else:
        n_used = int(n)

    try:
        from sklearn.covariance import ledoit_wolf

        cov_np, shrinkage = ledoit_wolf(x.numpy(), assume_centered=False)
        cov = torch.from_numpy(cov_np).double()
        shrinkage_f = float(shrinkage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ledoit–Wolf failed (%s); falling back to sample cov + ridge", exc)
        x64 = x.double()
        x64 = x64 - x64.mean(dim=0, keepdim=True)
        cov = (x64.T @ x64) / max(float(n_used - 1), 1.0)
        shrinkage_f = float("nan")

    cov = 0.5 * (cov + cov.T)
    cov = cov + ridge_floor * torch.eye(d, dtype=cov.dtype, device=cov.device)
    meta = {
        "method": "ledoit_wolf",
        "n_samples_used": n_used,
        "n_samples_total": int(n),
        "dim": int(d),
        "shrinkage": shrinkage_f,
        "ridge_floor": ridge_floor,
        "kappa_cov": condition_number_from_cov(cov),
    }
    return cov, meta


def condition_number_from_cov(cov: torch.Tensor) -> float:
    """κ₂(C) ≈ λ_max / λ_min via symmetric eigendecomposition."""
    c = cov.detach().double()
    evals = torch.linalg.eigvalsh(c).clamp_min(1e-12)
    return float((evals[-1] / evals[0]).item())


def condition_number_from_singular_values(
    singular_values: torch.Tensor,
    rank: Optional[int] = None,
) -> float:
    """Truncation condition ``σ₁ / σ_r`` (full spectrum if ``rank`` is None)."""
    s = singular_values.detach().double().flatten().clamp_min(0.0)
    if s.numel() == 0:
        return float("inf")
    r = int(s.numel() if rank is None else max(1, min(rank, s.numel())))
    s1 = float(s[0].item())
    sr = float(s[r - 1].item())
    if sr <= 0.0:
        return float("inf")
    return s1 / sr


def gate_rank_by_kappa(
    singular_values: torch.Tensor,
    rank: int,
    *,
    kappa_max: float = 1e4,
) -> tuple[int, float, bool]:
    """Increase ``rank`` until ``σ₁/σ_r ≤ kappa_max`` (or full rank).

    Returns ``(gated_rank, kappa_at_rank, was_bumped)``.
    """
    s = singular_values.detach().double().flatten()
    q = int(s.numel())
    if q == 0:
        return 1, float("inf"), False
    r = int(max(1, min(rank, q)))
    kappa = condition_number_from_singular_values(s, r)
    bumped = False
    while r < q and kappa > kappa_max:
        r += 1
        kappa = condition_number_from_singular_values(s, r)
        bumped = True
    return r, float(kappa), bumped


def reconstruction_relative_error(
    weight: torch.Tensor,
    u_hat: torch.Tensor,
    v_hat: torch.Tensor,
) -> float:
    """``‖W − ÛV̂‖_F / ‖W‖_F`` for fused factors (CPU float64)."""
    w = weight.detach().double().cpu()
    approx = (u_hat.double().cpu() @ v_hat.double().cpu())
    num = torch.linalg.norm(w - approx, ord="fro")
    den = torch.linalg.norm(w, ord="fro").clamp_min(1e-12)
    return float((num / den).item())
