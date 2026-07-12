"""SpectraLite Phase-4 compression: whitened SVD + spectral rank map."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import nn

from spectralite.rank_alloc import LayerAllocMeta, allocate_ranks_for_budget, allocation_table
from spectralite.spectral import spectrum_metrics
from spectralite.svd_activation import lowrank_from_factors
from spectralite.svd_vanilla import (
    DEFAULT_COMPRESS_SUFFIXES,
    flop_break_even_rank,
    set_module_by_name,
    should_compress_layer,
)
from spectralite.utils import get_logger, print_kv, print_section
from spectralite.whitening import cholesky_factor, estimate_input_covariance

logger = get_logger(__name__)


def build_whitened_svd_cache(
    model: nn.Module,
    activations: dict[str, torch.Tensor],
    *,
    ridge: float = 1e-2,
    cov_method: str = "ridge",
    suffixes: Sequence[str] = DEFAULT_COMPRESS_SUFFIXES,
) -> dict[str, dict[str, Any]]:
    """Per-layer whitened SVD factors + spectral scores (CPU float64).

    Computes once and reuses across multiple FLOP budgets.
    """
    from spectralite.stability import condition_number_from_cov

    cache: dict[str, dict[str, Any]] = {}
    linears = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and should_compress_layer(name, suffixes=suffixes)
    ]

    for name, linear in linears:
        if name not in activations:
            logger.warning("No activations for %s — skip cache entry", name)
            continue
        acts = activations[name]
        if acts.shape[-1] != linear.in_features:
            logger.warning(
                "Skip %s: act dim %d != in_features %d",
                name,
                acts.shape[-1],
                linear.in_features,
            )
            continue

        m, n = int(linear.out_features), int(linear.in_features)
        cov = estimate_input_covariance(acts, ridge=ridge, method=cov_method)
        kappa_cov = condition_number_from_cov(cov)
        L = cholesky_factor(cov.detach().double().cpu())
        w = linear.weight.detach().double().cpu()
        w_tilde = w @ L
        u, s, vh = torch.linalg.svd(w_tilde, full_matrices=False)
        metrics = spectrum_metrics(s)

        cache[name] = {
            "u": u,
            "s": s,
            "vh": vh,
            "L": L,
            "W": w,
            "in_features": n,
            "out_features": m,
            "bias": None if linear.bias is None else linear.bias.detach().cpu(),
            "dtype": linear.weight.dtype,
            "device": linear.weight.device,
            "has_bias": linear.bias is not None,
            "cov_method": cov_method,
            "kappa_cov": kappa_cov,
            **metrics,
        }
        logger.info(
            "Spectrum %s: q=%d rho_eff=%.2f s=%.3f stable_rank=%.2f protect=%.4f kappa_cov=%.3g",
            name,
            metrics["q"],
            metrics["rho_eff"],
            metrics["compressibility"],
            metrics["stable_rank"],
            metrics["protect"],
            kappa_cov,
        )
    return cache


def cache_to_alloc_metas(cache: dict[str, dict[str, Any]]) -> list[LayerAllocMeta]:
    """Convert whitened SVD cache into allocator metadata."""
    metas: list[LayerAllocMeta] = []
    for name, entry in cache.items():
        metas.append(
            LayerAllocMeta(
                name=name,
                in_features=int(entry["in_features"]),
                out_features=int(entry["out_features"]),
                q=int(entry["q"]),
                rho_eff=float(entry["rho_eff"]),
                compressibility=float(entry["compressibility"]),
                stable_rank=float(entry["stable_rank"]),
                protect=float(entry["protect"]),
            )
        )
    return metas


def _factors_from_cache(
    entry: dict[str, Any],
    rank: int,
    *,
    kappa_max: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Truncate whitened SVD and fuse ``Ŵ = UΣVᵀ L⁻¹`` into ``(U_hat, V_hat)``."""
    from spectralite.stability import (
        gate_rank_by_kappa,
        reconstruction_relative_error,
    )

    u = entry["u"]
    s = entry["s"]
    vh = entry["vh"]
    L = entry["L"]
    rank = int(min(rank, u.shape[1], vh.shape[0], s.numel()))
    kappa_trunc = float("nan")
    bumped = False
    if kappa_max is not None:
        rank, kappa_trunc, bumped = gate_rank_by_kappa(s, rank, kappa_max=float(kappa_max))
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]
    z = torch.linalg.solve_triangular(L.T, vh_r.T, upper=True)
    v_hat = z.T.contiguous()
    u_hat = (u_r * s_r.unsqueeze(0)).contiguous()
    w = entry.get("W")
    recon = (
        reconstruction_relative_error(w, u_hat, v_hat)
        if w is not None
        else float("nan")
    )
    stats = {
        "rank": rank,
        "kappa_cov": float(entry.get("kappa_cov", float("nan"))),
        "kappa_trunc": kappa_trunc,
        "kappa_bumped": bumped,
        "recon_rel_error": recon,
    }
    return u_hat, v_hat, stats


def apply_spectralite_ranks(
    model: nn.Module,
    cache: dict[str, dict[str, Any]],
    ranks: dict[str, int],
    *,
    clone: bool = True,
    kappa_max: Optional[float] = None,
) -> dict[str, Any]:
    """Replace Linears using cached whitened SVD truncated to ``ranks[name]``."""
    target = copy.deepcopy(model) if clone else model
    target.eval()

    replacements: list[dict[str, Any]] = []
    dense_params = 0
    lowrank_params = 0
    skipped: list[str] = []

    for name, entry in cache.items():
        if name not in ranks:
            skipped.append(name)
            continue
        # Resolve live module on the (possibly cloned) target for dtype/device/bias.
        linear = dict(target.named_modules()).get(name)
        if not isinstance(linear, nn.Linear):
            skipped.append(name)
            continue

        rank = int(ranks[name])
        u_hat, v_hat, stab = _factors_from_cache(entry, rank, kappa_max=kappa_max)
        rank = int(stab["rank"])
        lr = lowrank_from_factors(linear, u_hat, v_hat)
        set_module_by_name(target, name, lr)

        m, n = int(entry["out_features"]), int(entry["in_features"])
        dense_n = linear.weight.numel() + (linear.bias.numel() if linear.bias is not None else 0)
        lr_n = lr.param_count
        dense_params += dense_n
        lowrank_params += lr_n
        be = flop_break_even_rank(n, m)
        replacements.append(
            {
                "name": name,
                "in_features": n,
                "out_features": m,
                "rank": rank,
                "rank_ratio": rank / max(int(entry["q"]), 1),
                "rho_eff": float(entry["rho_eff"]),
                "compressibility": float(entry["compressibility"]),
                "stable_rank": float(entry["stable_rank"]),
                "protect": float(entry["protect"]),
                "kappa_cov": stab["kappa_cov"],
                "kappa_trunc": stab["kappa_trunc"],
                "kappa_bumped": stab["kappa_bumped"],
                "recon_rel_error": stab["recon_rel_error"],
                "flop_break_even": be,
                "below_break_even": rank < be,
                "dense_params": dense_n,
                "lowrank_params": lr_n,
                "param_ratio": lr_n / max(dense_n, 1),
            }
        )
        logger.info(
            "SpectraLite %s: (%d×%d)→r=%d (rho=%.1f protect=%.4f) params %d→%d recon=%.3g",
            name,
            m,
            n,
            rank,
            float(entry["rho_eff"]),
            float(entry["protect"]),
            dense_n,
            lr_n,
            stab["recon_rel_error"],
        )

    summary = {
        "method": "spectralite_spectral_alloc",
        "kappa_max": kappa_max,
        "num_replaced": len(replacements),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "dense_params_touched": dense_params,
        "lowrank_params_touched": lowrank_params,
        "params_saved_touched": dense_params - lowrank_params,
        "param_keep_ratio_touched": lowrank_params / max(dense_params, 1),
        "replacements": replacements,
        "mean_recon_rel_error": (
            sum(r["recon_rel_error"] for r in replacements) / max(len(replacements), 1)
        ),
        "num_kappa_bumped": sum(1 for r in replacements if r.get("kappa_bumped")),
    }
    return {"model": target, "summary": summary}


def allocate_and_compress(
    model: nn.Module,
    cache: dict[str, dict[str, Any]],
    target_keep_ratio: float,
    *,
    clone: bool = True,
    kappa_max: Optional[float] = None,
) -> dict[str, Any]:
    """Allocate ranks under ``target_keep_ratio`` and build compressed model."""
    metas = cache_to_alloc_metas(cache)
    alloc = allocate_ranks_for_budget(metas, target_keep_ratio)
    packed = apply_spectralite_ranks(
        model, cache, alloc["ranks"], clone=clone, kappa_max=kappa_max
    )
    packed["allocation"] = alloc
    packed["allocation_rows"] = allocation_table(metas, alloc["ranks"])
    packed["summary"]["target_keep_ratio"] = alloc["target_keep_ratio"]
    packed["summary"]["achieved_keep_ratio"] = float(
        packed["summary"]["param_keep_ratio_touched"]
    )
    packed["summary"]["lambda"] = alloc["lambda"]
    return packed


def print_spectralite_summary(summary: dict[str, Any], allocation: Optional[dict[str, Any]] = None) -> None:
    """Pretty-print Phase-4 compression summary."""
    print_section("SpectraLite spectral allocation")
    if allocation is not None:
        print_kv("Target keep", f"{allocation['target_keep_ratio']:.4f}")
        print_kv("Achieved keep", f"{allocation['achieved_keep_ratio']:.4f}")
        print_kv("Lambda", f"{allocation['lambda']:.6f}")
    print_kv("Layers replaced", summary["num_replaced"])
    print_kv("Dense params (touched)", f"{summary['dense_params_touched']:,}")
    print_kv("Low-rank params (touched)", f"{summary['lowrank_params_touched']:,}")
    print_kv("Keep ratio (touched)", f"{summary['param_keep_ratio_touched']:.4f}")
