"""Activation-aware truncated SVD (SVD-LLM / ASVD whitening recipe)."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import nn

from spectralite.lowrank import LowRankLinear
from spectralite.svd_vanilla import (
    DEFAULT_COMPRESS_SUFFIXES,
    choose_rank,
    flop_break_even_rank,
    set_module_by_name,
    should_compress_layer,
)
from spectralite.utils import get_logger, print_kv, print_section
from spectralite.whitening import cholesky_factor, estimate_input_covariance

logger = get_logger(__name__)


def _activation_aware_factors(
    weight: torch.Tensor,
    cov_in: torch.Tensor,
    rank: int,
    *,
    kappa_max: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Whitened truncated SVD → fused factors ``(U_hat[out,r], V_hat[r,in])``.

    Follows SVD-LLM / ASVD:
      C ≈ LLᵀ,  W̃ = W L,  truncate SVD(W̃),  Ŵ = UΣVᵀ L⁻¹
    with σ fused into ``U_hat``.

    Runs in float64 on CPU so weight (often CUDA) and covariance (CPU) match,
    then the caller casts factors back onto the layer device.
    """
    from spectralite.stability import (
        condition_number_from_cov,
        gate_rank_by_kappa,
        reconstruction_relative_error,
    )

    w = weight.detach().double().cpu()  # (out, in)
    L = cholesky_factor(cov_in.detach().double().cpu())  # (in, in)
    w_tilde = w @ L
    u, s, vh = torch.linalg.svd(w_tilde, full_matrices=False)
    rank = int(min(rank, u.shape[1], vh.shape[0]))
    kappa_before = float("nan")
    bumped = False
    if kappa_max is not None:
        rank, kappa_before, bumped = gate_rank_by_kappa(s, rank, kappa_max=float(kappa_max))
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    # V_hat = Vh @ inv(L)  via triangular solve: L.T Z = Vh.T ⇒ Z.T = Vh inv(L)
    z = torch.linalg.solve_triangular(L.T, vh_r.T, upper=True)
    v_hat = z.T.contiguous()
    u_hat = (u_r * s_r.unsqueeze(0)).contiguous()
    stats = {
        "rank": rank,
        "kappa_cov": condition_number_from_cov(cov_in),
        "kappa_trunc": float(kappa_before) if kappa_max is not None else float("nan"),
        "kappa_bumped": bumped,
        "recon_rel_error": reconstruction_relative_error(w, u_hat, v_hat),
    }
    return u_hat, v_hat, stats


def lowrank_from_factors(
    linear: nn.Linear,
    u_hat: torch.Tensor,
    v_hat: torch.Tensor,
) -> LowRankLinear:
    """Build ``LowRankLinear`` from float64 factors (cast to layer dtype/device)."""
    out_features, rank = u_hat.shape
    rank2, in_features = v_hat.shape
    assert rank == rank2
    module = LowRankLinear(
        in_features,
        out_features,
        rank,
        bias=linear.bias is not None,
        dtype=linear.weight.dtype,
        device=linear.weight.device,
    )
    with torch.no_grad():
        module.v.weight.copy_(v_hat.to(dtype=module.v.weight.dtype))
        module.u.weight.copy_(u_hat.to(dtype=module.u.weight.dtype))
        if linear.bias is not None and module.u.bias is not None:
            module.u.bias.copy_(linear.bias.detach().to(dtype=module.u.bias.dtype))
    return module


def apply_activation_aware_svd(
    model: nn.Module,
    activations: dict[str, torch.Tensor],
    rank_ratio: float,
    *,
    ridge: float = 1e-2,
    cov_method: str = "ridge",
    kappa_max: Optional[float] = None,
    latency_gate: bool = False,
    kappa_speed: float = 1.0,
    suffixes: Sequence[str] = DEFAULT_COMPRESS_SUFFIXES,
    clone: bool = True,
) -> dict[str, Any]:
    """Replace Linear layers using activation-whitened truncated SVD.

    Layers without collected activations are left dense.
    When ``latency_gate=True``, layers with ``r ≥ κ_speed·mn/(m+n)`` stay dense
    (Phase 6 feasibility gate).
    """
    from spectralite.latency_gate import gate_decision

    target = copy.deepcopy(model) if clone else model
    target.eval()

    replacements: list[dict[str, Any]] = []
    gated_dense: list[dict[str, Any]] = []
    dense_params = 0
    lowrank_params = 0
    skipped: list[str] = []

    linears = [
        (name, mod)
        for name, mod in target.named_modules()
        if isinstance(mod, nn.Linear) and should_compress_layer(name, suffixes=suffixes)
    ]

    for name, linear in linears:
        if name not in activations:
            skipped.append(name)
            continue
        acts = activations[name]
        if acts.shape[-1] != linear.in_features:
            logger.warning(
                "Skip %s: act dim %d != in_features %d",
                name,
                acts.shape[-1],
                linear.in_features,
            )
            skipped.append(name)
            continue

        m, n = int(linear.out_features), int(linear.in_features)
        rank = choose_rank(n, m, rank_ratio)
        decision = gate_decision(n, m, rank, kappa_speed=kappa_speed)
        dense_n = linear.weight.numel() + (linear.bias.numel() if linear.bias is not None else 0)

        if latency_gate and not decision["passes_gate"]:
            gated_dense.append({"name": name, **decision, "dense_params": dense_n})
            dense_params += dense_n
            lowrank_params += dense_n  # unchanged
            logger.info(
                "Latency-gate KEEP DENSE %s: r=%d ≥ thresh=%.1f (be=%.1f κ=%.2f)",
                name,
                rank,
                decision["latency_threshold"],
                decision["flop_break_even"],
                kappa_speed,
            )
            continue

        cov = estimate_input_covariance(acts, ridge=ridge, method=cov_method)
        u_hat, v_hat, stab = _activation_aware_factors(
            linear.weight, cov, rank, kappa_max=kappa_max
        )
        rank = int(stab["rank"])
        # Re-check gate after κ bump (Phase 5+6 interaction)
        if latency_gate:
            decision = gate_decision(n, m, rank, kappa_speed=kappa_speed)
            if not decision["passes_gate"]:
                gated_dense.append(
                    {
                        "name": name,
                        **decision,
                        "dense_params": dense_n,
                        "note": "failed_gate_after_kappa_bump",
                    }
                )
                dense_params += dense_n
                lowrank_params += dense_n
                continue

        lr = lowrank_from_factors(linear, u_hat, v_hat)
        set_module_by_name(target, name, lr)

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
                "rank_ratio": rank_ratio,
                "ridge": ridge,
                "cov_method": cov_method,
                "kappa_max": kappa_max,
                "latency_gate": latency_gate,
                "kappa_speed": kappa_speed,
                "latency_threshold": decision["latency_threshold"],
                "kappa_cov": stab["kappa_cov"],
                "kappa_trunc": stab["kappa_trunc"],
                "kappa_bumped": stab["kappa_bumped"],
                "recon_rel_error": stab["recon_rel_error"],
                "flop_break_even": be,
                "below_break_even": rank < be,
                "dense_params": dense_n,
                "lowrank_params": lr_n,
                "param_ratio": lr_n / max(dense_n, 1),
                "num_act_tokens": int(acts.shape[0]),
            }
        )
        logger.info(
            "ActSVD %s: (%d×%d)→r=%d | acts=%d | params %d→%d | kappa_cov=%.3g recon=%.3g",
            name,
            m,
            n,
            rank,
            acts.shape[0],
            dense_n,
            lr_n,
            stab["kappa_cov"],
            stab["recon_rel_error"],
        )

    summary = {
        "method": "activation_aware_svd",
        "rank_ratio": rank_ratio,
        "ridge": ridge,
        "cov_method": cov_method,
        "kappa_max": kappa_max,
        "latency_gate": latency_gate,
        "kappa_speed": kappa_speed,
        "num_replaced": len(replacements),
        "num_gated_dense": len(gated_dense),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "gated_dense": gated_dense,
        "dense_params_touched": dense_params,
        "lowrank_params_touched": lowrank_params,
        "params_saved_touched": dense_params - lowrank_params,
        "param_keep_ratio_touched": lowrank_params / max(dense_params, 1),
        "replacements": replacements,
        "mean_recon_rel_error": (
            sum(r["recon_rel_error"] for r in replacements) / max(len(replacements), 1)
            if replacements
            else float("nan")
        ),
        "num_kappa_bumped": sum(1 for r in replacements if r.get("kappa_bumped")),
    }
    return {"model": target, "summary": summary}


def print_actsvd_summary(summary: dict[str, Any]) -> None:
    """Pretty-print activation-aware SVD summary."""
    print_section(
        f"Activation-aware SVD — ratio={summary['rank_ratio']} "
        f"cov={summary.get('cov_method', 'ridge')} "
        f"gate={summary.get('latency_gate', False)}"
    )
    print_kv("Layers replaced", summary["num_replaced"])
    print_kv("Layers gated dense", summary.get("num_gated_dense", 0))
    print_kv("Layers skipped", summary["num_skipped"])
    print_kv("Dense params (touched)", f"{summary['dense_params_touched']:,}")
    print_kv("Low-rank params (touched)", f"{summary['lowrank_params_touched']:,}")
    print_kv("Keep ratio (touched)", f"{summary['param_keep_ratio_touched']:.4f}")
