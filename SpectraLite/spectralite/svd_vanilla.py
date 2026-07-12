"""Phase 2 — vanilla truncated SVD (Eckart–Young) with uniform rank ratios."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import nn

from spectralite.lowrank import LowRankLinear
from spectralite.model_analysis import list_linear_layers
from spectralite.utils import get_logger, print_kv, print_section

logger = get_logger(__name__)

# Compress attention + MLP projections; keep lm_head / embeddings dense.
DEFAULT_COMPRESS_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ATTN_COMPRESS_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)

MLP_COMPRESS_SUFFIXES: tuple[str, ...] = (
    "fc1",
    "fc2",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _leaf_name(qualified: str) -> str:
    return qualified.rsplit(".", 1)[-1]


def should_compress_layer(
    qualified_name: str,
    *,
    suffixes: Sequence[str] = DEFAULT_COMPRESS_SUFFIXES,
    skip_substrings: Sequence[str] = ("lm_head", "embed"),
) -> bool:
    """Return True if this Linear should be SVD-compressed."""
    lower = qualified_name.lower()
    if any(s in lower for s in skip_substrings):
        return False
    leaf = _leaf_name(qualified_name)
    return leaf in suffixes


def choose_rank(in_features: int, out_features: int, rank_ratio: float) -> int:
    """Uniform rank: ``r = max(1, round(ratio * min(m, n)))`` capped by min(m,n)."""
    if not 0.0 < rank_ratio <= 1.0:
        raise ValueError(f"rank_ratio must be in (0, 1], got {rank_ratio}")
    full = min(in_features, out_features)
    rank = max(1, int(round(rank_ratio * full)))
    return int(min(rank, full))


def flop_break_even_rank(in_features: int, out_features: int) -> float:
    """Exact FLOP break-even ``r < mn/(m+n)`` for replacing one GEMM with two."""
    m, n = out_features, in_features
    return (m * n) / (m + n)


def set_module_by_name(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace a nested submodule given its ``named_modules`` path."""
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_vanilla_svd(
    model: nn.Module,
    rank_ratio: float,
    *,
    suffixes: Sequence[str] = DEFAULT_COMPRESS_SUFFIXES,
    clone: bool = True,
) -> dict[str, Any]:
    """Replace selected ``nn.Linear`` layers with fused truncated-SVD factors.

    Args:
        model: Dense model (modified in-place unless ``clone=True``).
        rank_ratio: Uniform fraction of ``min(m,n)`` retained.
        suffixes: Leaf-name allow-list (attention/MLP).
        clone: If True, deep-copy the model first.

    Returns:
        Dict with the (possibly copied) model and a per-layer replacement log.
    """
    target = copy.deepcopy(model) if clone else model
    target.eval()

    replacements: list[dict[str, Any]] = []
    dense_params = 0
    lowrank_params = 0

    # Collect first to avoid mutating while iterating named_modules.
    linears = [
        (name, mod)
        for name, mod in target.named_modules()
        if isinstance(mod, nn.Linear) and should_compress_layer(name, suffixes=suffixes)
    ]

    for name, linear in linears:
        m, n = int(linear.out_features), int(linear.in_features)
        rank = choose_rank(n, m, rank_ratio)
        be = flop_break_even_rank(n, m)
        dense_n = linear.weight.numel() + (linear.bias.numel() if linear.bias is not None else 0)

        lr = LowRankLinear.from_linear(linear, rank)
        set_module_by_name(target, name, lr)

        lr_n = lr.param_count
        dense_params += dense_n
        lowrank_params += lr_n
        replacements.append(
            {
                "name": name,
                "in_features": n,
                "out_features": m,
                "rank": rank,
                "rank_ratio": rank_ratio,
                "flop_break_even": be,
                "below_break_even": rank < be,
                "dense_params": dense_n,
                "lowrank_params": lr_n,
                "param_ratio": lr_n / max(dense_n, 1),
            }
        )
        logger.info(
            "SVD %s: (%d×%d) → r=%d (%.0f%% of min-dim); params %d→%d",
            name,
            m,
            n,
            rank,
            100.0 * rank / max(min(m, n), 1),
            dense_n,
            lr_n,
        )

    saved = dense_params - lowrank_params
    summary = {
        "rank_ratio": rank_ratio,
        "num_replaced": len(replacements),
        "dense_params_touched": dense_params,
        "lowrank_params_touched": lowrank_params,
        "params_saved_touched": saved,
        "param_keep_ratio_touched": lowrank_params / max(dense_params, 1),
        "replacements": replacements,
    }
    return {"model": target, "summary": summary}


def print_svd_summary(summary: dict[str, Any]) -> None:
    """Pretty-print a vanilla-SVD compression summary."""
    print_section(f"Vanilla SVD — rank_ratio={summary['rank_ratio']}")
    print_kv("Layers replaced", summary["num_replaced"])
    print_kv("Dense params (touched)", f"{summary['dense_params_touched']:,}")
    print_kv("Low-rank params (touched)", f"{summary['lowrank_params_touched']:,}")
    print_kv("Params saved (touched)", f"{summary['params_saved_touched']:,}")
    print_kv(
        "Keep ratio (touched)",
        f"{summary['param_keep_ratio_touched']:.4f}",
    )
    below = sum(1 for r in summary["replacements"] if r["below_break_even"])
    print_kv("Layers below FLOP break-even", f"{below}/{summary['num_replaced']}")
