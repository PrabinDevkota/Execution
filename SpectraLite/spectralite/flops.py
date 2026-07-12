"""FLOP accounting helpers for SpectraLite Phase 1."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from spectralite.model_loader import count_parameters, estimate_model_size_bytes, get_model_device
from spectralite.utils import format_bytes, get_logger

logger = get_logger(__name__)


def analytic_dense_linear_flops(m: int, n: int, batch_tokens: int = 1) -> int:
    """Return MACs for a dense GEMM ``(batch_tokens, n) @ (n, m)`` → approx ``2*batch*m*n`` FLOPs.

    We follow the common LLM convention of counting multiply-adds as 2 FLOPs.
    """
    return int(2 * batch_tokens * m * n)


def analytic_model_param_stats(model: nn.Module) -> dict[str, Any]:
    """Parameter count and weight memory for the dense model."""
    counts = count_parameters(model)
    size_bytes = estimate_model_size_bytes(model)
    return {
        "param_count": counts["total"],
        "trainable_params": counts["trainable"],
        "param_memory_bytes": size_bytes,
        "param_memory_mb": size_bytes / (1024**2),
        "param_memory_human": format_bytes(size_bytes),
    }


@torch.inference_mode()
def measure_forward_flops(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Count forward FLOPs with PyTorch ``FlopCounterMode`` (uncompiled model).

    Args:
        model: Causal LM in eval mode.
        input_ids: ``[batch, seq]`` token ids on the model device.
        attention_mask: Optional mask.

    Returns:
        Dict with total FLOPs and a short per-module breakdown (top entries).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        raise ImportError(
            "FlopCounterMode requires torch>=2.2. Upgrade PyTorch."
        ) from exc

    device = get_model_device(model)
    input_ids = input_ids.to(device)
    kwargs: dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        _ = model(**kwargs)

    total = int(flop_counter.get_total_flops())
    # Module table can be large; keep a compact summary string.
    try:
        table = flop_counter.get_table(depth=2)
    except Exception:
        table = ""
    return {
        "empirical_flops_fwd": total,
        "empirical_gflops_fwd": total / 1e9,
        "flop_table_preview": table[:2000] if isinstance(table, str) else str(table)[:2000],
    }


def measure_calflops_decode(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompt: str = "Hello",
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Optional decode-oriented FLOP estimate via ``calflops`` if installed.

    Returns zeros / n/a fields when calflops is unavailable or fails.
    """
    try:
        from calflops import calculate_flops
    except ImportError:
        logger.warning("calflops not installed; skipping decode FLOP estimate")
        return {
            "calflops_mflops_per_token": None,
            "calflops_raw": None,
            "calflops_error": "not_installed",
        }

    try:
        # calflops API varies; use a conservative generate-style probe when possible.
        device = get_model_device(model)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        flops, macs, params = calculate_flops(
            model=model,
            kwargs=dict(inputs),
            print_results=False,
            output_as_string=False,
        )
        # convert to MFLOPs per forward token (rough); caller can reinterpret.
        seq = int(inputs["input_ids"].numel())
        mflops_per_tok = (float(flops) / max(seq, 1)) / 1e6
        return {
            "calflops_mflops_per_token": mflops_per_tok,
            "calflops_raw": {"flops": flops, "macs": macs, "params": params},
            "calflops_error": None,
        }
    except Exception as exc:  # noqa: BLE001 — optional path must not kill Phase 1
        logger.warning("calflops failed: %s", exc)
        return {
            "calflops_mflops_per_token": None,
            "calflops_raw": None,
            "calflops_error": str(exc),
        }
