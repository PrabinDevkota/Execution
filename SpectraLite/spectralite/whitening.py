"""Activation covariance + Cholesky whitening (SVD-LLM / ASVD style)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import nn

from spectralite.calibration import move_batch_to_device
from spectralite.svd_vanilla import DEFAULT_COMPRESS_SUFFIXES, should_compress_layer
from spectralite.utils import get_logger

logger = get_logger(__name__)


@torch.inference_mode()
def collect_linear_input_activations(
    model: nn.Module,
    batches: Sequence[dict[str, torch.Tensor]],
    *,
    layer_names: Optional[Sequence[str]] = None,
    max_tokens_per_layer: int = 50_000,
) -> dict[str, torch.Tensor]:
    """Hook selected ``nn.Linear`` modules and collect flattened input activations.

    Returns:
        Mapping ``name → FloatTensor[N, in_features]`` on CPU.
    """
    device = next(model.parameters()).device
    if layer_names is None:
        layer_names = [
            name
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and should_compress_layer(name)
        ]

    caches: dict[str, list[torch.Tensor]] = {n: [] for n in layer_names}
    handles = []

    def _make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float().cpu()
            # Reservoir-style cap
            have = sum(t.shape[0] for t in caches[name])
            if have >= max_tokens_per_layer:
                return
            remain = max_tokens_per_layer - have
            caches[name].append(x[:remain])

        return hook

    name_set = set(layer_names)
    for name, mod in model.named_modules():
        if name in name_set and isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(_make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            batch = move_batch_to_device(batch, device)
            _ = model(**batch)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    out: dict[str, torch.Tensor] = {}
    for name, parts in caches.items():
        if not parts:
            logger.warning("No activations collected for %s", name)
            continue
        out[name] = torch.cat(parts, dim=0)
        logger.info("Activations %s: %s", name, tuple(out[name].shape))
    return out


def estimate_input_covariance(
    activations: torch.Tensor,
    *,
    ridge: float = 1e-2,
    method: str = "ridge",
    max_samples: int = 8192,
    seed: int = 42,
) -> torch.Tensor:
    """Estimate input covariance for whitening.

    Args:
        activations: ``[N, in_features]``.
        ridge: Diagonal jitter for ``method="ridge"`` (ASVD-style Phase 3).
        method: ``"ridge"`` or ``"ledoit_wolf"`` (Phase 5).
        max_samples: Subsample cap for Ledoit–Wolf.
        seed: Subsample RNG seed.
    """
    if method in {"ledoit_wolf", "lw", "ledoit-wolf"}:
        from spectralite.stability import estimate_covariance_ledoit_wolf

        cov, meta = estimate_covariance_ledoit_wolf(
            activations,
            max_samples=max_samples,
            seed=seed,
        )
        logger.info(
            "Ledoit–Wolf cov: dim=%s n=%s shrinkage=%s kappa=%.3g",
            meta.get("dim"),
            meta.get("n_samples_used"),
            meta.get("shrinkage"),
            meta.get("kappa_cov"),
        )
        return cov

    if activations.ndim != 2:
        raise ValueError("activations must be 2D [N, in_features]")
    n, d = activations.shape
    if n < 2:
        raise ValueError("Need at least 2 activation rows")
    x = activations.double()
    c = (x.T @ x) / float(n)
    c = c + ridge * torch.eye(d, dtype=c.dtype, device=c.device)
    # Symmetrize numerically
    c = 0.5 * (c + c.T)
    return c


def cholesky_factor(cov: torch.Tensor) -> torch.Tensor:
    """Return lower-triangular ``L`` with ``cov = L @ L.T`` (float64)."""
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError:
        # Eigen floor if still indefinite
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=1e-8)
        cov_pd = (evecs * evals.unsqueeze(0)) @ evecs.T
        return torch.linalg.cholesky(cov_pd)
