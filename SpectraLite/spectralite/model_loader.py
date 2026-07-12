"""Model and tokenizer loading utilities for SpectraLite Phase 0."""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import nn

from spectralite.config import Config, default_config
from spectralite.system import cuda_available, resolve_device
from spectralite.utils import format_bytes, format_params, get_logger, print_kv, print_section

logger = get_logger(__name__)

# Hugging Face types are resolved at call time so importing this module
# does not require transformers until a model is actually loaded.
ModelType = Any
TokenizerType = Any


def _require_transformers():
    """Import transformers lazily with a clear error if missing."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for model loading. "
            "Install Phase 0 deps: pip install -r requirements.txt"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(dtype: Optional[str] = None, prefer_fp16_on_cuda: bool = True) -> torch.dtype:
    """Map a dtype string to a :class:`torch.dtype`.

    When CUDA is available and ``prefer_fp16_on_cuda`` is true, FP16 is used
    regardless of a ``float32`` request from callers that forget to branch —
    Phase 0 always prefers FP16 on GPU as specified by the project brief.

    Args:
        dtype: One of ``float16``, ``fp16``, ``float32``, ``fp32``, ``bfloat16``,
            ``bf16``. If ``None``, choose FP16 on CUDA else FP32.
        prefer_fp16_on_cuda: Force FP16 when CUDA is visible.

    Returns:
        Resolved torch dtype.
    """
    if prefer_fp16_on_cuda and cuda_available():
        return torch.float16

    if dtype is None:
        return torch.float16 if cuda_available() else torch.float32

    key = dtype.lower().strip()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype!r}")
    return mapping[key]


def load_tokenizer(
    model_name: Optional[str] = None,
    *,
    config: Optional[Config] = None,
) -> TokenizerType:
    """Load a Hugging Face tokenizer for the configured model.

    Args:
        model_name: Override model identifier. Defaults to ``config.model_name``.
        config: Optional :class:`Config`. Uses Phase 0 defaults when omitted.

    Returns:
        Loaded tokenizer with ``pad_token`` set when missing.
    """
    _, AutoTokenizer = _require_transformers()
    cfg = config or default_config()
    name = model_name or cfg.model_name
    logger.info("Loading tokenizer: %s", name)

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

    # Causal LMs often omit an explicit pad token; reuse EOS for batching safety.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer had no pad_token; set pad_token = eos_token")

    return tokenizer


def load_model(
    model_name: Optional[str] = None,
    *,
    config: Optional[Config] = None,
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
    device_map: Optional[str] = None,
) -> ModelType:
    """Load a causal language model with Phase 0 defaults.

    Uses ``device_map="auto"`` and FP16 whenever CUDA is available.

    Args:
        model_name: Override model identifier.
        config: Optional :class:`Config`.
        torch_dtype: Explicit dtype override (string or torch dtype).
        device_map: Hugging Face device-map strategy.

    Returns:
        Loaded :class:`~transformers.PreTrainedModel` in eval mode.
    """
    AutoModelForCausalLM, _ = _require_transformers()
    cfg = config or default_config()
    name = model_name or cfg.model_name
    map_strategy = device_map if device_map is not None else cfg.device_map

    if isinstance(torch_dtype, torch.dtype):
        dtype = torch_dtype
    elif isinstance(torch_dtype, str):
        dtype = resolve_torch_dtype(torch_dtype, prefer_fp16_on_cuda=cuda_available())
    else:
        dtype = resolve_torch_dtype(cfg.dtype, prefer_fp16_on_cuda=cuda_available())

    logger.info(
        "Loading model: %s | dtype=%s | device_map=%s",
        name,
        dtype,
        map_strategy,
    )

    # Prefer `dtype=` (current HF API); fall back to `torch_dtype=` for older installs.
    load_kwargs = {
        "device_map": map_strategy,
        "trust_remote_code": cfg.trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, **load_kwargs
        )
    model.eval()
    return model


def load_model_and_tokenizer(
    model_name: Optional[str] = None,
    *,
    config: Optional[Config] = None,
) -> tuple[ModelType, TokenizerType]:
    """Load both model and tokenizer with shared configuration.

    Args:
        model_name: Optional HF model id override.
        config: Optional :class:`Config`.

    Returns:
        ``(model, tokenizer)`` pair.
    """
    cfg = config or default_config()
    name = model_name or cfg.model_name
    tokenizer = load_tokenizer(name, config=cfg)
    model = load_model(name, config=cfg)
    return model, tokenizer


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Dictionary with ``total`` and ``trainable`` integer counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def estimate_model_size_bytes(model: nn.Module) -> int:
    """Estimate parameter memory footprint from tensor storage.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Sum of ``numel * element_size`` over all parameters (bytes).
    """
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def get_model_device(model: nn.Module) -> torch.device:
    """Infer the primary device of a (possibly device-mapped) model.

    Args:
        model: Loaded model.

    Returns:
        Device of the first parameter, or CPU if the model has none.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Infer the primary floating dtype of a model.

    Args:
        model: Loaded model.

    Returns:
        Dtype of the first floating-point parameter, else ``torch.float32``.
    """
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def generate_text(
    model: ModelType,
    tokenizer: TokenizerType,
    prompt: str,
    *,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Run a short greedy (or sampled) generation for smoke testing.

    Args:
        model: Causal LM.
        tokenizer: Matching tokenizer.
        prompt: Input text.
        max_new_tokens: Generation length budget.
        do_sample: Whether to sample (default greedy for reproducibility).
        temperature: Sampling temperature (ignored when ``do_sample=False``).

    Returns:
        Full decoded string including the prompt prefix.
    """
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generate_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def print_model_load_summary(
    model: ModelType,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Print a compact load summary and return the same fields.

    Args:
        model: Loaded model.
        model_name: HF identifier string for display.

    Returns:
        Summary dictionary.
    """
    counts = count_parameters(model)
    size_bytes = estimate_model_size_bytes(model)
    device = get_model_device(model)
    dtype = get_model_dtype(model)

    summary = {
        "model_name": model_name,
        "architecture": model.__class__.__name__,
        "total_parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "total_parameters_human": format_params(counts["total"]),
        "trainable_parameters_human": format_params(counts["trainable"]),
        "model_size_bytes": size_bytes,
        "model_size": format_bytes(size_bytes),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "resolved_runtime_device": resolve_device(),
    }

    print_section("Model Load Summary")
    print_kv("Model Name", summary["model_name"])
    print_kv("Architecture", summary["architecture"])
    print_kv("Total Parameters", f"{summary['total_parameters_human']} ({counts['total']:,})")
    print_kv(
        "Trainable Parameters",
        f"{summary['trainable_parameters_human']} ({counts['trainable']:,})",
    )
    print_kv("Model Size (params)", summary["model_size"])
    print_kv("Tensor Data Type", summary["dtype"])
    print_kv("Current Device", summary["device"])
    return summary
