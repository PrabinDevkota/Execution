"""Model architecture inspection utilities for SpectraLite.

These helpers inventory Transformer blocks and every ``nn.Linear`` layer.
The Linear catalog is the foundation for later SVD compression targeting;
Phase 0 only prints and returns the inventory — it does not modify weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

from torch import nn

from spectralite.model_loader import (
    count_parameters,
    estimate_model_size_bytes,
    get_model_device,
    get_model_dtype,
)
from spectralite.utils import format_bytes, format_params, print_kv, print_section


@dataclass(frozen=True)
class LinearLayerInfo:
    """Metadata for a single ``nn.Linear`` module.

    Attributes:
        name: Fully-qualified module name (``named_modules`` path).
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        weight_shape: Shape of the weight matrix as a tuple.
        has_bias: Whether the layer has a bias parameter.
        param_count: Total parameter elements (weight + optional bias).
    """

    name: str
    in_features: int
    out_features: int
    weight_shape: tuple[int, ...]
    has_bias: bool
    param_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)


def iter_linear_layers(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    """Yield ``(name, module)`` for every ``nn.Linear`` in ``model``.

    Args:
        model: Root module to traverse.

    Yields:
        Qualified name and Linear module pairs.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def list_linear_layers(model: nn.Module) -> list[LinearLayerInfo]:
    """Build a structured inventory of all ``nn.Linear`` layers.

    Args:
        model: Root module to traverse.

    Returns:
        List of :class:`LinearLayerInfo` in traversal order.
    """
    inventory: list[LinearLayerInfo] = []
    for name, module in iter_linear_layers(model):
        weight_shape = tuple(module.weight.shape)
        param_count = module.weight.numel()
        has_bias = module.bias is not None
        if has_bias:
            param_count += module.bias.numel()
        inventory.append(
            LinearLayerInfo(
                name=name,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                weight_shape=weight_shape,
                has_bias=has_bias,
                param_count=int(param_count),
            )
        )
    return inventory


def get_transformer_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Locate decoder / encoder block containers across common HF layouts.

    Supports OPT (``model.decoder.layers``), LLaMA/Pythia-style
    (``model.layers``), and GPT-2 (``transformer.h``).

    Args:
        model: Hugging Face causal LM (or inner backbone).

    Returns:
        List of ``(qualified_name, block_module)`` pairs. Empty if unknown.
    """
    candidates: list[tuple[str, Any]] = [
        ("model.decoder.layers", _safe_getattr(model, "model", "decoder", "layers")),
        ("model.layers", _safe_getattr(model, "model", "layers")),
        ("transformer.h", _safe_getattr(model, "transformer", "h")),
        ("gpt_neox.layers", _safe_getattr(model, "gpt_neox", "layers")),
    ]
    for qualified, layers in candidates:
        if layers is None:
            continue
        try:
            return [(f"{qualified}.{idx}", block) for idx, block in enumerate(layers)]
        except TypeError:
            continue
    return []


def _safe_getattr(obj: Any, *names: str) -> Any:
    """Traverse nested attributes, returning ``None`` on the first miss."""
    current = obj
    for name in names:
        if current is None or not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def count_attention_and_mlp_linears(
    linear_layers: list[LinearLayerInfo],
) -> dict[str, int]:
    """Heuristic counts of attention vs MLP Linear projections.

    Name-based heuristics cover OPT / GPT-2 / LLaMA / Pythia naming.

    Args:
        linear_layers: Inventory from :func:`list_linear_layers`.

    Returns:
        Dictionary with ``attention``, ``mlp``, and ``other`` counts.
    """
    attention_keys = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "query",
        "key",
        "value",
        "c_attn",
        "c_proj",  # GPT-2 attn output; also used in MLP — disambiguate below
    )
    mlp_keys = (
        "fc1",
        "fc2",
        "gate_proj",
        "up_proj",
        "down_proj",
        "dense_h_to_4h",
        "dense_4h_to_h",
        "c_fc",
    )

    attention = 0
    mlp = 0
    other = 0

    for info in linear_layers:
        name = info.name.lower()
        leaf = name.rsplit(".", 1)[-1]

        # GPT-2: attention uses c_attn / c_proj under attn; MLP uses c_fc / c_proj under mlp.
        if "attn" in name and leaf in {"c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}:
            attention += 1
        elif leaf in mlp_keys or ("mlp" in name and leaf in {"c_fc", "c_proj"}):
            mlp += 1
        elif leaf in attention_keys or any(k in name for k in ("self_attn", "attention")):
            # Avoid counting lm_head / embed as attention.
            if leaf in {"lm_head", "embed_out"}:
                other += 1
            else:
                attention += 1
        elif any(k in name for k in ("mlp", "fc", "ffn")):
            mlp += 1
        else:
            other += 1

    return {"attention": attention, "mlp": mlp, "other": other}


def collect_model_analysis(
    model: nn.Module,
    *,
    model_name: str = "",
) -> dict[str, Any]:
    """Assemble a full architecture analysis dictionary.

    Args:
        model: Loaded model.
        model_name: Optional HF id for the report.

    Returns:
        Structured analysis including Linear inventory and counts.
    """
    blocks = get_transformer_blocks(model)
    linears = list_linear_layers(model)
    role_counts = count_attention_and_mlp_linears(linears)
    param_counts = count_parameters(model)
    size_bytes = estimate_model_size_bytes(model)

    return {
        "model_name": model_name or getattr(model.config, "_name_or_path", model.__class__.__name__),
        "architecture": model.__class__.__name__,
        "num_transformer_blocks": len(blocks),
        "num_attention_linear_layers": role_counts["attention"],
        "num_mlp_linear_layers": role_counts["mlp"],
        "num_other_linear_layers": role_counts["other"],
        "num_linear_layers": len(linears),
        "total_parameters": param_counts["total"],
        "trainable_parameters": param_counts["trainable"],
        "total_parameters_human": format_params(param_counts["total"]),
        "trainable_parameters_human": format_params(param_counts["trainable"]),
        "model_size_bytes": size_bytes,
        "model_size": format_bytes(size_bytes),
        "dtype": str(get_model_dtype(model)).replace("torch.", ""),
        "device": str(get_model_device(model)),
        "linear_layers": linears,
        "block_names": [name for name, _ in blocks],
    }


def print_architecture_summary(analysis: dict[str, Any]) -> None:
    """Print high-level architecture statistics.

    Args:
        analysis: Output of :func:`collect_model_analysis`.
    """
    print_section("Model Information")
    print_kv("Model Name", analysis["model_name"])
    print_kv("Architecture", analysis["architecture"])
    print_kv("Transformer Blocks", analysis["num_transformer_blocks"])
    print_kv("Attention Linear Layers", analysis["num_attention_linear_layers"])
    print_kv("MLP Linear Layers", analysis["num_mlp_linear_layers"])
    print_kv("Other Linear Layers", analysis["num_other_linear_layers"])
    print_kv("Total nn.Linear Layers", analysis["num_linear_layers"])
    print_kv(
        "Total Parameters",
        f"{analysis['total_parameters_human']} ({analysis['total_parameters']:,})",
    )
    print_kv(
        "Trainable Parameters",
        f"{analysis['trainable_parameters_human']} ({analysis['trainable_parameters']:,})",
    )
    print_kv("Model Size / Memory Footprint", analysis["model_size"])
    print_kv("Tensor Data Type", analysis["dtype"])
    print_kv("Current Device", analysis["device"])


def print_linear_inventory(
    linear_layers: Optional[list[LinearLayerInfo]] = None,
    *,
    model: Optional[nn.Module] = None,
) -> list[LinearLayerInfo]:
    """Print every ``nn.Linear`` layer with name and shapes.

    Provide either ``linear_layers`` or ``model``.

    Args:
        linear_layers: Precomputed inventory.
        model: Model to inventory when ``linear_layers`` is omitted.

    Returns:
        The inventory that was printed.

    Raises:
        ValueError: If neither argument is provided.
    """
    if linear_layers is None:
        if model is None:
            raise ValueError("Provide linear_layers or model")
        linear_layers = list_linear_layers(model)

    print_section(f"nn.Linear Inventory ({len(linear_layers)} layers)")
    header = f"  {'#':<4} {'Layer Name':<55} {'In':>6} {'Out':>6} {'Weight Shape':<16}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for idx, info in enumerate(linear_layers):
        shape_str = str(tuple(info.weight_shape))
        print(
            f"  {idx:<4} {info.name:<55} "
            f"{info.in_features:>6} {info.out_features:>6} {shape_str:<16}"
        )

    return linear_layers


def print_full_model_analysis(
    model: nn.Module,
    *,
    model_name: str = "",
) -> dict[str, Any]:
    """Run and print the complete Phase 0 model analysis.

    Args:
        model: Loaded model.
        model_name: HF identifier for display.

    Returns:
        Analysis dictionary from :func:`collect_model_analysis`.
    """
    analysis = collect_model_analysis(model, model_name=model_name)
    print_architecture_summary(analysis)
    print_linear_inventory(analysis["linear_layers"])
    return analysis
