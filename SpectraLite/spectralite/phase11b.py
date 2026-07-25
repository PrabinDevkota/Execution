"""Phase 11b: LLaMA-3.2-1B zero-shot only (no PPL/latency remeasure)."""

from __future__ import annotations

import gc
from typing import Any, Optional, Sequence

import torch

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, config_for_model
from spectralite.downstream import DEFAULT_ZERO_SHOT_TASKS, run_lm_eval
from spectralite.model_loader import load_model_and_tokenizer
from spectralite.svd_activation import apply_activation_aware_svd
from spectralite.svd_spectralite import allocate_and_compress, build_whitened_svd_cache
from spectralite.utils import get_logger, print_kv, print_section, set_seed
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_phase11b_llama32_1b_zeroshot(
    dense_model: Any = None,
    tokenizer: Any = None,
    *,
    config: Optional[Config] = None,
    model_name: str = DEFAULT_MODEL,
    keep_ratio: float = 0.75,
    rank_ratio: float = 0.5,
    kappa_speed: float = 1.0,
    protect_mode: str = "rho",
    calib_num_sequences: int = 16,
    calib_seq_len: int = 256,
    calib_batch_size: int = 1,
    max_tokens_per_layer: int = 4096,
    ridge: float = 1e-2,
    tasks: Sequence[str] = DEFAULT_ZERO_SHOT_TASKS,
    batch_size: int | str = 2,
    limit: Optional[int | float] = None,
    eval_dense: bool = True,
    eval_actsvd_gated: bool = True,
    eval_spectralite_gated: bool = True,
) -> dict[str, Any]:
    """Zero-shot lm-eval on dense / ActSVD+gate / Spec-ρ+gate for LLaMA-3.2-1B.

    Skips perplexity and latency (already in Phase 11). Order is RAM-friendly:
    dense → calib → ActSVD+gate → consume acts into Spec cache → Spec-ρ+gate.
    """
    cfg = config or config_for_model(model_name)
    cfg.model_name = model_name
    cfg.ensure_directories()
    set_seed(cfg.seed)

    if dense_model is None or tokenizer is None:
        print_section(f"Phase 11b — Load {model_name}")
        dense_model, tokenizer = load_model_and_tokenizer(config=cfg)
    dense_model.eval()
    _free()

    rows: list[dict[str, Any]] = []

    def _eval(model: Any, method: str, notes: str) -> None:
        out = run_lm_eval(
            model,
            tokenizer,
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
            limit=limit,
            method=method,
        )
        out["notes"] = notes
        rows.append(out)
        write_json(f"phase11b_{method}.json", out)
        print_kv(method, f"avg={out.get('zero_shot_avg')}")

    if eval_dense:
        print_section("Phase 11b — Zero-shot dense")
        _eval(dense_model, "dense", f"Phase 11b dense {model_name}")
        _free()

    need_acts = eval_actsvd_gated or eval_spectralite_gated
    activations = None
    if need_acts:
        print_section(
            f"Phase 11b — Calibration (seqs={calib_num_sequences}, "
            f"len={calib_seq_len}, max_tok/layer={max_tokens_per_layer})"
        )
        batches = load_wikitext2_calibration_batches(
            tokenizer,
            num_sequences=calib_num_sequences,
            seq_len=calib_seq_len,
            batch_size=calib_batch_size,
            seed=cfg.seed,
        )
        activations = collect_linear_input_activations(
            dense_model,
            batches,
            max_tokens_per_layer=max_tokens_per_layer,
        )
        del batches
        _free()

    if eval_actsvd_gated:
        assert activations is not None
        print_section("Phase 11b — Zero-shot ActSVD + gate")
        packed = apply_activation_aware_svd(
            dense_model,
            activations,
            rank_ratio=rank_ratio,
            ridge=ridge,
            cov_method="ridge",
            latency_gate=True,
            kappa_speed=kappa_speed,
            clone=True,
        )
        _eval(
            packed["model"],
            "actsvd_gate_r0.50",
            (
                f"Phase 11b ActSVD+gate ratio={rank_ratio} kappa_speed={kappa_speed} "
                f"replaced={packed['summary']['num_replaced']} "
                f"gated_dense={packed['summary'].get('num_gated_dense', 0)}"
            ),
        )
        del packed
        _free()

    if eval_spectralite_gated:
        assert activations is not None
        print_section("Phase 11b — Whitened spectral cache + Spec-ρ + gate")
        cache = build_whitened_svd_cache(
            dense_model,
            activations,
            ridge=ridge,
            cov_method="ridge",
            factor_dtype=torch.float32,
            store_weight=False,
            consume_activations=True,
        )
        del activations
        _free()
        packed = allocate_and_compress(
            dense_model,
            cache,
            float(keep_ratio),
            clone=True,
            protect_mode=protect_mode,
            latency_gate=True,
            kappa_speed=kappa_speed,
        )
        _eval(
            packed["model"],
            "spectralite_rho_gate_k0.75",
            (
                f"Phase 11b Spec-ρ+gate keep={keep_ratio} kappa_speed={kappa_speed} "
                f"replaced={packed['summary']['num_replaced']} "
                f"gated_dense={packed['summary'].get('num_gated_dense', 0)}"
            ),
        )
        del packed, cache
        _free()
    elif activations is not None:
        del activations
        _free()

    table = []
    for r in rows:
        entry = {"method": r["method"], "zero_shot_avg": r.get("zero_shot_avg")}
        for t, m in (r.get("per_task") or {}).items():
            entry[t] = m.get("acc")
        table.append(entry)

    claim = {
        "model_name": model_name,
        "tasks": list(tasks),
    }
    for entry in table:
        claim[f"{entry['method']}_zero_shot_avg"] = entry.get("zero_shot_avg")

    print_section("Phase 11b — Claim snapshot")
    for k, v in claim.items():
        print_kv(k, v)

    payload = {
        "phase": "10b",
        "model_name": model_name,
        "keep_ratio": keep_ratio,
        "rank_ratio": rank_ratio,
        "kappa_speed": kappa_speed,
        "tasks": list(tasks),
        "batch_size": batch_size,
        "limit": limit,
        "rows": rows,
        "table": table,
        "claim": claim,
    }
    write_json("phase11b_summary.json", payload)
    write_json("phase11b_table.json", {"table": table, "tasks": list(tasks)})
    write_json("phase11b_claim.json", claim)

    best = None
    if table:
        best = max(table, key=lambda x: (x.get("zero_shot_avg") or float("-inf")))

    mark_phase_complete(
        "10b",
        artifacts={
            "summary": "results/phase11b_summary.json",
            "table": "results/phase11b_table.json",
            "claim": "results/phase11b_claim.json",
            "status": "results/phase_status.json",
        },
        metrics={
            "model_name": model_name,
            "num_methods": len(table),
            "best_method": best.get("method") if best else None,
            "best_zero_shot_avg": best.get("zero_shot_avg") if best else None,
            "dense_avg": next(
                (t.get("zero_shot_avg") for t in table if t["method"] == "dense"), None
            ),
        },
        notes=(
            f"LLaMA-3.2-1B zero-shot only (dense / ActSVD+gate / Spec-ρ+gate) "
            f"at keep≈{keep_ratio}; complements Phase 11 PPL/latency."
        ),
        config=cfg,
    )
    print_git_save_instructions()
    return payload
