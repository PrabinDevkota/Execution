"""Phase 8 orchestrator: lm-eval zero-shot on dense + best compressors."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, default_config
from spectralite.downstream import DEFAULT_ZERO_SHOT_TASKS, run_lm_eval
from spectralite.svd_activation import apply_activation_aware_svd
from spectralite.svd_spectralite import allocate_and_compress, build_whitened_svd_cache
from spectralite.utils import get_logger, print_section
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_phase8_downstream_eval(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    tasks: Sequence[str] = DEFAULT_ZERO_SHOT_TASKS,
    num_fewshot: int = 0,
    batch_size: int | str = 8,
    limit: Optional[int | float] = None,
    calib_num_sequences: int = 32,
    calib_seq_len: int = 512,
    calib_batch_size: int = 2,
    ridge: float = 1e-2,
    rank_ratio: float = 0.5,
    keep_ratio: float = 0.75,
    kappa_speed: float = 1.0,
    eval_dense: bool = True,
    eval_actsvd: bool = True,
    eval_actsvd_gated: bool = True,
    eval_spectralite_rho: bool = True,
) -> dict[str, Any]:
    """Zero-shot lm-eval on dense + Phase-3/4/6 headline compressors.

    Default operating point: ratio 0.5 / keep ≈ 0.75 (best-studied setting).
    """
    cfg = config or default_config()
    cfg.ensure_directories()

    rows: list[dict[str, Any]] = []

    need_acts = eval_actsvd or eval_actsvd_gated or eval_spectralite_rho
    activations = None
    cache = None
    if need_acts:
        print_section("Phase 8 — Calibration for compressed variants")
        batches = load_wikitext2_calibration_batches(
            tokenizer,
            num_sequences=calib_num_sequences,
            seq_len=calib_seq_len,
            batch_size=calib_batch_size,
            seed=cfg.seed,
        )
        activations = collect_linear_input_activations(dense_model, batches)
        if eval_spectralite_rho:
            cache = build_whitened_svd_cache(
                dense_model, activations, ridge=ridge, cov_method="ridge"
            )

    def _eval(model: Any, method: str, notes: str) -> None:
        out = run_lm_eval(
            model,
            tokenizer,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            method=method,
        )
        out["notes"] = notes
        rows.append(out)
        write_json(f"phase8_{method}.json", out)

    if eval_dense:
        _eval(dense_model, "dense", "Phase 8 dense OPT-125M baseline")

    if eval_actsvd:
        assert activations is not None
        packed = apply_activation_aware_svd(
            dense_model,
            activations,
            rank_ratio=rank_ratio,
            ridge=ridge,
            cov_method="ridge",
            latency_gate=False,
            clone=True,
        )
        _eval(
            packed["model"],
            "actsvd_r0.50",
            f"ridge ActSVD ungated ratio={rank_ratio} keep={packed['summary']['param_keep_ratio_touched']:.4f}",
        )
        del packed
        _empty_cache()

    if eval_actsvd_gated:
        assert activations is not None
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
                f"ridge ActSVD gated ratio={rank_ratio} kappa_speed={kappa_speed} "
                f"replaced={packed['summary']['num_replaced']} "
                f"gated_dense={packed['summary'].get('num_gated_dense', 0)}"
            ),
        )
        del packed
        _empty_cache()

    if eval_spectralite_rho:
        assert cache is not None
        packed = allocate_and_compress(
            dense_model,
            cache,
            float(keep_ratio),
            clone=True,
            protect_mode="rho",
        )
        _eval(
            packed["model"],
            "spectralite_rho_k0.75",
            (
                f"SpectraLite protect=rho keep_target={keep_ratio} "
                f"achieved={packed['summary']['param_keep_ratio_touched']:.4f}"
            ),
        )
        del packed
        _empty_cache()

    # Compact table
    table = []
    for r in rows:
        entry = {"method": r["method"], "zero_shot_avg": r.get("zero_shot_avg")}
        for t, m in (r.get("per_task") or {}).items():
            entry[t] = m.get("acc")
        table.append(entry)

    payload = {
        "phase": "8",
        "model_name": cfg.model_name,
        "tasks": list(tasks),
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "limit": limit,
        "rank_ratio": rank_ratio,
        "keep_ratio": keep_ratio,
        "rows": rows,
        "table": table,
    }
    write_json("phase8_summary.json", payload)
    write_json("phase8_table.json", {"table": table, "tasks": list(tasks)})

    best = None
    if table:
        best = max(table, key=lambda x: (x.get("zero_shot_avg") or float("-inf")))

    mark_phase_complete(
        "8",
        artifacts={
            "summary": "results/phase8_summary.json",
            "table": "results/phase8_table.json",
            "status": "results/phase_status.json",
        },
        metrics={
            "num_methods": len(table),
            "tasks": list(tasks),
            "best_method": best.get("method") if best else None,
            "best_zero_shot_avg": best.get("zero_shot_avg") if best else None,
            "dense_avg": next((t.get("zero_shot_avg") for t in table if t["method"] == "dense"), None),
        },
        notes=(
            "Zero-shot lm-eval on dense, ActSVD, gated ActSVD, SpectraLite-rho "
            f"at ratio={rank_ratio}/keep={keep_ratio}."
        ),
        config=cfg,
    )
    print_git_save_instructions()
    return payload
