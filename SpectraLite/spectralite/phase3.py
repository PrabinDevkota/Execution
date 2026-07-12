"""Phase 3 orchestrator: activation-aware SVD sweep → metrics → artifacts."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, default_config
from spectralite.svd_activation import apply_activation_aware_svd, print_actsvd_summary
from spectralite.utils import get_logger, print_section
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)


def run_phase3_activation_aware_sweep(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    rank_ratios: Sequence[float] = (0.5, 0.4, 0.3),
    calib_num_sequences: int = 32,
    calib_seq_len: int = 512,
    calib_batch_size: int = 2,
    ridge: float = 1e-2,
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 30_000,
    latency_reps_prefill: int = 30,
    latency_reps_decode: int = 20,
    csv_name: str = "phase3_activation_aware_svd.csv",
) -> dict[str, Any]:
    """Calibrate on WikiText-2, whiten+SVD at several ratios, profile each.

    Same rank ratios as Phase 2 for a fair comparison. Activations are collected
    once on the dense model and reused across ratios (covariance uses XᵀX/n + λI).
    """
    cfg = config or default_config()
    cfg.ensure_directories()

    print_section("Phase 3 — Calibration (WikiText-2)")
    batches = load_wikitext2_calibration_batches(
        tokenizer,
        num_sequences=calib_num_sequences,
        seq_len=calib_seq_len,
        batch_size=calib_batch_size,
        seed=cfg.seed,
    )

    print_section("Phase 3 — Collecting input activations")
    activations = collect_linear_input_activations(dense_model, batches)
    write_json(
        "phase3_calibration_meta.json",
        {
            "num_sequences": calib_num_sequences,
            "seq_len": calib_seq_len,
            "batch_size": calib_batch_size,
            "ridge": ridge,
            "layers_with_acts": {k: int(v.shape[0]) for k, v in activations.items()},
        },
    )

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    print_section("Phase 3 — Activation-aware SVD sweep")
    for ratio in rank_ratios:
        logger.info("ActSVD rank_ratio=%.2f", ratio)
        packed = apply_activation_aware_svd(
            dense_model,
            activations,
            rank_ratio=float(ratio),
            ridge=ridge,
            clone=True,
        )
        compressed = packed["model"]
        summary = packed["summary"]
        print_actsvd_summary(summary)

        metrics = run_phase1_dense_baseline(
            compressed,
            tokenizer,
            config=cfg,
            prompt_len=getattr(cfg, "latency_prompt_len", 128),
            gen_tokens=getattr(cfg, "latency_gen_tokens", 64),
            latency_warmup=getattr(cfg, "latency_warmup", 10),
            latency_reps_prefill=latency_reps_prefill,
            latency_reps_decode=latency_reps_decode,
            ppl_seq_len=ppl_seq_len,
            ppl_max_tokens=ppl_max_tokens,
            run_calflops=False,
            run_ppl=True,
            csv_name=csv_name,
            phase="3",
            method=f"actsvd_r{ratio:.2f}",
            notes=(
                f"activation_aware_svd ratio={ratio} ridge={ridge}; "
                f"calib={calib_num_sequences}x{calib_seq_len}; "
                f"replaced={summary['num_replaced']}; "
                f"param_keep={summary['param_keep_ratio_touched']:.4f}"
            ),
            persist_artifacts=False,
            analytic_flops_fwd_ratio=float(summary["param_keep_ratio_touched"]),
        )
        all_rows.append(metrics["row"])
        all_summaries.append(summary)

        del compressed, packed, metrics
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    compact = []
    for s in all_summaries:
        compact.append(
            {
                "rank_ratio": s["rank_ratio"],
                "ridge": s["ridge"],
                "num_replaced": s["num_replaced"],
                "num_skipped": s["num_skipped"],
                "param_keep_ratio_touched": s["param_keep_ratio_touched"],
                "params_saved_touched": s["params_saved_touched"],
                "replacements_preview": s["replacements"][:3],
            }
        )

    payload = {
        "phase": "3",
        "rank_ratios": list(rank_ratios),
        "ridge": ridge,
        "calib_num_sequences": calib_num_sequences,
        "calib_seq_len": calib_seq_len,
        "rows": all_rows,
        "summaries": compact,
        "csv": f"results/{csv_name}",
    }
    write_json(
        "phase3_replacements.json",
        {"rank_ratios": list(rank_ratios), "summaries": all_summaries},
    )
    write_json("phase3_summary.json", payload)
    mark_phase_complete(
        "3",
        artifacts={
            "summary": "results/phase3_summary.json",
            "replacements": "results/phase3_replacements.json",
            "calibration": "results/phase3_calibration_meta.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "rank_ratios": list(rank_ratios),
            "num_settings": len(all_rows),
            "last_method": all_rows[-1]["method"] if all_rows else None,
            "last_ppl_c4": all_rows[-1].get("ppl_c4") if all_rows else None,
            "last_prefill_ms": all_rows[-1].get("prefill_ms_mean") if all_rows else None,
        },
        notes="Activation-whitened truncated SVD (SVD-LLM/ASVD-style Cholesky + ridge).",
        config=cfg,
    )
    print_git_save_instructions()
    return payload
