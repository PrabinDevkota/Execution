"""Phase 5 orchestrator: Ledoit–Wolf + κ-gated ActSVD and SpectraLite."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, read_json, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, default_config
from spectralite.svd_activation import apply_activation_aware_svd, print_actsvd_summary
from spectralite.svd_spectralite import (
    allocate_and_compress,
    build_whitened_svd_cache,
    print_spectralite_summary,
)
from spectralite.utils import get_logger, print_section
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)


def _budgets_from_phase3(config: Config) -> list[float]:
    try:
        payload = read_json("phase3_summary.json", config)
        keeps = [
            float(s["param_keep_ratio_touched"])
            for s in payload.get("summaries", [])
            if s.get("param_keep_ratio_touched") is not None
        ]
        if keeps:
            return keeps
    except Exception as exc:  # noqa: BLE001
        logger.warning("Phase-3 budgets unavailable (%s); using defaults", exc)
    return [0.75, 0.60, 0.45]


def _profile(
    model: Any,
    tokenizer: Any,
    *,
    cfg: Config,
    method: str,
    notes: str,
    csv_name: str,
    analytic_keep: float,
    ppl_seq_len: int,
    ppl_max_tokens: int,
    latency_reps_prefill: int,
    latency_reps_decode: int,
) -> dict[str, Any]:
    return run_phase1_dense_baseline(
        model,
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
        phase="5",
        method=method,
        notes=notes,
        persist_artifacts=False,
        analytic_flops_fwd_ratio=float(analytic_keep),
    )


def run_phase5_stability_sweep(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    rank_ratios: Sequence[float] = (0.5, 0.4, 0.3),
    target_keep_ratios: Optional[Sequence[float]] = None,
    calib_num_sequences: int = 32,
    calib_seq_len: int = 512,
    calib_batch_size: int = 2,
    kappa_max: float = 1e4,
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 30_000,
    latency_reps_prefill: int = 30,
    latency_reps_decode: int = 20,
    csv_name: str = "phase5_stability.csv",
    run_actsvd: bool = True,
    run_spectralite: bool = True,
) -> dict[str, Any]:
    """Phase 5: Ledoit–Wolf whitening + κ gating on ActSVD and SpectraLite.

    Compares against Phase 3 (ridge ActSVD) and Phase 4 (ridge SpectraLite).
    """
    cfg = config or default_config()
    cfg.ensure_directories()
    budgets = list(target_keep_ratios) if target_keep_ratios is not None else _budgets_from_phase3(cfg)

    print_section("Phase 5 — Calibration (WikiText-2)")
    batches = load_wikitext2_calibration_batches(
        tokenizer,
        num_sequences=calib_num_sequences,
        seq_len=calib_seq_len,
        batch_size=calib_batch_size,
        seed=cfg.seed,
    )
    print_section("Phase 5 — Collect activations")
    activations = collect_linear_input_activations(dense_model, batches)

    all_rows: list[dict[str, Any]] = []
    act_summaries: list[dict[str, Any]] = []
    spec_summaries: list[dict[str, Any]] = []

    if run_actsvd:
        print_section("Phase 5 — ActSVD + Ledoit–Wolf + κ gate")
        for ratio in rank_ratios:
            packed = apply_activation_aware_svd(
                dense_model,
                activations,
                rank_ratio=float(ratio),
                cov_method="ledoit_wolf",
                kappa_max=kappa_max,
                clone=True,
            )
            print_actsvd_summary(packed["summary"])
            metrics = _profile(
                packed["model"],
                tokenizer,
                cfg=cfg,
                method=f"actsvd_lw_k{ratio:.2f}",
                notes=(
                    f"phase5 ActSVD+LW+kappa ratio={ratio} kappa_max={kappa_max}; "
                    f"bumped={packed['summary'].get('num_kappa_bumped')}; "
                    f"mean_recon={packed['summary'].get('mean_recon_rel_error'):.4g}; "
                    f"keep={packed['summary']['param_keep_ratio_touched']:.4f}"
                ),
                csv_name=csv_name,
                analytic_keep=packed["summary"]["param_keep_ratio_touched"],
                ppl_seq_len=ppl_seq_len,
                ppl_max_tokens=ppl_max_tokens,
                latency_reps_prefill=latency_reps_prefill,
                latency_reps_decode=latency_reps_decode,
            )
            all_rows.append(metrics["row"])
            act_summaries.append(
                {
                    "rank_ratio": ratio,
                    "param_keep_ratio_touched": packed["summary"]["param_keep_ratio_touched"],
                    "mean_recon_rel_error": packed["summary"].get("mean_recon_rel_error"),
                    "num_kappa_bumped": packed["summary"].get("num_kappa_bumped"),
                    "replacements_preview": packed["summary"]["replacements"][:3],
                }
            )
            del packed, metrics
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    if run_spectralite:
        print_section("Phase 5 — SpectraLite + Ledoit–Wolf + κ gate")
        cache = build_whitened_svd_cache(
            dense_model,
            activations,
            cov_method="ledoit_wolf",
        )
        write_json(
            "phase5_spectrum_meta.json",
            {
                "cov_method": "ledoit_wolf",
                "kappa_max": kappa_max,
                "layers": {
                    name: {
                        "q": int(e["q"]),
                        "rho_eff": float(e["rho_eff"]),
                        "protect": float(e["protect"]),
                        "kappa_cov": float(e.get("kappa_cov", float("nan"))),
                    }
                    for name, e in cache.items()
                },
            },
        )
        for keep in budgets:
            packed = allocate_and_compress(
                dense_model,
                cache,
                float(keep),
                clone=True,
                kappa_max=kappa_max,
            )
            print_spectralite_summary(packed["summary"], packed["allocation"])
            metrics = _profile(
                packed["model"],
                tokenizer,
                cfg=cfg,
                method=f"spectralite_lw_k{keep:.2f}",
                notes=(
                    f"phase5 SpectraLite+LW+kappa keep_target={keep:.4f} "
                    f"achieved={packed['summary']['param_keep_ratio_touched']:.4f} "
                    f"kappa_max={kappa_max}; bumped={packed['summary'].get('num_kappa_bumped')}; "
                    f"mean_recon={packed['summary'].get('mean_recon_rel_error'):.4g}"
                ),
                csv_name=csv_name,
                analytic_keep=packed["summary"]["param_keep_ratio_touched"],
                ppl_seq_len=ppl_seq_len,
                ppl_max_tokens=ppl_max_tokens,
                latency_reps_prefill=latency_reps_prefill,
                latency_reps_decode=latency_reps_decode,
            )
            all_rows.append(metrics["row"])
            spec_summaries.append(
                {
                    "target_keep_ratio": keep,
                    "achieved_keep_ratio": packed["summary"]["param_keep_ratio_touched"],
                    "lambda": packed["allocation"]["lambda"],
                    "mean_recon_rel_error": packed["summary"].get("mean_recon_rel_error"),
                    "num_kappa_bumped": packed["summary"].get("num_kappa_bumped"),
                    "replacements_preview": packed["summary"]["replacements"][:3],
                }
            )
            del packed, metrics
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    payload = {
        "phase": "5",
        "kappa_max": kappa_max,
        "cov_method": "ledoit_wolf",
        "rank_ratios": list(rank_ratios),
        "target_keep_ratios": budgets,
        "calib_num_sequences": calib_num_sequences,
        "calib_seq_len": calib_seq_len,
        "rows": all_rows,
        "actsvd_lw_summaries": act_summaries,
        "spectralite_lw_summaries": spec_summaries,
        "csv": f"results/{csv_name}",
        "references": {
            "phase3_c4": [122.61, 555.45, 2286.51],
            "phase4_c4": [4798.62, 7257.75, 5780.87],
        },
    }
    write_json("phase5_stability_details.json", {
        "actsvd_lw_summaries": act_summaries,
        "spectralite_lw_summaries": spec_summaries,
        "kappa_max": kappa_max,
    })
    write_json("phase5_summary.json", payload)
    mark_phase_complete(
        "5",
        artifacts={
            "summary": "results/phase5_summary.json",
            "details": "results/phase5_stability_details.json",
            "spectrum": "results/phase5_spectrum_meta.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "kappa_max": kappa_max,
            "num_settings": len(all_rows),
            "last_method": all_rows[-1]["method"] if all_rows else None,
            "last_ppl_c4": all_rows[-1].get("ppl_c4") if all_rows else None,
        },
        notes=(
            "Ledoit–Wolf covariance + κ truncation gate on ActSVD and SpectraLite; "
            "logs recon error / κ bumps vs Phase 3–4."
        ),
        config=cfg,
    )
    print_git_save_instructions()
    return payload
