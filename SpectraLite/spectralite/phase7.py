"""Phase 7 orchestrator: publishable ablations at matched keep / ratio."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, default_config
from spectralite.svd_activation import apply_activation_aware_svd, print_actsvd_summary
from spectralite.svd_spectralite import allocate_and_compress, build_whitened_svd_cache, print_spectralite_summary
from spectralite.svd_vanilla import (
    ATTN_COMPRESS_SUFFIXES,
    DEFAULT_COMPRESS_SUFFIXES,
    MLP_COMPRESS_SUFFIXES,
    apply_vanilla_svd,
    print_svd_summary,
)
from spectralite.utils import get_logger, print_section
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)


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
        phase="7",
        method=method,
        notes=notes,
        persist_artifacts=False,
        analytic_flops_fwd_ratio=float(analytic_keep),
    )


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_phase7_ablation_suite(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    keep_ratio: float = 0.75,
    rank_ratio: float = 0.5,
    calib_num_sequences: int = 32,
    calib_seq_len: int = 512,
    calib_batch_size: int = 2,
    ridge: float = 1e-2,
    kappa_speed: float = 1.0,
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 30_000,
    latency_reps_prefill: int = 30,
    latency_reps_decode: int = 20,
    csv_name: str = "phase7_ablations.csv",
) -> dict[str, Any]:
    """Run core Phase-7 ablations at one matched operating point.

    Ablations (plan §7, OPT-125M Colab-sized):
      1. Vanilla SVD vs ActSVD (whitening)
      2. ActSVD ± latency gate
      3. ActSVD ± Ledoit–Wolf
      4. ActSVD attention-only vs MLP-only
      5. SpectraLite protect: full / rho-only / stable-rank-only
      6. Smaller calibration (16 seq) ActSVD sanity check
    """
    cfg = config or default_config()
    cfg.ensure_directories()

    print_section("Phase 7 — Calibration ×32 (main)")
    batches32 = load_wikitext2_calibration_batches(
        tokenizer,
        num_sequences=calib_num_sequences,
        seq_len=calib_seq_len,
        batch_size=calib_batch_size,
        seed=cfg.seed,
    )
    acts32 = collect_linear_input_activations(dense_model, batches32)

    print_section("Phase 7 — Calibration ×16 (robustness)")
    batches16 = load_wikitext2_calibration_batches(
        tokenizer,
        num_sequences=min(16, calib_num_sequences),
        seq_len=calib_seq_len,
        batch_size=calib_batch_size,
        seed=cfg.seed,
    )
    acts16 = collect_linear_input_activations(dense_model, batches16)

    print_section("Phase 7 — Whitened spectrum cache (ridge)")
    cache = build_whitened_svd_cache(dense_model, acts32, ridge=ridge, cov_method="ridge")

    abl_specs: list[dict[str, Any]] = [
        {"id": "vanilla_r0.50", "family": "vanilla", "claim": "whitening_on_off"},
        {
            "id": "actsvd_r0.50",
            "family": "actsvd",
            "cov": "ridge",
            "gate": False,
            "suffixes": "all",
            "acts": "32",
            "claim": "reference",
        },
        {
            "id": "actsvd_gate_r0.50",
            "family": "actsvd",
            "cov": "ridge",
            "gate": True,
            "suffixes": "all",
            "acts": "32",
            "claim": "latency_gate",
        },
        {
            "id": "actsvd_lw_r0.50",
            "family": "actsvd",
            "cov": "ledoit_wolf",
            "gate": False,
            "suffixes": "all",
            "acts": "32",
            "claim": "ledoit_wolf",
        },
        {
            "id": "actsvd_attn_r0.50",
            "family": "actsvd",
            "cov": "ridge",
            "gate": False,
            "suffixes": "attn",
            "acts": "32",
            "claim": "attn_vs_mlp",
        },
        {
            "id": "actsvd_mlp_r0.50",
            "family": "actsvd",
            "cov": "ridge",
            "gate": False,
            "suffixes": "mlp",
            "acts": "32",
            "claim": "attn_vs_mlp",
        },
        {
            "id": "actsvd_calib16_r0.50",
            "family": "actsvd",
            "cov": "ridge",
            "gate": False,
            "suffixes": "all",
            "acts": "16",
            "claim": "calib_size",
        },
        {
            "id": "spec_full_k0.75",
            "family": "spectralite",
            "protect": "full",
            "claim": "spectral_vs_uniform",
        },
        {
            "id": "spec_rho_k0.75",
            "family": "spectralite",
            "protect": "rho",
            "claim": "sensitivity_weighting",
        },
        {
            "id": "spec_sr_k0.75",
            "family": "spectralite",
            "protect": "stable_rank",
            "claim": "sensitivity_weighting",
        },
    ]

    all_rows: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []

    print_section("Phase 7 — Ablation sweep")
    for spec in abl_specs:
        aid = spec["id"]
        logger.info("Ablation %s (%s)", aid, spec.get("claim"))
        family = spec["family"]

        if family == "vanilla":
            packed = apply_vanilla_svd(dense_model, rank_ratio=rank_ratio, clone=True)
            print_svd_summary(packed["summary"])
            summary = packed["summary"]
            model_c = packed["model"]
            notes = f"phase7 vanilla SVD ratio={rank_ratio}"
        elif family == "actsvd":
            suffixes_key = spec.get("suffixes", "all")
            if suffixes_key == "attn":
                suffixes: Sequence[str] = ATTN_COMPRESS_SUFFIXES
            elif suffixes_key == "mlp":
                suffixes = MLP_COMPRESS_SUFFIXES
            else:
                suffixes = DEFAULT_COMPRESS_SUFFIXES
            acts = acts16 if spec.get("acts") == "16" else acts32
            packed = apply_activation_aware_svd(
                dense_model,
                acts,
                rank_ratio=rank_ratio,
                ridge=ridge,
                cov_method=str(spec.get("cov", "ridge")),
                latency_gate=bool(spec.get("gate", False)),
                kappa_speed=kappa_speed,
                suffixes=suffixes,
                clone=True,
            )
            print_actsvd_summary(packed["summary"])
            summary = packed["summary"]
            model_c = packed["model"]
            notes = (
                f"phase7 ActSVD ratio={rank_ratio} cov={spec.get('cov')} "
                f"gate={spec.get('gate')} suffixes={suffixes_key} "
                f"calib={spec.get('acts')} replaced={summary['num_replaced']} "
                f"gated_dense={summary.get('num_gated_dense', 0)}"
            )
        else:  # spectralite
            protect = str(spec.get("protect", "full"))
            packed = allocate_and_compress(
                dense_model,
                cache,
                float(keep_ratio),
                clone=True,
                protect_mode=protect,
            )
            print_spectralite_summary(packed["summary"], packed["allocation"])
            summary = packed["summary"]
            model_c = packed["model"]
            notes = (
                f"phase7 SpectraLite keep={keep_ratio} protect={protect} "
                f"achieved={summary['param_keep_ratio_touched']:.4f} "
                f"lambda={summary.get('lambda')}"
            )

        metrics = _profile(
            model_c,
            tokenizer,
            cfg=cfg,
            method=aid,
            notes=notes,
            csv_name=csv_name,
            analytic_keep=float(summary["param_keep_ratio_touched"]),
            ppl_seq_len=ppl_seq_len,
            ppl_max_tokens=ppl_max_tokens,
            latency_reps_prefill=latency_reps_prefill,
            latency_reps_decode=latency_reps_decode,
        )
        row = metrics["row"]
        all_rows.append(row)
        compact.append(
            {
                "id": aid,
                "claim": spec.get("claim"),
                "family": family,
                "keep": row.get("analytic_flops_fwd_ratio"),
                "prefill_ms": row.get("prefill_ms_mean"),
                "decode_ms": row.get("decode_ms_per_token_mean"),
                "ppl_c4": row.get("ppl_c4"),
                "ppl_wikitext2": row.get("ppl_wikitext2"),
                "num_replaced": summary.get("num_replaced"),
                "num_gated_dense": summary.get("num_gated_dense"),
                "protect_mode": summary.get("protect_mode"),
            }
        )
        del packed, model_c, metrics
        _empty_cache()

    # Mini claim table for the paper notebook output
    by_id = {c["id"]: c for c in compact}
    claim_table = {
        "whitening_on_off": {
            "vanilla_c4": by_id.get("vanilla_r0.50", {}).get("ppl_c4"),
            "actsvd_c4": by_id.get("actsvd_r0.50", {}).get("ppl_c4"),
        },
        "latency_gate": {
            "ungated_decode_ms": by_id.get("actsvd_r0.50", {}).get("decode_ms"),
            "gated_decode_ms": by_id.get("actsvd_gate_r0.50", {}).get("decode_ms"),
            "ungated_c4": by_id.get("actsvd_r0.50", {}).get("ppl_c4"),
            "gated_c4": by_id.get("actsvd_gate_r0.50", {}).get("ppl_c4"),
        },
        "ledoit_wolf": {
            "ridge_c4": by_id.get("actsvd_r0.50", {}).get("ppl_c4"),
            "lw_c4": by_id.get("actsvd_lw_r0.50", {}).get("ppl_c4"),
        },
        "attn_vs_mlp": {
            "attn_c4": by_id.get("actsvd_attn_r0.50", {}).get("ppl_c4"),
            "mlp_c4": by_id.get("actsvd_mlp_r0.50", {}).get("ppl_c4"),
            "attn_decode_ms": by_id.get("actsvd_attn_r0.50", {}).get("decode_ms"),
            "mlp_decode_ms": by_id.get("actsvd_mlp_r0.50", {}).get("decode_ms"),
        },
        "spectral_alloc": {
            "actsvd_c4": by_id.get("actsvd_r0.50", {}).get("ppl_c4"),
            "spec_full_c4": by_id.get("spec_full_k0.75", {}).get("ppl_c4"),
            "spec_rho_c4": by_id.get("spec_rho_k0.75", {}).get("ppl_c4"),
            "spec_sr_c4": by_id.get("spec_sr_k0.75", {}).get("ppl_c4"),
        },
        "calib_size": {
            "calib32_c4": by_id.get("actsvd_r0.50", {}).get("ppl_c4"),
            "calib16_c4": by_id.get("actsvd_calib16_r0.50", {}).get("ppl_c4"),
        },
    }

    payload = {
        "phase": "7",
        "keep_ratio": keep_ratio,
        "rank_ratio": rank_ratio,
        "calib_num_sequences": calib_num_sequences,
        "rows": all_rows,
        "ablations": compact,
        "claim_table": claim_table,
        "csv": f"results/{csv_name}",
        "references": {
            "phase3_actsvd_r0.50_c4": 122.61,
            "phase6_gated_r0.50_decode_ms": 8.32,
        },
    }
    write_json("phase7_claim_table.json", claim_table)
    write_json("phase7_summary.json", payload)
    mark_phase_complete(
        "7",
        artifacts={
            "summary": "results/phase7_summary.json",
            "claim_table": "results/phase7_claim_table.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "num_ablations": len(compact),
            "keep_ratio": keep_ratio,
            "rank_ratio": rank_ratio,
            "best_c4_id": min(compact, key=lambda c: c.get("ppl_c4") or 1e18).get("id")
            if compact
            else None,
            "best_c4": min((c.get("ppl_c4") or 1e18) for c in compact) if compact else None,
        },
        notes="Phase 7 ablations: whitening, gate, LW, attn/MLP, SpectraLite protect modes, calib size.",
        config=cfg,
    )
    print_git_save_instructions()
    return payload
