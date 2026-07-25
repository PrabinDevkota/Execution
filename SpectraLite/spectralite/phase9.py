"""Phase 9: SpectraLite-ρ + latency gate (default deploy config on OPT-125M)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, default_config
from spectralite.downstream import DEFAULT_ZERO_SHOT_TASKS, run_lm_eval
from spectralite.svd_activation import apply_activation_aware_svd, print_actsvd_summary
from spectralite.svd_spectralite import (
    allocate_and_compress,
    build_whitened_svd_cache,
    print_spectralite_summary,
)
from spectralite.utils import get_logger, print_kv, print_section
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _speedups(row: dict[str, Any], dense_row: dict[str, Any]) -> dict[str, float]:
    dp = float(dense_row.get("prefill_ms_mean") or float("nan"))
    dd = float(dense_row.get("decode_ms_per_token_mean") or float("nan"))
    rp = float(row.get("prefill_ms_mean") or float("nan"))
    rd = float(row.get("decode_ms_per_token_mean") or float("nan"))
    return {
        "prefill_speedup_vs_dense": dp / rp if rp and rp > 0 else float("nan"),
        "decode_speedup_vs_dense": dd / rd if rd and rd > 0 else float("nan"),
    }


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
        phase="9",
        method=method,
        notes=notes,
        persist_artifacts=False,
        analytic_flops_fwd_ratio=float(analytic_keep),
    )


def run_phase9_spectralite_gate(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    keep_ratio: float = 0.75,
    rank_ratio: float = 0.5,
    kappa_speed: float = 1.0,
    protect_mode: str = "rho",
    calib_num_sequences: int = 32,
    calib_seq_len: int = 512,
    calib_batch_size: int = 2,
    ridge: float = 1e-2,
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 30_000,
    latency_reps_prefill: int = 30,
    latency_reps_decode: int = 20,
    run_zero_shot: bool = True,
    zero_shot_tasks: Sequence[str] = DEFAULT_ZERO_SHOT_TASKS,
    zero_shot_batch_size: int | str = 8,
    zero_shot_limit: Optional[int | float] = None,
    csv_name: str = "phase9_spectralite_gate.csv",
) -> dict[str, Any]:
    """Compare Spec-ρ ± gate vs ActSVD ± gate at matched keep / ratio.

    This closes the OPT-125M deploy gap: SpectraLite-ρ + latency gate as one config.
    """
    cfg = config or default_config()
    cfg.ensure_directories()

    print_section("Phase 9 — Dense reference")
    dense_metrics = _profile(
        dense_model,
        tokenizer,
        cfg=cfg,
        method="dense",
        notes="phase9 dense reference",
        csv_name=csv_name,
        analytic_keep=1.0,
        ppl_seq_len=ppl_seq_len,
        ppl_max_tokens=ppl_max_tokens,
        latency_reps_prefill=latency_reps_prefill,
        latency_reps_decode=latency_reps_decode,
    )
    dense_row = dense_metrics["row"]

    print_section("Phase 9 — Calibration + whitened spectra")
    batches = load_wikitext2_calibration_batches(
        tokenizer,
        num_sequences=calib_num_sequences,
        seq_len=calib_seq_len,
        batch_size=calib_batch_size,
        seed=cfg.seed,
    )
    activations = collect_linear_input_activations(dense_model, batches)
    cache = build_whitened_svd_cache(
        dense_model, activations, ridge=ridge, cov_method="ridge"
    )

    variants: list[tuple[str, dict[str, Any]]] = []

    # ActSVD ungated / gated (reference)
    for gated, tag in ((False, "actsvd"), (True, "actsvd_gate")):
        print_section(f"Phase 9 — {tag}")
        packed = apply_activation_aware_svd(
            dense_model,
            activations,
            rank_ratio=rank_ratio,
            ridge=ridge,
            cov_method="ridge",
            latency_gate=gated,
            kappa_speed=kappa_speed,
            clone=True,
        )
        print_actsvd_summary(packed["summary"])
        method = f"{tag}_r{rank_ratio:.2f}"
        metrics = _profile(
            packed["model"],
            tokenizer,
            cfg=cfg,
            method=method,
            notes=(
                f"ActSVD ridge gate={gated} kappa_speed={kappa_speed} "
                f"replaced={packed['summary']['num_replaced']} "
                f"gated_dense={packed['summary'].get('num_gated_dense', 0)}"
            ),
            csv_name=csv_name,
            analytic_keep=float(packed["summary"]["param_keep_ratio_touched"]),
            ppl_seq_len=ppl_seq_len,
            ppl_max_tokens=ppl_max_tokens,
            latency_reps_prefill=latency_reps_prefill,
            latency_reps_decode=latency_reps_decode,
        )
        row = {**metrics["row"], **_speedups(metrics["row"], dense_row)}
        variants.append(
            (
                method,
                {
                    "summary": packed["summary"],
                    "row": row,
                    "latency_gate": gated,
                    "family": "actsvd",
                },
            )
        )
        del packed, metrics
        _empty_cache()

    # SpectraLite-ρ ungated / gated (headline)
    for gated, tag in ((False, "spectralite_rho"), (True, "spectralite_rho_gate")):
        print_section(f"Phase 9 — {tag}")
        packed = allocate_and_compress(
            dense_model,
            cache,
            float(keep_ratio),
            clone=True,
            protect_mode=protect_mode,
            latency_gate=gated,
            kappa_speed=kappa_speed,
        )
        print_spectralite_summary(packed["summary"], packed["allocation"])
        method = f"{tag}_k{keep_ratio:.2f}"
        metrics = _profile(
            packed["model"],
            tokenizer,
            cfg=cfg,
            method=method,
            notes=(
                f"SpectraLite protect={protect_mode} gate={gated} "
                f"kappa_speed={kappa_speed} keep_target={keep_ratio} "
                f"achieved={packed['summary']['param_keep_ratio_touched']:.4f} "
                f"replaced={packed['summary']['num_replaced']} "
                f"gated_dense={packed['summary'].get('num_gated_dense', 0)}"
            ),
            csv_name=csv_name,
            analytic_keep=float(packed["summary"]["param_keep_ratio_touched"]),
            ppl_seq_len=ppl_seq_len,
            ppl_max_tokens=ppl_max_tokens,
            latency_reps_prefill=latency_reps_prefill,
            latency_reps_decode=latency_reps_decode,
        )
        row = {**metrics["row"], **_speedups(metrics["row"], dense_row)}
        variants.append(
            (
                method,
                {
                    "summary": {
                        k: packed["summary"][k]
                        for k in packed["summary"]
                        if k not in {"replacements", "gated_dense"}
                    },
                    "allocation": {
                        "target_keep_ratio": packed["allocation"]["target_keep_ratio"],
                        "achieved_keep_ratio": packed["allocation"]["achieved_keep_ratio"],
                        "lambda": packed["allocation"]["lambda"],
                    },
                    "row": row,
                    "latency_gate": gated,
                    "family": "spectralite",
                    "num_replaced": packed["summary"]["num_replaced"],
                    "num_gated_dense": packed["summary"].get("num_gated_dense", 0),
                },
            )
        )
        del packed, metrics
        _empty_cache()

    zero_shot_rows: list[dict[str, Any]] = []
    if run_zero_shot:
        print_section("Phase 9 — Zero-shot on SpectraLite-ρ + gate")
        packed_zs = allocate_and_compress(
            dense_model,
            cache,
            float(keep_ratio),
            clone=True,
            protect_mode=protect_mode,
            latency_gate=True,
            kappa_speed=kappa_speed,
        )
        zs = run_lm_eval(
            packed_zs["model"],
            tokenizer,
            tasks=zero_shot_tasks,
            num_fewshot=0,
            batch_size=zero_shot_batch_size,
            limit=zero_shot_limit,
            method="spectralite_rho_gate_k0.75",
        )
        zs["notes"] = (
            f"Phase 9 Spec-ρ+gate keep={keep_ratio} kappa_speed={kappa_speed} "
            f"replaced={packed_zs['summary']['num_replaced']} "
            f"gated_dense={packed_zs['summary'].get('num_gated_dense', 0)}"
        )
        zero_shot_rows.append(zs)
        write_json("phase9_zeroshot_spectralite_rho_gate.json", zs)
        del packed_zs
        _empty_cache()

    claim = {}
    by_method = {m: v for m, v in variants}
    ungated = by_method.get("spectralite_rho_k0.75")
    gated = by_method.get("spectralite_rho_gate_k0.75")
    act_gate = by_method.get("actsvd_gate_r0.50")
    if ungated and gated:
        claim = {
            "spec_rho_ungated_c4": ungated["row"].get("ppl_c4"),
            "spec_rho_gated_c4": gated["row"].get("ppl_c4"),
            "spec_rho_ungated_decode_ms": ungated["row"].get("decode_ms_per_token_mean"),
            "spec_rho_gated_decode_ms": gated["row"].get("decode_ms_per_token_mean"),
            "actsvd_gated_c4": act_gate["row"].get("ppl_c4") if act_gate else None,
            "actsvd_gated_decode_ms": (
                act_gate["row"].get("decode_ms_per_token_mean") if act_gate else None
            ),
            "spec_rho_gated_zero_shot_avg": (
                zero_shot_rows[0].get("zero_shot_avg") if zero_shot_rows else None
            ),
        }
        print_section("Phase 9 — Claim snapshot")
        for k, v in claim.items():
            print_kv(k, v)

    payload = {
        "phase": "9",
        "model_name": cfg.model_name,
        "keep_ratio": keep_ratio,
        "rank_ratio": rank_ratio,
        "kappa_speed": kappa_speed,
        "protect_mode": protect_mode,
        "dense_row": dense_row,
        "variants": {m: v for m, v in variants},
        "claim": claim,
        "zero_shot": zero_shot_rows,
    }
    write_json("phase9_summary.json", payload)
    write_json("phase9_claim.json", claim)

    mark_phase_complete(
        "9",
        artifacts={
            "summary": "results/phase9_summary.json",
            "claim": "results/phase9_claim.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "keep_ratio": keep_ratio,
            "kappa_speed": kappa_speed,
            "spec_rho_gated_c4": claim.get("spec_rho_gated_c4"),
            "spec_rho_gated_decode_ms": claim.get("spec_rho_gated_decode_ms"),
            "spec_rho_gated_zero_shot_avg": claim.get("spec_rho_gated_zero_shot_avg"),
        },
        notes=(
            "SpectraLite-ρ ± latency gate vs ActSVD ± gate at keep≈0.75; "
            "closes OPT-125M deploy default."
        ),
        config=cfg,
    )
    print_git_save_instructions()
    return payload
