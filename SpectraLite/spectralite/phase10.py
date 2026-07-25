"""Phase 10: scale SpectraLite to OPT-1.3B (same-family ladder step)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, config_for_model, default_config
from spectralite.downstream import DEFAULT_ZERO_SHOT_TASKS, run_lm_eval
from spectralite.model_loader import load_model_and_tokenizer
from spectralite.svd_activation import apply_activation_aware_svd, print_actsvd_summary
from spectralite.svd_spectralite import (
    allocate_and_compress,
    build_whitened_svd_cache,
    print_spectralite_summary,
)
from spectralite.utils import get_logger, print_kv, print_section, set_seed
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)

DEFAULT_MODEL = "facebook/opt-1.3b"


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
        phase="10",
        method=method,
        notes=notes,
        persist_artifacts=False,
        analytic_flops_fwd_ratio=float(analytic_keep),
    )


def run_phase10_opt13b_ladder(
    dense_model: Any = None,
    tokenizer: Any = None,
    *,
    config: Optional[Config] = None,
    model_name: str = DEFAULT_MODEL,
    keep_ratio: float = 0.75,
    rank_ratio: float = 0.5,
    kappa_speed: float = 1.0,
    protect_mode: str = "rho",
    calib_num_sequences: Optional[int] = None,
    calib_seq_len: Optional[int] = None,
    calib_batch_size: Optional[int] = None,
    ridge: Optional[float] = None,
    ppl_seq_len: Optional[int] = None,
    ppl_max_tokens: Optional[int] = None,
    latency_reps_prefill: int = 20,
    latency_reps_decode: int = 15,
    run_zero_shot: bool = True,
    zero_shot_tasks: Sequence[str] = DEFAULT_ZERO_SHOT_TASKS,
    zero_shot_batch_size: int | str = 4,
    zero_shot_limit: Optional[int | float] = None,
    csv_name: str = "phase10_opt13b.csv",
) -> dict[str, Any]:
    """Dense → ActSVD → ActSVD+gate → Spec-ρ → Spec-ρ+gate on OPT-1.3B.

    Loads the model if ``dense_model`` / ``tokenizer`` are not provided.
    Writes model-scoped artifacts under ``results/`` (phase10_*).
    """
    cfg = config or config_for_model(model_name)
    cfg.model_name = model_name
    cfg.ensure_directories()
    set_seed(cfg.seed)

    calib_num_sequences = calib_num_sequences or cfg.calib_num_sequences
    calib_seq_len = calib_seq_len or min(cfg.calib_seq_len, 512)
    calib_batch_size = calib_batch_size or cfg.calib_batch_size
    ridge = cfg.whitening_ridge if ridge is None else ridge
    ppl_seq_len = ppl_seq_len or min(cfg.ppl_seq_len, 512)
    ppl_max_tokens = ppl_max_tokens or min(cfg.ppl_max_tokens, 30_000)

    if dense_model is None or tokenizer is None:
        print_section(f"Phase 10 — Load {model_name}")
        dense_model, tokenizer = load_model_and_tokenizer(config=cfg)

    print_section(f"Phase 10 — Dense baseline ({model_name})")
    dense_metrics = _profile(
        dense_model,
        tokenizer,
        cfg=cfg,
        method="dense",
        notes=f"phase10 dense {model_name}",
        csv_name=csv_name,
        analytic_keep=1.0,
        ppl_seq_len=ppl_seq_len,
        ppl_max_tokens=ppl_max_tokens,
        latency_reps_prefill=latency_reps_prefill,
        latency_reps_decode=latency_reps_decode,
    )
    dense_row = dense_metrics["row"]

    print_section("Phase 10 — Calibration + spectra")
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

    variants: dict[str, Any] = {}

    def _run_actsvd(gated: bool) -> None:
        tag = "actsvd_gate" if gated else "actsvd"
        print_section(f"Phase 10 — {tag}")
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
                f"{model_name} ActSVD gate={gated} kappa_speed={kappa_speed} "
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
        variants[method] = {
            "row": {**metrics["row"], **_speedups(metrics["row"], dense_row)},
            "num_replaced": packed["summary"]["num_replaced"],
            "num_gated_dense": packed["summary"].get("num_gated_dense", 0),
            "param_keep_ratio_touched": packed["summary"]["param_keep_ratio_touched"],
        }
        del packed, metrics
        _empty_cache()

    def _run_spec(gated: bool) -> Any:
        tag = "spectralite_rho_gate" if gated else "spectralite_rho"
        print_section(f"Phase 10 — {tag}")
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
                f"{model_name} Spec-ρ gate={gated} kappa_speed={kappa_speed} "
                f"keep={keep_ratio} achieved={packed['summary']['param_keep_ratio_touched']:.4f} "
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
        variants[method] = {
            "row": {**metrics["row"], **_speedups(metrics["row"], dense_row)},
            "num_replaced": packed["summary"]["num_replaced"],
            "num_gated_dense": packed["summary"].get("num_gated_dense", 0),
            "param_keep_ratio_touched": packed["summary"]["param_keep_ratio_touched"],
            "lambda": packed["allocation"]["lambda"],
        }
        model_out = packed["model"] if gated else None
        if not gated:
            del packed
        del metrics
        _empty_cache()
        return model_out

    _run_actsvd(False)
    _run_actsvd(True)
    _run_spec(False)
    gated_model = _run_spec(True)

    zero_shot: list[dict[str, Any]] = []
    if run_zero_shot:
        print_section("Phase 10 — Zero-shot (Spec-ρ + gate + ActSVD gate)")
        if gated_model is None:
            gated_model = allocate_and_compress(
                dense_model,
                cache,
                float(keep_ratio),
                clone=True,
                protect_mode=protect_mode,
                latency_gate=True,
                kappa_speed=kappa_speed,
            )["model"]
        for method, model_eval in (
            ("spectralite_rho_gate_k0.75", gated_model),
        ):
            zs = run_lm_eval(
                model_eval,
                tokenizer,
                tasks=zero_shot_tasks,
                num_fewshot=0,
                batch_size=zero_shot_batch_size,
                limit=zero_shot_limit,
                method=method,
            )
            zs["notes"] = f"Phase 10 {model_name} {method}"
            zero_shot.append(zs)
            write_json(f"phase10_zeroshot_{method}.json", zs)
            _empty_cache()

        # ActSVD gated zero-shot for fair comparison
        packed_ag = apply_activation_aware_svd(
            dense_model,
            activations,
            rank_ratio=rank_ratio,
            ridge=ridge,
            cov_method="ridge",
            latency_gate=True,
            kappa_speed=kappa_speed,
            clone=True,
        )
        zs = run_lm_eval(
            packed_ag["model"],
            tokenizer,
            tasks=zero_shot_tasks,
            num_fewshot=0,
            batch_size=zero_shot_batch_size,
            limit=zero_shot_limit,
            method="actsvd_gate_r0.50",
        )
        zs["notes"] = f"Phase 10 {model_name} actsvd_gate"
        zero_shot.append(zs)
        write_json("phase10_zeroshot_actsvd_gate_r0.50.json", zs)
        del packed_ag, gated_model
        _empty_cache()

    claim = {
        "model_name": model_name,
        "dense_c4": dense_row.get("ppl_c4"),
        "dense_decode_ms": dense_row.get("decode_ms_per_token_mean"),
    }
    for key in (
        "actsvd_r0.50",
        "actsvd_gate_r0.50",
        "spectralite_rho_k0.75",
        "spectralite_rho_gate_k0.75",
    ):
        if key in variants:
            claim[f"{key}_c4"] = variants[key]["row"].get("ppl_c4")
            claim[f"{key}_decode_ms"] = variants[key]["row"].get(
                "decode_ms_per_token_mean"
            )
            claim[f"{key}_decode_speedup"] = variants[key]["row"].get(
                "decode_speedup_vs_dense"
            )
    for zs in zero_shot:
        claim[f"{zs['method']}_zero_shot_avg"] = zs.get("zero_shot_avg")

    print_section("Phase 10 — Claim snapshot")
    for k, v in claim.items():
        print_kv(k, v)

    payload = {
        "phase": "10",
        "model_name": model_name,
        "keep_ratio": keep_ratio,
        "rank_ratio": rank_ratio,
        "kappa_speed": kappa_speed,
        "dense_row": dense_row,
        "variants": variants,
        "claim": claim,
        "zero_shot": zero_shot,
    }
    write_json("phase10_summary.json", payload)
    write_json("phase10_claim.json", claim)

    mark_phase_complete(
        "10",
        artifacts={
            "summary": "results/phase10_summary.json",
            "claim": "results/phase10_claim.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "model_name": model_name,
            "spec_rho_gated_c4": claim.get("spectralite_rho_gate_k0.75_c4"),
            "spec_rho_gated_decode_speedup": claim.get(
                "spectralite_rho_gate_k0.75_decode_speedup"
            ),
            "spec_rho_gated_zero_shot_avg": claim.get(
                "spectralite_rho_gate_k0.75_zero_shot_avg"
            ),
        },
        notes=f"Scale ladder on {model_name}: dense/ActSVD/Spec-ρ ± latency gate.",
        config=cfg,
    )
    print_git_save_instructions()
    return payload
