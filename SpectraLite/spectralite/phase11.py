"""Phase 11: scale SpectraLite to LLaMA-3.2-1B (memory-lean Colab path)."""

from __future__ import annotations

import gc
from typing import Any, Optional, Sequence

import torch

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, config_for_model
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

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass


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
        # LLaMA GQA crashes FlopCounterMode; soft-skip inside flops.py anyway.
        run_empirical_flops=False,
        run_ppl=True,
        csv_name=csv_name,
        phase="11",
        method=method,
        notes=notes,
        persist_artifacts=False,
        analytic_flops_fwd_ratio=float(analytic_keep),
    )


def run_phase11_llama32_1b_ladder(
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
    max_tokens_per_layer: int = 4096,
    ridge: Optional[float] = None,
    ppl_seq_len: Optional[int] = None,
    ppl_max_tokens: Optional[int] = None,
    latency_reps_prefill: int = 15,
    latency_reps_decode: int = 10,
    run_zero_shot: bool = False,
    zero_shot_tasks: Sequence[str] = DEFAULT_ZERO_SHOT_TASKS,
    zero_shot_batch_size: int | str = 2,
    zero_shot_limit: Optional[int | float] = None,
    csv_name: str = "phase11_llama32_1b.csv",
) -> dict[str, Any]:
    """Dense → ActSVD → ActSVD+gate → Spec-ρ → Spec-ρ+gate on LLaMA-3.2-1B.

    Memory-lean defaults for Colab system RAM:
    - ``max_tokens_per_layer=4096`` (was 50k — the OOM cause on 8192-d MLP inputs)
    - float32 SVD cache, no stored dense ``W``
    - activations freed before Spec-ρ; one compressed clone at a time
    - ``run_zero_shot=False`` by default (enable in a second pass if needed)
    """
    cfg = config or config_for_model(model_name)
    cfg.model_name = model_name
    cfg.ensure_directories()
    set_seed(cfg.seed)

    calib_num_sequences = calib_num_sequences or min(int(cfg.calib_num_sequences), 16)
    calib_seq_len = calib_seq_len or min(int(cfg.calib_seq_len), 256)
    calib_batch_size = calib_batch_size or 1
    ridge = cfg.whitening_ridge if ridge is None else ridge
    ppl_seq_len = ppl_seq_len or min(int(cfg.ppl_seq_len), 256)
    ppl_max_tokens = ppl_max_tokens or min(int(cfg.ppl_max_tokens), 15_000)

    if dense_model is None or tokenizer is None:
        print_section(f"Phase 11 — Load {model_name}")
        dense_model, tokenizer = load_model_and_tokenizer(config=cfg)
    dense_model.eval()
    _free()

    print_section(f"Phase 11 — Dense baseline ({model_name})")
    dense_metrics = _profile(
        dense_model,
        tokenizer,
        cfg=cfg,
        method="dense",
        notes=f"phase11 dense {model_name}",
        csv_name=csv_name,
        analytic_keep=1.0,
        ppl_seq_len=ppl_seq_len,
        ppl_max_tokens=ppl_max_tokens,
        latency_reps_prefill=latency_reps_prefill,
        latency_reps_decode=latency_reps_decode,
    )
    dense_row = dense_metrics["row"]
    del dense_metrics
    _free()

    print_section(
        f"Phase 11 — Calibration (seqs={calib_num_sequences}, "
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

    variants: dict[str, Any] = {}

    # --- ActSVD paths need activations; run them first, then free acts ---
    for gated, tag in ((False, "actsvd"), (True, "actsvd_gate")):
        print_section(f"Phase 11 — {tag}")
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
        compressed = packed["model"]
        summary = packed["summary"]
        del packed
        _free()
        metrics = _profile(
            compressed,
            tokenizer,
            cfg=cfg,
            method=method,
            notes=(
                f"{model_name} ActSVD gate={gated} kappa_speed={kappa_speed} "
                f"replaced={summary['num_replaced']} "
                f"gated_dense={summary.get('num_gated_dense', 0)}"
            ),
            csv_name=csv_name,
            analytic_keep=float(summary["param_keep_ratio_touched"]),
            ppl_seq_len=ppl_seq_len,
            ppl_max_tokens=ppl_max_tokens,
            latency_reps_prefill=latency_reps_prefill,
            latency_reps_decode=latency_reps_decode,
        )
        variants[method] = {
            "row": {**metrics["row"], **_speedups(metrics["row"], dense_row)},
            "num_replaced": summary["num_replaced"],
            "num_gated_dense": summary.get("num_gated_dense", 0),
            "param_keep_ratio_touched": summary["param_keep_ratio_touched"],
        }
        del compressed, metrics, summary
        _free()

    print_section("Phase 11 — Build float32 spectral cache (consume activations)")
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

    for gated, tag in ((False, "spectralite_rho"), (True, "spectralite_rho_gate")):
        print_section(f"Phase 11 — {tag}")
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
        compressed = packed["model"]
        summary = packed["summary"]
        alloc = packed["allocation"]
        del packed
        _free()
        metrics = _profile(
            compressed,
            tokenizer,
            cfg=cfg,
            method=method,
            notes=(
                f"{model_name} Spec-ρ gate={gated} kappa_speed={kappa_speed} "
                f"keep={keep_ratio} achieved={summary['param_keep_ratio_touched']:.4f} "
                f"replaced={summary['num_replaced']} "
                f"gated_dense={summary.get('num_gated_dense', 0)}"
            ),
            csv_name=csv_name,
            analytic_keep=float(summary["param_keep_ratio_touched"]),
            ppl_seq_len=ppl_seq_len,
            ppl_max_tokens=ppl_max_tokens,
            latency_reps_prefill=latency_reps_prefill,
            latency_reps_decode=latency_reps_decode,
        )
        variants[method] = {
            "row": {**metrics["row"], **_speedups(metrics["row"], dense_row)},
            "num_replaced": summary["num_replaced"],
            "num_gated_dense": summary.get("num_gated_dense", 0),
            "param_keep_ratio_touched": summary["param_keep_ratio_touched"],
            "lambda": alloc["lambda"],
        }
        del compressed, metrics, summary, alloc
        _free()

    zero_shot: list[dict[str, Any]] = []
    if run_zero_shot:
        print_section("Phase 11 — Zero-shot (one model at a time)")
        for gated, method, builder in (
            (
                True,
                "spectralite_rho_gate_k0.75",
                lambda: allocate_and_compress(
                    dense_model,
                    cache,
                    float(keep_ratio),
                    clone=True,
                    protect_mode=protect_mode,
                    latency_gate=True,
                    kappa_speed=kappa_speed,
                )["model"],
            ),
            (
                True,
                "actsvd_gate_r0.50",
                None,  # needs fresh activations — skip if consumed
            ),
        ):
            if method.startswith("actsvd"):
                logger.warning(
                    "Skipping %s zero-shot in lean mode (activations consumed). "
                    "Re-run with a dedicated ActSVD-only zero-shot cell if needed.",
                    method,
                )
                continue
            model_eval = builder()
            _free()
            zs = run_lm_eval(
                model_eval,
                tokenizer,
                tasks=zero_shot_tasks,
                num_fewshot=0,
                batch_size=zero_shot_batch_size,
                limit=zero_shot_limit,
                method=method,
            )
            zs["notes"] = f"Phase 11 {model_name} {method}"
            zero_shot.append(zs)
            write_json(f"phase11_zeroshot_{method}.json", zs)
            del model_eval
            _free()

    del cache
    _free()

    claim = {
        "model_name": model_name,
        "dense_c4": dense_row.get("ppl_c4"),
        "dense_decode_ms": dense_row.get("decode_ms_per_token_mean"),
        "max_tokens_per_layer": max_tokens_per_layer,
        "calib_num_sequences": calib_num_sequences,
        "calib_seq_len": calib_seq_len,
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

    print_section("Phase 11 — Claim snapshot")
    for k, v in claim.items():
        print_kv(k, v)

    payload = {
        "phase": "11",
        "model_name": model_name,
        "keep_ratio": keep_ratio,
        "rank_ratio": rank_ratio,
        "kappa_speed": kappa_speed,
        "memory_lean": True,
        "max_tokens_per_layer": max_tokens_per_layer,
        "dense_row": dense_row,
        "variants": variants,
        "claim": claim,
        "zero_shot": zero_shot,
    }
    write_json("phase11_summary.json", payload)
    write_json("phase11_claim.json", claim)

    mark_phase_complete(
        "11",
        artifacts={
            "summary": "results/phase11_summary.json",
            "claim": "results/phase11_claim.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "model_name": model_name,
            "actsvd_gated_c4": claim.get("actsvd_gate_r0.50_c4"),
            "actsvd_gated_decode_speedup": claim.get(
                "actsvd_gate_r0.50_decode_speedup"
            ),
            "spec_rho_gated_c4": claim.get("spectralite_rho_gate_k0.75_c4"),
            "spec_rho_gated_decode_speedup": claim.get(
                "spectralite_rho_gate_k0.75_decode_speedup"
            ),
            "spec_rho_gated_zero_shot_avg": claim.get(
                "spectralite_rho_gate_k0.75_zero_shot_avg"
            ),
            "max_tokens_per_layer": max_tokens_per_layer,
        },
        notes=(
            f"Memory-lean LLaMA-3.2-1B ladder on {model_name}: "
            f"max_tokens/layer={max_tokens_per_layer}, float32 cache, "
            "zero_shot deferred to Phase 11b."
        ),
        config=cfg,
    )
    print_git_save_instructions()
    return payload
