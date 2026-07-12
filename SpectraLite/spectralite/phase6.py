"""Phase 6 orchestrator: latency gate + fused ActSVD speedup measurement."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, default_config
from spectralite.latency import measure_prefill_cuda_graph
from spectralite.model_loader import get_model_device
from spectralite.svd_activation import apply_activation_aware_svd, print_actsvd_summary
from spectralite.utils import get_logger, print_kv, print_section
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)


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
    run_ppl: bool = True,
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
        run_ppl=run_ppl,
        csv_name=csv_name,
        phase="6",
        method=method,
        notes=notes,
        persist_artifacts=False,
        analytic_flops_fwd_ratio=float(analytic_keep),
    )


def run_phase6_latency_gate_sweep(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    rank_ratios: Sequence[float] = (0.5, 0.4, 0.3),
    kappa_speed: float = 1.0,
    calib_num_sequences: int = 32,
    calib_seq_len: int = 512,
    calib_batch_size: int = 2,
    ridge: float = 1e-2,
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 30_000,
    latency_reps_prefill: int = 30,
    latency_reps_decode: int = 20,
    csv_name: str = "phase6_latency_gate.csv",
    try_cuda_graph: bool = True,
) -> dict[str, Any]:
    """Phase 6: compare dense / ungated ActSVD / latency-gated ActSVD.

    Uses Phase-3-style ridge ActSVD (best quality path so far). Factor fusion is
    already in ``LowRankLinear``. Optionally times CUDA-graph prefill.
    """
    cfg = config or default_config()
    cfg.ensure_directories()

    print_section("Phase 6 — Dense reference (same session)")
    dense_metrics = _profile(
        dense_model,
        tokenizer,
        cfg=cfg,
        method="dense",
        notes="phase6 same-session dense reference",
        csv_name=csv_name,
        analytic_keep=1.0,
        ppl_seq_len=ppl_seq_len,
        ppl_max_tokens=ppl_max_tokens,
        latency_reps_prefill=latency_reps_prefill,
        latency_reps_decode=latency_reps_decode,
        run_ppl=True,
    )
    dense_row = dense_metrics["row"]

    cuda_graph_meta: dict[str, Any] = {"attempted": try_cuda_graph, "ok": False}
    if try_cuda_graph:
        print_section("Phase 6 — CUDA-graph prefill probe (dense)")
        device = get_model_device(dense_model)
        ids = torch.randint(
            0,
            max(int(tokenizer.vocab_size) - 1, 1),
            (1, getattr(cfg, "latency_prompt_len", 128)),
            device=device,
        )
        cg = measure_prefill_cuda_graph(
            dense_model,
            ids,
            warmup=5,
            reps=20,
        )
        cuda_graph_meta = {
            "attempted": True,
            "ok": bool(cg.get("cuda_graph", 0.0) >= 1.0),
            "dense_prefill_ms_mean": cg.get("prefill_ms_mean"),
            "eager_prefill_ms_mean": dense_row.get("prefill_ms_mean"),
        }
        print_kv("CUDA-graph prefill OK", cuda_graph_meta["ok"])
        print_kv("Graph prefill ms", cg.get("prefill_ms_mean"))

    print_section("Phase 6 — Calibration for ActSVD")
    batches = load_wikitext2_calibration_batches(
        tokenizer,
        num_sequences=calib_num_sequences,
        seq_len=calib_seq_len,
        batch_size=calib_batch_size,
        seed=cfg.seed,
    )
    activations = collect_linear_input_activations(dense_model, batches)

    all_rows: list[dict[str, Any]] = [dense_row]
    gate_logs: list[dict[str, Any]] = []

    print_section("Phase 6 — Ungated vs latency-gated ActSVD")
    for ratio in rank_ratios:
        for gated, tag in ((False, "ungated"), (True, "gated")):
            logger.info("ActSVD ratio=%.2f mode=%s kappa_speed=%.2f", ratio, tag, kappa_speed)
            packed = apply_activation_aware_svd(
                dense_model,
                activations,
                rank_ratio=float(ratio),
                ridge=ridge,
                cov_method="ridge",
                latency_gate=gated,
                kappa_speed=kappa_speed,
                clone=True,
            )
            print_actsvd_summary(packed["summary"])
            method = f"actsvd_{tag}_r{ratio:.2f}"
            metrics = _profile(
                packed["model"],
                tokenizer,
                cfg=cfg,
                method=method,
                notes=(
                    f"phase6 ActSVD {tag} ratio={ratio} kappa_speed={kappa_speed}; "
                    f"replaced={packed['summary']['num_replaced']}; "
                    f"gated_dense={packed['summary'].get('num_gated_dense', 0)}; "
                    f"keep={packed['summary']['param_keep_ratio_touched']:.4f}; "
                    f"fused_LowRankLinear=True"
                ),
                csv_name=csv_name,
                analytic_keep=packed["summary"]["param_keep_ratio_touched"],
                ppl_seq_len=ppl_seq_len,
                ppl_max_tokens=ppl_max_tokens,
                latency_reps_prefill=latency_reps_prefill,
                latency_reps_decode=latency_reps_decode,
                run_ppl=True,
            )
            row = dict(metrics["row"])
            sp = _speedups(row, dense_row)
            row.update(sp)
            all_rows.append(row)
            gate_logs.append(
                {
                    "method": method,
                    "rank_ratio": ratio,
                    "latency_gate": gated,
                    "kappa_speed": kappa_speed,
                    "num_replaced": packed["summary"]["num_replaced"],
                    "num_gated_dense": packed["summary"].get("num_gated_dense", 0),
                    "param_keep_ratio_touched": packed["summary"]["param_keep_ratio_touched"],
                    "prefill_ms_mean": row.get("prefill_ms_mean"),
                    "decode_ms_per_token_mean": row.get("decode_ms_per_token_mean"),
                    "prefill_speedup_vs_dense": sp["prefill_speedup_vs_dense"],
                    "decode_speedup_vs_dense": sp["decode_speedup_vs_dense"],
                    "ppl_c4": row.get("ppl_c4"),
                    "ppl_wikitext2": row.get("ppl_wikitext2"),
                    "gated_dense_preview": packed["summary"].get("gated_dense", [])[:5],
                    "replacements_preview": packed["summary"]["replacements"][:3],
                }
            )
            print_kv("Prefill speedup vs dense", f"{sp['prefill_speedup_vs_dense']:.3f}x")
            print_kv("Decode speedup vs dense", f"{sp['decode_speedup_vs_dense']:.3f}x")

            del packed, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "phase": "6",
        "kappa_speed": kappa_speed,
        "rank_ratios": list(rank_ratios),
        "ridge": ridge,
        "factor_fusion": True,
        "cuda_graph": cuda_graph_meta,
        "dense_row": dense_row,
        "rows": all_rows,
        "gate_logs": gate_logs,
        "csv": f"results/{csv_name}",
        "note": (
            "Latency gate keeps layers dense when r >= kappa_speed*mn/(m+n). "
            "Packed-MLP / FlashSVD runtime handoff deferred; fusion already in LowRankLinear."
        ),
    }
    write_json("phase6_gate_logs.json", {"kappa_speed": kappa_speed, "logs": gate_logs})
    write_json("phase6_summary.json", payload)
    mark_phase_complete(
        "6",
        artifacts={
            "summary": "results/phase6_summary.json",
            "gate_logs": "results/phase6_gate_logs.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "kappa_speed": kappa_speed,
            "num_settings": len(all_rows),
            "dense_prefill_ms": dense_row.get("prefill_ms_mean"),
            "dense_decode_ms": dense_row.get("decode_ms_per_token_mean"),
            "cuda_graph_ok": cuda_graph_meta.get("ok"),
            "best_decode_speedup": max(
                (g.get("decode_speedup_vs_dense") or 0.0) for g in gate_logs
            )
            if gate_logs
            else None,
        },
        notes=(
            "Latency feasibility gate on ridge ActSVD; fused LowRankLinear; "
            "optional CUDA-graph prefill probe."
        ),
        config=cfg,
    )
    print_git_save_instructions()
    return payload
