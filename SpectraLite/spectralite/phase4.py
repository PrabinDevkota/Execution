"""Phase 4 orchestrator: spectral-entropy rank allocation vs Phase-3 budgets."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, read_json, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.calibration import load_wikitext2_calibration_batches
from spectralite.config import Config, default_config
from spectralite.svd_spectralite import (
    allocate_and_compress,
    build_whitened_svd_cache,
    print_spectralite_summary,
)
from spectralite.utils import get_logger, print_section
from spectralite.whitening import collect_linear_input_activations

logger = get_logger(__name__)


def _budgets_from_phase3(config: Config) -> list[float]:
    """Matched FLOP keep ratios from Phase-3 ActSVD summaries when available."""
    try:
        payload = read_json("phase3_summary.json", config)
        keeps: list[float] = []
        for s in payload.get("summaries", []):
            k = s.get("param_keep_ratio_touched")
            if k is not None:
                keeps.append(float(k))
        if keeps:
            logger.info("Using Phase-3 matched budgets: %s", keeps)
            return keeps
    except FileNotFoundError:
        logger.warning("phase3_summary.json missing — using default keep ratios")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read Phase-3 budgets (%s); using defaults", exc)
    return [0.75, 0.60, 0.45]


def run_phase4_spectralite_sweep(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    target_keep_ratios: Optional[Sequence[float]] = None,
    calib_num_sequences: int = 32,
    calib_seq_len: int = 512,
    calib_batch_size: int = 2,
    ridge: float = 1e-2,
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 30_000,
    latency_reps_prefill: int = 30,
    latency_reps_decode: int = 20,
    csv_name: str = "phase4_spectralite.csv",
) -> dict[str, Any]:
    """Whitened spectral-entropy allocation at Phase-3-matched FLOP budgets.

    Novelty (Phase 4): per-matrix Roy–Vetterli ``ρ_eff`` + stable-rank importance
    → protect score → global binary-search ``λ`` under a keep-ratio budget.
    Sensitivity Fisher / Ledoit–Wolf / latency gate arrive in Phases 5–6.
    """
    cfg = config or default_config()
    cfg.ensure_directories()

    budgets = list(target_keep_ratios) if target_keep_ratios is not None else _budgets_from_phase3(cfg)

    print_section("Phase 4 — Calibration (WikiText-2)")
    batches = load_wikitext2_calibration_batches(
        tokenizer,
        num_sequences=calib_num_sequences,
        seq_len=calib_seq_len,
        batch_size=calib_batch_size,
        seed=cfg.seed,
    )

    print_section("Phase 4 — Collect activations + whitened spectra")
    activations = collect_linear_input_activations(dense_model, batches)
    cache = build_whitened_svd_cache(dense_model, activations, ridge=ridge)
    write_json(
        "phase4_spectrum_meta.json",
        {
            "ridge": ridge,
            "calib_num_sequences": calib_num_sequences,
            "calib_seq_len": calib_seq_len,
            "layers": {
                name: {
                    "q": int(e["q"]),
                    "rho_eff": float(e["rho_eff"]),
                    "compressibility": float(e["compressibility"]),
                    "stable_rank": float(e["stable_rank"]),
                    "protect": float(e["protect"]),
                    "in_features": int(e["in_features"]),
                    "out_features": int(e["out_features"]),
                }
                for name, e in cache.items()
            },
        },
    )

    all_rows: list[dict[str, Any]] = []
    all_pack_summaries: list[dict[str, Any]] = []
    all_allocs: list[dict[str, Any]] = []

    print_section("Phase 4 — SpectraLite allocation sweep (matched vs Phase 3)")
    for keep in budgets:
        logger.info("SpectraLite target_keep=%.4f", keep)
        packed = allocate_and_compress(dense_model, cache, float(keep), clone=True)
        compressed = packed["model"]
        summary = packed["summary"]
        alloc = packed["allocation"]
        print_spectralite_summary(summary, alloc)

        method = f"spectralite_keep{keep:.2f}"
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
            phase="4",
            method=method,
            notes=(
                f"spectralite spectral-alloc keep_target={keep:.4f} "
                f"achieved={alloc['achieved_keep_ratio']:.4f} lambda={alloc['lambda']:.6f}; "
                f"ridge={ridge}; calib={calib_num_sequences}x{calib_seq_len}; "
                f"replaced={summary['num_replaced']}"
            ),
            persist_artifacts=False,
            analytic_flops_fwd_ratio=float(summary["param_keep_ratio_touched"]),
        )
        all_rows.append(metrics["row"])
        all_pack_summaries.append(
            {
                "target_keep_ratio": alloc["target_keep_ratio"],
                "achieved_keep_ratio": alloc["achieved_keep_ratio"],
                "lambda": alloc["lambda"],
                "num_replaced": summary["num_replaced"],
                "param_keep_ratio_touched": summary["param_keep_ratio_touched"],
                "params_saved_touched": summary["params_saved_touched"],
                "replacements_preview": summary["replacements"][:3],
                "allocation_rows": packed["allocation_rows"],
            }
        )
        all_allocs.append(alloc)

        del compressed, packed, metrics
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    payload = {
        "phase": "4",
        "target_keep_ratios": budgets,
        "ridge": ridge,
        "calib_num_sequences": calib_num_sequences,
        "calib_seq_len": calib_seq_len,
        "rows": all_rows,
        "summaries": all_pack_summaries,
        "csv": f"results/{csv_name}",
        "phase3_c4_reference": [122.61, 555.45, 2286.51],
    }
    write_json(
        "phase4_allocations.json",
        {"budgets": budgets, "allocations": all_allocs, "summaries": all_pack_summaries},
    )
    write_json("phase4_summary.json", payload)
    mark_phase_complete(
        "4",
        artifacts={
            "summary": "results/phase4_summary.json",
            "allocations": "results/phase4_allocations.json",
            "spectrum": "results/phase4_spectrum_meta.json",
            "csv": f"results/{csv_name}",
            "status": "results/phase_status.json",
        },
        metrics={
            "target_keep_ratios": budgets,
            "num_settings": len(all_rows),
            "last_method": all_rows[-1]["method"] if all_rows else None,
            "last_ppl_c4": all_rows[-1].get("ppl_c4") if all_rows else None,
            "last_prefill_ms": all_rows[-1].get("prefill_ms_mean") if all_rows else None,
        },
        notes=(
            "SpectraLite novelty: Roy–Vetterli ρ_eff + stable-rank protect scores; "
            "binary-search λ under Phase-3-matched FLOP keep ratios."
        ),
        config=cfg,
    )
    print_git_save_instructions()
    return payload
