"""Phase 2 orchestrator: vanilla SVD sweep → metrics → git artifacts."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from spectralite.artifacts import mark_phase_complete, print_git_save_instructions, write_json
from spectralite.benchmark import run_phase1_dense_baseline
from spectralite.config import Config, default_config
from spectralite.svd_vanilla import apply_vanilla_svd, print_svd_summary
from spectralite.utils import get_logger, print_section

logger = get_logger(__name__)


def run_phase2_vanilla_svd_sweep(
    dense_model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    rank_ratios: Sequence[float] = (0.5, 0.4, 0.3),
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 30_000,
    latency_reps_prefill: int = 30,
    latency_reps_decode: int = 20,
    csv_name: str = "phase2_vanilla_svd.csv",
) -> dict[str, Any]:
    """Compress with uniform truncated SVD at several ratios; profile each.

    Deep-copies the dense model per ratio so the original stays intact.
    """
    cfg = config or default_config()
    cfg.ensure_directories()

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    print_section("Phase 2 — Vanilla truncated SVD sweep")
    for ratio in rank_ratios:
        logger.info("Applying vanilla SVD rank_ratio=%.2f", ratio)
        packed = apply_vanilla_svd(dense_model, rank_ratio=float(ratio), clone=True)
        compressed = packed["model"]
        summary = packed["summary"]
        print_svd_summary(summary)

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
            phase="2",
            method=f"vanilla_svd_r{ratio:.2f}",
            notes=(
                f"vanilla_svd uniform ratio={ratio}; "
                f"replaced={summary['num_replaced']}; "
                f"param_keep_touched={summary['param_keep_ratio_touched']:.4f}"
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

    compact_summaries = []
    for s in all_summaries:
        compact_summaries.append(
            {
                "rank_ratio": s["rank_ratio"],
                "num_replaced": s["num_replaced"],
                "dense_params_touched": s["dense_params_touched"],
                "lowrank_params_touched": s["lowrank_params_touched"],
                "params_saved_touched": s["params_saved_touched"],
                "param_keep_ratio_touched": s["param_keep_ratio_touched"],
                "replacements_preview": s["replacements"][:3],
            }
        )

    payload = {
        "phase": "2",
        "rank_ratios": list(rank_ratios),
        "rows": all_rows,
        "summaries": compact_summaries,
        "csv": f"results/{csv_name}",
    }
    write_json(
        "phase2_replacements.json",
        {"rank_ratios": list(rank_ratios), "summaries": all_summaries},
    )
    write_json("phase2_summary.json", payload)
    mark_phase_complete(
        "2",
        artifacts={
            "summary": "results/phase2_summary.json",
            "replacements": "results/phase2_replacements.json",
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
        notes="Vanilla Eckart-Young SVD with fused LowRankLinear; uniform ratios.",
        config=cfg,
    )
    print_git_save_instructions()
    return payload
