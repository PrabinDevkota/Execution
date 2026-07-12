"""Phase 1 dense baseline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from spectralite.config import Config, default_config
from spectralite.flops import (
    analytic_model_param_stats,
    measure_calflops_decode,
    measure_forward_flops,
)
from spectralite.latency import measure_decode_latency, measure_prefill_latency
from spectralite.model_loader import get_model_device, get_model_dtype
from spectralite.perplexity import evaluate_ppl_suite
from spectralite.results_io import append_results, empty_row
from spectralite.utils import get_logger, print_kv, print_section

logger = get_logger(__name__)


def run_phase1_dense_baseline(
    model: Any,
    tokenizer: Any,
    *,
    config: Optional[Config] = None,
    prompt_len: int = 128,
    gen_tokens: int = 64,
    latency_warmup: int = 10,
    latency_reps_prefill: int = 50,
    latency_reps_decode: int = 30,
    ppl_seq_len: int = 512,
    ppl_max_tokens: int = 50_000,
    run_calflops: bool = True,
    run_ppl: bool = True,
    csv_name: str = "phase1_dense_baselines.csv",
) -> dict[str, Any]:
    """Run Phase-1 dense profiling and append one CSV row.

    Reuses an already-loaded Phase-0 model when available.
    """
    cfg = config or default_config()
    cfg.ensure_directories()
    device = get_model_device(model)
    dtype = get_model_dtype(model)

    print_section("Phase 1 — Dense Baseline Profiling")
    print_kv("Model", cfg.model_name)
    print_kv("Device", str(device))
    print_kv("Dtype", str(dtype).replace("torch.", ""))

    # --- params ---
    param_stats = analytic_model_param_stats(model)
    print_kv("Parameters", f"{param_stats['param_count']:,}")
    print_kv("Weight memory", param_stats["param_memory_human"])

    # --- synthetic prefill batch for FLOPs + latency ---
    input_ids = torch.randint(
        low=0,
        high=max(tokenizer.vocab_size - 1, 1),
        size=(1, prompt_len),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    print_section("Empirical FLOPs (FlopCounterMode)")
    flop_stats = measure_forward_flops(model, input_ids, attention_mask=attention_mask)
    print_kv("Forward FLOPs", f"{flop_stats['empirical_flops_fwd']:,}")
    print_kv("Forward GFLOPs", f"{flop_stats['empirical_gflops_fwd']:.4f}")

    calflops_stats: dict[str, Any] = {
        "calflops_mflops_per_token": None,
        "calflops_error": "skipped",
    }
    if run_calflops:
        print_section("calflops probe (optional)")
        calflops_stats = measure_calflops_decode(
            model, tokenizer, prompt="Hello", max_new_tokens=8
        )
        print_kv("MFLOPs/token (calflops)", calflops_stats.get("calflops_mflops_per_token"))
        if calflops_stats.get("calflops_error"):
            print_kv("calflops note", calflops_stats["calflops_error"])

    print_section("Latency — Prefill")
    prefill = measure_prefill_latency(
        model,
        input_ids,
        attention_mask=attention_mask,
        warmup=latency_warmup,
        reps=latency_reps_prefill,
    )
    print_kv("Prefill ms (mean±std)", f"{prefill['prefill_ms_mean']:.3f} ± {prefill['prefill_ms_std']:.3f}")

    print_section("Latency — Generate (decode-oriented)")
    decode = measure_decode_latency(
        model,
        tokenizer,
        prompt=cfg.smoke_prompt,
        gen_tokens=gen_tokens,
        warmup=max(latency_warmup // 2, 3),
        reps=latency_reps_decode,
        batch_size=1,
    )
    print_kv(
        "ms/token (mean±std)",
        f"{decode['decode_ms_per_token_mean']:.3f} ± {decode['decode_ms_per_token_std']:.3f}",
    )
    print_kv("Throughput tok/s", f"{decode['throughput_tokens_per_s']:.2f}")

    ppl_stats: dict[str, float] = {
        "ppl_wikitext2": float("nan"),
        "ppl_ptb": float("nan"),
        "ppl_c4": float("nan"),
        "ppl_seq_len": float(ppl_seq_len),
        "ppl_max_tokens": float(ppl_max_tokens),
    }
    if run_ppl:
        print_section("Perplexity suite (WikiText-2 / PTB / C4)")
        ppl_stats = evaluate_ppl_suite(
            model,
            tokenizer,
            seq_len=ppl_seq_len,
            max_tokens=ppl_max_tokens,
        )
        print_kv("WikiText-2 PPL", ppl_stats.get("ppl_wikitext2"))
        print_kv("PTB PPL", ppl_stats.get("ppl_ptb"))
        print_kv("C4 PPL", ppl_stats.get("ppl_c4"))

    row = empty_row(
        phase="1",
        method="dense",
        model_name=cfg.model_name,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        seed=cfg.seed,
        param_count=param_stats["param_count"],
        param_memory_mb=param_stats["param_memory_mb"],
        analytic_flops_fwd_ratio=1.0,
        empirical_flops_fwd=flop_stats["empirical_flops_fwd"],
        calflops_mflops_per_token=calflops_stats.get("calflops_mflops_per_token"),
        prefill_ms_mean=prefill["prefill_ms_mean"],
        prefill_ms_std=prefill["prefill_ms_std"],
        decode_ms_per_token_mean=decode["decode_ms_per_token_mean"],
        decode_ms_per_token_std=decode["decode_ms_per_token_std"],
        throughput_tokens_per_s=decode["throughput_tokens_per_s"],
        prompt_len=prompt_len,
        gen_tokens=gen_tokens,
        batch_size=1,
        ppl_wikitext2=ppl_stats.get("ppl_wikitext2"),
        ppl_ptb=ppl_stats.get("ppl_ptb"),
        ppl_c4=ppl_stats.get("ppl_c4"),
        ppl_seq_len=ppl_stats.get("ppl_seq_len"),
        ppl_max_tokens=ppl_stats.get("ppl_max_tokens"),
        zero_shot_avg=None,
        notes="phase1_dense_opt125m_colab_dev; ppl_seq_len may be <2048 for iteration",
    )

    csv_path = Path(cfg.results_dir) / csv_name
    append_results(csv_path, row)
    print_section("Results written")
    print_kv("CSV", str(csv_path))

    return {
        "row": row,
        "csv_path": str(csv_path),
        "param_stats": param_stats,
        "flop_stats": flop_stats,
        "calflops_stats": calflops_stats,
        "prefill": prefill,
        "decode": decode,
        "ppl": ppl_stats,
    }
