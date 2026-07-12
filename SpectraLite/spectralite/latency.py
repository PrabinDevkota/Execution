"""CUDA-event latency measurement: prefill vs decode."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from spectralite.model_loader import get_model_device
from spectralite.utils import get_logger

logger = get_logger(__name__)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_prefill_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    warmup: int = 10,
    reps: int = 50,
) -> dict[str, float]:
    """Time a full forward (prefill) with ``torch.cuda.Event``.

    Falls back to CUDA synchronize + CPU timer if CUDA events are unavailable.
    """
    device = get_model_device(model)
    input_ids = input_ids.to(device)
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)

    for _ in range(max(warmup, 0)):
        _ = model(**kwargs)
    _synchronize(device)

    times_ms: list[float] = []
    use_cuda_events = device.type == "cuda"

    for _ in range(max(reps, 1)):
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(**kwargs)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
        else:
            import time

            t0 = time.perf_counter()
            _ = model(**kwargs)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean = sum(times_ms) / len(times_ms)
    var = sum((t - mean) ** 2 for t in times_ms) / max(len(times_ms) - 1, 1)
    return {
        "prefill_ms_mean": mean,
        "prefill_ms_std": var**0.5,
        "prefill_ms_median": sorted(times_ms)[len(times_ms) // 2],
        "reps": float(len(times_ms)),
    }


@torch.inference_mode()
def measure_decode_latency(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    gen_tokens: int = 64,
    warmup: int = 5,
    reps: int = 30,
    batch_size: int = 1,
) -> dict[str, float]:
    """Time autoregressive ``generate`` and report ms/token + throughput.

    Prefill+decode are included in ``generate`` wall time; we report
    end-to-end generate latency normalized by ``gen_tokens`` as decode-oriented
    ms/token (standard serving smoke metric). A stricter KV-cache-only step
    timer can be added in Phase 6.
    """
    device = get_model_device(model)
    encoded = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[-1])

    gen_kwargs = dict(
        max_new_tokens=gen_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for _ in range(max(warmup, 0)):
        _ = model.generate(**encoded, **gen_kwargs)
    _synchronize(device)

    times_ms: list[float] = []
    use_cuda_events = device.type == "cuda"

    for _ in range(max(reps, 1)):
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model.generate(**encoded, **gen_kwargs)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
        else:
            import time

            t0 = time.perf_counter()
            _ = model.generate(**encoded, **gen_kwargs)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean = sum(times_ms) / len(times_ms)
    var = sum((t - mean) ** 2 for t in times_ms) / max(len(times_ms) - 1, 1)
    ms_per_token = mean / max(gen_tokens, 1)
    throughput = (gen_tokens * batch_size * 1000.0) / max(mean, 1e-9)

    return {
        "prompt_len": float(prompt_len),
        "gen_tokens": float(gen_tokens),
        "batch_size": float(batch_size),
        "e2e_generate_ms_mean": mean,
        "e2e_generate_ms_std": var**0.5,
        "decode_ms_per_token_mean": ms_per_token,
        "decode_ms_per_token_std": (var**0.5) / max(gen_tokens, 1),
        "throughput_tokens_per_s": throughput,
        "reps": float(len(times_ms)),
    }
