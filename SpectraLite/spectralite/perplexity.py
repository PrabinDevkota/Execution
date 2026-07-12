"""Perplexity evaluation on WikiText-2 / PTB / C4."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from spectralite.model_loader import get_model_device
from spectralite.utils import get_logger

logger = get_logger(__name__)


def _load_text_corpus(name: str, *, streaming_c4: bool = True) -> str:
    """Load a concatenated plain-text corpus for perplexity."""
    from datasets import load_dataset

    name = name.lower()
    if name in {"wikitext2", "wikitext-2", "wt2"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(t for t in ds["text"] if t and t.strip())
    if name in {"ptb", "penn"}:
        ds = load_dataset("ptb_text_only", split="test")
        # field is usually "sentence"
        key = "sentence" if "sentence" in ds.column_names else ds.column_names[0]
        return "\n".join(t for t in ds[key] if t and str(t).strip())
    if name in {"c4"}:
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=streaming_c4)
        chunks: list[str] = []
        # Cap streamed docs so Colab disk/time stay bounded.
        for i, row in enumerate(ds):
            text = row.get("text") or ""
            if text.strip():
                chunks.append(text)
            if i >= 200:
                break
        return "\n\n".join(chunks)
    raise ValueError(f"Unknown corpus: {name}")


@torch.inference_mode()
def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    corpus_name: str,
    *,
    seq_len: int = 512,
    max_tokens: int = 50_000,
    stride: Optional[int] = None,
) -> dict[str, float]:
    """Sliding-window perplexity over a corpus (non-overlapping if stride=seq_len).

    For the paper protocol use ``seq_len=2048`` and a large ``max_tokens``.
    Phase 1 defaults are lighter for Colab iteration on OPT-125M.
    """
    device = get_model_device(model)
    text = _load_text_corpus(corpus_name)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    if input_ids.numel() > max_tokens:
        input_ids = input_ids[:max_tokens]

    stride = seq_len if stride is None else stride
    nlls: list[torch.Tensor] = []
    total_tokens = 0

    # HF causal LMs: shift logits/labels inside loss when labels provided.
    for begin in range(0, input_ids.numel(), stride):
        end = min(begin + seq_len, input_ids.numel())
        if end - begin < 2:
            break
        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        outputs = model(input_ids=chunk, labels=chunk)
        # outputs.loss is mean NLL over non-masked tokens
        n_tok = chunk.numel() - 1  # typical causal shift
        nlls.append(outputs.loss.float() * n_tok)
        total_tokens += n_tok
        if end >= input_ids.numel():
            break

    if total_tokens == 0:
        return {"perplexity": float("nan"), "tokens": 0.0, "seq_len": float(seq_len)}

    nll_sum = torch.stack(nlls).sum()
    ppl = float(torch.exp(nll_sum / total_tokens).item())
    return {
        "perplexity": ppl,
        "tokens": float(total_tokens),
        "seq_len": float(seq_len),
        "max_tokens": float(max_tokens),
    }


def evaluate_ppl_suite(
    model: nn.Module,
    tokenizer: Any,
    *,
    seq_len: int = 512,
    max_tokens: int = 50_000,
    corpora: Optional[list[str]] = None,
) -> dict[str, float]:
    """Run WT2 / PTB / C4 perplexity and return a flat dict for CSV rows."""
    corpora = corpora or ["wikitext2", "ptb", "c4"]
    out: dict[str, float] = {
        "ppl_seq_len": float(seq_len),
        "ppl_max_tokens": float(max_tokens),
    }
    key_map = {"wikitext2": "ppl_wikitext2", "ptb": "ppl_ptb", "c4": "ppl_c4"}
    for name in corpora:
        logger.info("Computing perplexity: %s (seq_len=%d, max_tokens=%d)", name, seq_len, max_tokens)
        try:
            res = compute_perplexity(
                model, tokenizer, name, seq_len=seq_len, max_tokens=max_tokens
            )
            out[key_map[name]] = res["perplexity"]
            logger.info("%s ppl=%.4f over %d tokens", name, res["perplexity"], int(res["tokens"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("PPL failed for %s: %s", name, exc)
            out[key_map[name]] = float("nan")
    return out
