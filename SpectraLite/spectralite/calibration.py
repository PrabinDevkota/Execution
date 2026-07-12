"""Calibration data loading for activation-aware SVD (Phase 3+)."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from spectralite.utils import get_logger

logger = get_logger(__name__)


class _TokenChunkDataset(Dataset):
    """Fixed-length token chunks for calibration forwards."""

    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids  # (num_chunks, seq_len)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }


def load_wikitext2_calibration_batches(
    tokenizer: Any,
    *,
    num_sequences: int = 32,
    seq_len: int = 512,
    batch_size: int = 2,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Build calibration mini-batches from WikiText-2 (field-standard).

    Paper protocol uses up to 256×2048; Colab Phase-3 defaults are lighter
    (32×512) for OPT-125M iteration speed.
    """
    from datasets import load_dataset

    # Namespaced id required by newer huggingface_hub URI parsers.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t and str(t).strip())
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0]

    # Cut into non-overlapping windows, then subsample
    usable = (ids.numel() // seq_len) * seq_len
    if usable < seq_len:
        raise RuntimeError("WikiText-2 too short for requested seq_len")
    chunks = ids[:usable].view(-1, seq_len)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(chunks.shape[0], generator=g)
    take = min(num_sequences, int(chunks.shape[0]))
    selected = chunks[perm[:take]]

    dataset = _TokenChunkDataset(selected)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batches: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        batches.append(batch)

    logger.info(
        "Calibration ready: %d sequences × %d tokens (%d micro-batches)",
        take,
        seq_len,
        len(batches),
    )
    return batches


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a HF-style batch dict to ``device``."""
    return {k: v.to(device) for k, v in batch.items()}
