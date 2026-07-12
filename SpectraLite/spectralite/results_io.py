"""Results I/O: frozen CSV schema for SpectraLite experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

# Frozen Phase-1+ schema (append-only columns; do not reorder casually).
RESULTS_COLUMNS: list[str] = [
    "phase",
    "method",
    "model_name",
    "device",
    "dtype",
    "seed",
    "param_count",
    "param_memory_mb",
    "analytic_flops_fwd_ratio",
    "empirical_flops_fwd",
    "calflops_mflops_per_token",
    "prefill_ms_mean",
    "prefill_ms_std",
    "decode_ms_per_token_mean",
    "decode_ms_per_token_std",
    "throughput_tokens_per_s",
    "prompt_len",
    "gen_tokens",
    "batch_size",
    "ppl_wikitext2",
    "ppl_ptb",
    "ppl_c4",
    "ppl_seq_len",
    "ppl_max_tokens",
    "zero_shot_avg",
    "notes",
]


def empty_row(**overrides: Any) -> dict[str, Any]:
    """Return a schema row with ``None`` defaults, then apply overrides."""
    row = {col: None for col in RESULTS_COLUMNS}
    row.update(overrides)
    return row


def append_results(
    path: Path | str,
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> Path:
    """Append one or more result rows to a CSV, creating it with a header if needed.

    Args:
        path: Destination CSV path.
        rows: Single mapping or sequence of mappings (extra keys ignored).

    Returns:
        Resolved path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, Mapping):
        row_list: list[Mapping[str, Any]] = [rows]
    else:
        row_list = list(rows)

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in row_list:
            writer.writerow({col: row.get(col) for col in RESULTS_COLUMNS})
    return path


def load_results(path: Path | str) -> pd.DataFrame:
    """Load a results CSV into a DataFrame (empty schema if missing)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    return pd.read_csv(path)
