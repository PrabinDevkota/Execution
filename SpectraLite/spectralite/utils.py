"""Utility helpers: seeding, logging, and pretty printing."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed applied to all available RNG backends.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Torch may not be installed yet during early bootstrap.
        pass


def get_logger(
    name: str = "spectralite",
    level: str | int = "INFO",
    log_file: Optional[Path | str] = None,
) -> logging.Logger:
    """Create or retrieve a configured SpectraLite logger.

    Avoids duplicate handlers when called repeatedly from notebooks.

    Args:
        name: Logger name.
        level: Logging level name or numeric level.
        log_file: Optional path for a file handler.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
    logger.setLevel(numeric_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def format_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        String such as ``1.24 GB``.
    """
    if num_bytes < 0:
        return "n/a"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def format_params(count: int) -> str:
    """Format a parameter count with SI-style suffixes.

    Args:
        count: Number of parameters.

    Returns:
        String such as ``125.00M``.
    """
    abs_count = abs(count)
    if abs_count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if abs_count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if abs_count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def print_section(title: str, width: int = 72, char: str = "=") -> None:
    """Print a visually distinct section banner.

    Args:
        title: Section heading text.
        width: Banner width in characters.
        char: Fill character for the rule lines.
    """
    rule = char * width
    print(f"\n{rule}")
    print(f" {title}")
    print(f"{rule}")


def print_kv(
    key: str,
    value: Any,
    key_width: int = 28,
) -> None:
    """Print a single key–value pair aligned for readable reports.

    Args:
        key: Left-hand label.
        value: Right-hand value (converted via ``str``).
        key_width: Character width reserved for the key column.
    """
    print(f"  {key:<{key_width}} {value}")


def print_mapping(
    mapping: Mapping[str, Any],
    title: Optional[str] = None,
    key_width: int = 28,
) -> None:
    """Pretty-print a mapping as aligned key–value lines.

    Args:
        mapping: Dictionary-like object to display.
        title: Optional section title printed first.
        key_width: Character width reserved for keys.
    """
    if title:
        print_section(title)
    for key, value in mapping.items():
        print_kv(str(key), value, key_width=key_width)


def print_checklist(items: Sequence[tuple[str, bool]]) -> None:
    """Print a Phase-completion style checklist.

    Args:
        items: Sequence of ``(label, ok)`` pairs.
    """
    print_section("Phase 0 Status")
    for label, ok in items:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}")
