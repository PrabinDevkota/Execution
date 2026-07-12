"""SpectraLite: spectral-decay-guided SVD compression for decoder-only LLMs.

Phase 0 exposes environment, loading, and model-analysis utilities only.
Compression algorithms are intentionally absent.
"""

from __future__ import annotations

from spectralite.config import Config, default_config
from spectralite.utils import get_logger, set_seed

__all__ = [
    "Config",
    "default_config",
    "get_logger",
    "set_seed",
    "__version__",
]

__version__ = "0.1.0-phase0"
