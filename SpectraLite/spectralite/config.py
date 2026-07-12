"""Central configuration for SpectraLite experiments.

Phase 0 uses only the fields required for environment verification and a
single smoke-test generation on ``facebook/opt-125m``. Later phases will
extend this dataclass (calibration sizes, FLOP budgets, latency gates, etc.)
without scattering magic numbers through notebooks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Repository root: SpectraLite/  (parent of the spectralite/ package)
_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_DIR.parent


@dataclass
class Config:
    """Experiment configuration for SpectraLite Phase 0.

    Attributes:
        model_name: Hugging Face model identifier.
        seed: Global RNG seed for reproducibility.
        dtype: Preferred floating-point dtype string (``float16`` / ``float32``).
        device_map: Hugging Face ``device_map`` strategy (``auto`` for Phase 0).
        trust_remote_code: Passed to ``from_pretrained`` (OPT does not need it).
        max_new_tokens: Token budget for the smoke-test generation.
        smoke_prompt: Fixed prompt used in Phase 0 inference verification.
        results_dir: Directory for CSV / JSON experiment outputs.
        checkpoints_dir: Directory for saved weights (later phases).
        figures_dir: Directory for plots.
        logs_dir: Directory for log files.
        log_level: Default logging level name.
    """

    model_name: str = "facebook/opt-125m"
    seed: int = 42
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    max_new_tokens: int = 50
    smoke_prompt: str = "Explain Singular Value Decomposition in one sentence."
    # Phase 1 profiling defaults (paper protocol can raise these).
    calib_num_sequences: int = 256
    calib_seq_len: int = 2048
    ppl_seq_len: int = 512
    ppl_max_tokens: int = 50_000
    latency_warmup: int = 10
    latency_reps_prefill: int = 50
    latency_reps_decode: int = 30
    latency_prompt_len: int = 128
    latency_gen_tokens: int = 64
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")
    checkpoints_dir: Path = field(default_factory=lambda: REPO_ROOT / "checkpoints")
    figures_dir: Path = field(default_factory=lambda: REPO_ROOT / "figures")
    logs_dir: Path = field(default_factory=lambda: REPO_ROOT / "logs")
    log_level: str = "INFO"

    def ensure_directories(self) -> None:
        """Create standard output directories if they do not already exist."""
        for path in (
            self.results_dir,
            self.checkpoints_dir,
            self.figures_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of configuration values."""
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def default_config() -> Config:
    """Return a fresh Phase 0 default configuration instance."""
    return Config()


# Module-level convenience aliases (imported by notebooks / scripts).
MODEL_NAME: str = Config.model_name
DEVICE: str = "cuda"  # resolved at runtime; prefer system.resolve_device()
SEED: int = Config.seed
DTYPE: str = Config.dtype
MAX_NEW_TOKENS: int = Config.max_new_tokens
SMOKE_PROMPT: str = Config.smoke_prompt
