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
    # Optional HF Hub token for gated models (e.g. meta-llama/*). Prefer env HF_TOKEN.
    hf_token: str | None = None
    max_new_tokens: int = 50
    smoke_prompt: str = "Explain Singular Value Decomposition in one sentence."
    # Phase 1 profiling defaults (paper protocol can raise these).
    calib_num_sequences: int = 32
    calib_seq_len: int = 512
    calib_batch_size: int = 2
    whitening_ridge: float = 1e-2
    kappa_max: float = 1e4
    kappa_speed: float = 1.0
    spectral_protect_mode: str = "rho"
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
        # Never persist secrets into results artifacts.
        if payload.get("hf_token"):
            payload["hf_token"] = "***"
        return payload


def default_config() -> Config:
    """Return a fresh Phase 0 default configuration instance."""
    return Config()


# Model presets for the Phase-10+ scale ladder (override keep defaults light).
MODEL_PRESETS: dict[str, dict[str, object]] = {
    "facebook/opt-125m": {
        "calib_num_sequences": 32,
        "calib_seq_len": 512,
        "calib_batch_size": 2,
        "ppl_max_tokens": 50_000,
        "latency_reps_prefill": 50,
        "latency_reps_decode": 30,
    },
    "facebook/opt-1.3b": {
        "calib_num_sequences": 32,
        "calib_seq_len": 512,
        "calib_batch_size": 1,
        "ppl_max_tokens": 30_000,
        "latency_reps_prefill": 20,
        "latency_reps_decode": 15,
        "latency_prompt_len": 128,
        "latency_gen_tokens": 64,
    },
    "meta-llama/Llama-3.2-1B": {
        "calib_num_sequences": 32,
        "calib_seq_len": 512,
        "calib_batch_size": 1,
        "ppl_max_tokens": 30_000,
        "latency_reps_prefill": 20,
        "latency_reps_decode": 15,
        "trust_remote_code": False,
    },
}


def config_for_model(model_name: str, **overrides: object) -> Config:
    """Build a :class:`Config` for ``model_name`` using known presets."""
    cfg = default_config()
    cfg.model_name = model_name
    preset = MODEL_PRESETS.get(model_name, {})
    for key, value in preset.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


# Module-level convenience aliases (imported by notebooks / scripts).
MODEL_NAME: str = Config.model_name
DEVICE: str = "cuda"  # resolved at runtime; prefer system.resolve_device()
SEED: int = Config.seed
DTYPE: str = Config.dtype
MAX_NEW_TOKENS: int = Config.max_new_tokens
SMOKE_PROMPT: str = Config.smoke_prompt
