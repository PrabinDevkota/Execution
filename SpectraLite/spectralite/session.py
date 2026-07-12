"""Session restore helpers for new Colab runtimes."""

from __future__ import annotations

from typing import Any, Optional

from spectralite.artifacts import (
    is_phase_complete,
    load_status,
    print_git_save_instructions,
    print_progress_dashboard,
    read_json,
    should_skip_phase,
)
from spectralite.config import Config, default_config
from spectralite.model_loader import load_model_and_tokenizer
from spectralite.utils import get_logger, print_kv, print_section, set_seed

logger = get_logger(__name__)


def restore_session(
    *,
    config: Optional[Config] = None,
    load_model_if_needed: bool = True,
    force_reload_model: bool = False,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Restore experiment context after ``git pull`` on a fresh runtime.

    - Always shows the progress dashboard from ``results/phase_status.json``.
    - Reloads the HF model only if a later incomplete phase needs it (or forced).
    - Does **not** re-run FLOP/PPL/latency if those phases are marked complete.

    Returns:
        Dict with ``status``, ``cfg``, ``model``, ``tokenizer``, ``skip_phase0``,
        ``skip_phase1``, and cached summaries when available.
    """
    cfg = config or default_config()
    cfg.ensure_directories()
    set_seed(cfg.seed)

    status = print_progress_dashboard(cfg)
    skip0 = should_skip_phase("0", config=cfg)
    skip1 = should_skip_phase("1", config=cfg)

    phase0_summary = None
    phase1_summary = None
    if skip0:
        try:
            phase0_summary = read_json("phase0_summary.json", cfg)
            print_kv("Loaded Phase 0 summary", "results/phase0_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 0 marked complete but summary JSON missing")
            skip0 = False
    if skip1:
        try:
            phase1_summary = read_json("phase1_summary.json", cfg)
            print_kv("Loaded Phase 1 summary", "results/phase1_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 1 marked complete but summary JSON missing")
            skip1 = False

    # Decide whether we need weights in memory for the *next* work.
    next_needs_model = not is_phase_complete("1", cfg) or force_reload_model
    # Future phases 2+ will also need the model; if 0 and 1 done, still load for phase 2+.
    if is_phase_complete("0", cfg) and is_phase_complete("1", cfg):
        # If everything through 1 is done, next phase (2+) needs model unless all done.
        planned = ["2", "3", "4", "5", "6", "7", "8"]
        next_needs_model = any(not is_phase_complete(p, cfg) for p in planned)

    if model is not None and tokenizer is not None and not force_reload_model:
        print_section("Reusing in-memory model from this runtime")
    elif load_model_if_needed and next_needs_model:
        print_section("Loading model from Hugging Face (weights not stored in git)")
        model, tokenizer = load_model_and_tokenizer(config=cfg)
    else:
        print_section("Model load skipped (not required yet / already complete)")
        model, tokenizer = model, tokenizer

    print_git_save_instructions()

    return {
        "status": status,
        "cfg": cfg,
        "model": model,
        "tokenizer": tokenizer,
        "skip_phase0": skip0,
        "skip_phase1": skip1,
        "phase0_summary": phase0_summary,
        "phase1_summary": phase1_summary,
        "force_reload_model": force_reload_model,
    }
