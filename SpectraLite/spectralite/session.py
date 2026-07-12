"""Session restore helpers for new Colab runtimes."""

from __future__ import annotations

from typing import Any, Optional

from spectralite.artifacts import (
    is_phase_complete,
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
    """Restore experiment context after ``git pull`` on a fresh runtime."""
    cfg = config or default_config()
    cfg.ensure_directories()
    set_seed(cfg.seed)

    status = print_progress_dashboard(cfg)
    skip0 = should_skip_phase("0", config=cfg)
    skip1 = should_skip_phase("1", config=cfg)
    skip2 = should_skip_phase("2", config=cfg)
    skip3 = should_skip_phase("3", config=cfg)
    skip4 = should_skip_phase("4", config=cfg)
    skip5 = should_skip_phase("5", config=cfg)
    skip6 = should_skip_phase("6", config=cfg)
    skip7 = should_skip_phase("7", config=cfg)
    skip8 = should_skip_phase("8", config=cfg)

    phase0_summary = None
    phase1_summary = None
    phase2_summary = None
    phase3_summary = None
    phase4_summary = None
    phase5_summary = None
    phase6_summary = None
    phase7_summary = None
    phase8_summary = None
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
    if skip2:
        try:
            phase2_summary = read_json("phase2_summary.json", cfg)
            print_kv("Loaded Phase 2 summary", "results/phase2_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 2 marked complete but summary JSON missing")
            skip2 = False
    if skip3:
        try:
            phase3_summary = read_json("phase3_summary.json", cfg)
            print_kv("Loaded Phase 3 summary", "results/phase3_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 3 marked complete but summary JSON missing")
            skip3 = False
    if skip4:
        try:
            phase4_summary = read_json("phase4_summary.json", cfg)
            print_kv("Loaded Phase 4 summary", "results/phase4_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 4 marked complete but summary JSON missing")
            skip4 = False
    if skip5:
        try:
            phase5_summary = read_json("phase5_summary.json", cfg)
            print_kv("Loaded Phase 5 summary", "results/phase5_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 5 marked complete but summary JSON missing")
            skip5 = False
    if skip6:
        try:
            phase6_summary = read_json("phase6_summary.json", cfg)
            print_kv("Loaded Phase 6 summary", "results/phase6_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 6 marked complete but summary JSON missing")
            skip6 = False
    if skip7:
        try:
            phase7_summary = read_json("phase7_summary.json", cfg)
            print_kv("Loaded Phase 7 summary", "results/phase7_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 7 marked complete but summary JSON missing")
            skip7 = False
    if skip8:
        try:
            phase8_summary = read_json("phase8_summary.json", cfg)
            print_kv("Loaded Phase 8 summary", "results/phase8_summary.json")
        except FileNotFoundError:
            logger.warning("Phase 8 marked complete but summary JSON missing")
            skip8 = False

    planned = ["1", "2", "3", "4", "5", "6", "7", "8"]
    next_needs_model = force_reload_model or any(
        not is_phase_complete(p, cfg) for p in planned
    )

    if model is not None and tokenizer is not None and not force_reload_model:
        print_section("Reusing in-memory model from this runtime")
    elif load_model_if_needed and next_needs_model:
        print_section("Loading model from Hugging Face (weights not stored in git)")
        model, tokenizer = load_model_and_tokenizer(config=cfg)
    else:
        print_section("Model load skipped (not required yet / already complete)")

    print_git_save_instructions()

    return {
        "status": status,
        "cfg": cfg,
        "model": model,
        "tokenizer": tokenizer,
        "skip_phase0": skip0,
        "skip_phase1": skip1,
        "skip_phase2": skip2,
        "skip_phase3": skip3,
        "skip_phase4": skip4,
        "skip_phase5": skip5,
        "skip_phase6": skip6,
        "skip_phase7": skip7,
        "skip_phase8": skip8,
        "phase0_summary": phase0_summary,
        "phase1_summary": phase1_summary,
        "phase2_summary": phase2_summary,
        "phase3_summary": phase3_summary,
        "phase4_summary": phase4_summary,
        "phase5_summary": phase5_summary,
        "phase6_summary": phase6_summary,
        "phase7_summary": phase7_summary,
        "phase8_summary": phase8_summary,
        "force_reload_model": force_reload_model,
    }
