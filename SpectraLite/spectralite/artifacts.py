"""Persistent, git-friendly experiment artifacts for SpectraLite phases.

Design goals
------------
* Every completed phase writes small JSON/CSV under ``results/`` (committed to git).
* A new Colab runtime can ``git pull`` and **skip recomputing** finished metrics.
* Full model weights are **not** stored in git (too large); they reload from Hugging Face.
* ``phase_status.json`` is the source of truth for what is done.

Typical layout::

    results/
      phase_status.json
      phase0_summary.json
      phase0_linear_layers.json
      phase1_dense_baselines.csv
      phase1_summary.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from spectralite.config import Config, default_config
from spectralite.utils import get_logger, print_kv, print_section

logger = get_logger(__name__)

STATUS_FILENAME = "phase_status.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def results_dir(config: Optional[Config] = None) -> Path:
    cfg = config or default_config()
    path = Path(cfg.results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def status_path(config: Optional[Config] = None) -> Path:
    return results_dir(config) / STATUS_FILENAME


def load_status(config: Optional[Config] = None) -> dict[str, Any]:
    """Load ``phase_status.json`` (empty scaffold if missing)."""
    path = status_path(config)
    if not path.exists():
        return {
            "project": "SpectraLite",
            "updated_at": None,
            "phases": {},
        }
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_status(status: dict[str, Any], config: Optional[Config] = None) -> Path:
    """Write ``phase_status.json`` atomically-ish via replace."""
    path = status_path(config)
    status = dict(status)
    status["updated_at"] = _utc_now()
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    tmp.replace(path)
    return path


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def write_json(path: Path | str, payload: Any) -> Path:
    """Write a JSON artifact under results (or absolute path)."""
    path = Path(path)
    if not path.is_absolute():
        path = results_dir() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    return path


def read_json(path: Path | str, config: Optional[Config] = None) -> Any:
    """Read a JSON artifact; relative paths resolve under ``results/``."""
    path = Path(path)
    if not path.is_absolute():
        path = results_dir(config) / path
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def mark_phase_complete(
    phase: str,
    *,
    artifacts: Optional[dict[str, str]] = None,
    metrics: Optional[dict[str, Any]] = None,
    notes: str = "",
    config: Optional[Config] = None,
) -> dict[str, Any]:
    """Record a phase as complete in ``phase_status.json``.

    Args:
        phase: e.g. ``\"0\"``, ``\"1\"``, ``\"2\"``.
        artifacts: Map of logical name → relative path under repo/results.
        metrics: Small summary metrics for quick display.
        notes: Free-text note.
        config: Optional config for results directory.
    """
    status = load_status(config)
    phases = status.setdefault("phases", {})
    phases[str(phase)] = {
        "complete": True,
        "completed_at": _utc_now(),
        "artifacts": artifacts or {},
        "metrics": metrics or {},
        "notes": notes,
    }
    save_status(status, config)
    logger.info("Marked phase %s complete", phase)
    return status


def is_phase_complete(phase: str, config: Optional[Config] = None) -> bool:
    """Return True if ``phase_status.json`` marks the phase complete."""
    status = load_status(config)
    entry = status.get("phases", {}).get(str(phase), {})
    return bool(entry.get("complete"))


def get_phase_entry(phase: str, config: Optional[Config] = None) -> dict[str, Any]:
    """Return the status dict for a phase (possibly empty)."""
    return dict(load_status(config).get("phases", {}).get(str(phase), {}))


def clear_phase(phase: str, config: Optional[Config] = None) -> dict[str, Any]:
    """Mark a phase incomplete (does not delete artifact files)."""
    status = load_status(config)
    phases = status.setdefault("phases", {})
    if str(phase) in phases:
        phases[str(phase)]["complete"] = False
        phases[str(phase)]["cleared_at"] = _utc_now()
    save_status(status, config)
    return status


def print_progress_dashboard(config: Optional[Config] = None) -> dict[str, Any]:
    """Pretty-print which phases are done and where artifacts live."""
    status = load_status(config)
    print_section("SpectraLite progress (git-tracked artifacts)")
    print_kv("Status file", str(status_path(config)))
    print_kv("Updated at", status.get("updated_at") or "never")
    phases = status.get("phases", {})
    planned = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    for p in planned:
        entry = phases.get(p, {})
        done = bool(entry.get("complete"))
        mark = "DONE" if done else "todo"
        when = entry.get("completed_at", "")
        print(f"  Phase {p:<2} [{mark}]  {when}")
        if done and entry.get("artifacts"):
            for name, rel in entry["artifacts"].items():
                print(f"           - {name}: {rel}")
    next_phase = next((p for p in planned if not phases.get(p, {}).get("complete")), None)
    print_kv("Next phase to run", next_phase if next_phase is not None else "all complete")
    return status


def save_phase0_artifacts(
    *,
    env_info: dict[str, Any],
    load_summary: dict[str, Any],
    analysis: dict[str, Any],
    inference_ok: bool,
    config: Optional[Config] = None,
) -> dict[str, str]:
    """Persist Phase 0 summaries (not model weights) and mark complete."""
    cfg = config or default_config()
    # Linear layers → JSON-serializable
    linear_layers = []
    for item in analysis.get("linear_layers", []):
        if hasattr(item, "to_dict"):
            linear_layers.append(item.to_dict())
        elif isinstance(item, dict):
            linear_layers.append(item)
        else:
            linear_layers.append(dict(item))

    summary = {
        "phase": "0",
        "saved_at": _utc_now(),
        "model_name": load_summary.get("model_name") or cfg.model_name,
        "env": env_info,
        "load_summary": {
            k: load_summary[k]
            for k in (
                "model_name",
                "architecture",
                "total_parameters",
                "trainable_parameters",
                "model_size",
                "dtype",
                "device",
            )
            if k in load_summary
        },
        "analysis": {
            "num_transformer_blocks": analysis.get("num_transformer_blocks"),
            "num_linear_layers": analysis.get("num_linear_layers"),
            "num_attention_linear_layers": analysis.get("num_attention_linear_layers"),
            "num_mlp_linear_layers": analysis.get("num_mlp_linear_layers"),
            "total_parameters": analysis.get("total_parameters"),
            "dtype": analysis.get("dtype"),
            "device": analysis.get("device"),
        },
        "inference_ok": inference_ok,
    }
    p_summary = write_json("phase0_summary.json", summary)
    p_layers = write_json(
        "phase0_linear_layers.json",
        {"model_name": summary["model_name"], "layers": linear_layers},
    )

    # Relative paths for git / status
    artifacts = {
        "summary": "results/phase0_summary.json",
        "linear_layers": "results/phase0_linear_layers.json",
    }
    mark_phase_complete(
        "0",
        artifacts=artifacts,
        metrics={
            "total_parameters": analysis.get("total_parameters"),
            "num_linear_layers": analysis.get("num_linear_layers"),
            "device": str(analysis.get("device")),
            "inference_ok": inference_ok,
        },
        notes="Phase 0 smoke test. Model weights not saved; reload from HF next session.",
        config=cfg,
    )
    print_section("Phase 0 artifacts saved (git these files)")
    for k, v in artifacts.items():
        print_kv(k, v)
    print_kv("status", "results/phase_status.json")
    return artifacts


def save_phase1_artifacts(
    phase1_result: dict[str, Any],
    *,
    config: Optional[Config] = None,
) -> dict[str, str]:
    """Persist Phase 1 summary JSON + ensure CSV path recorded; mark complete."""
    cfg = config or default_config()
    row = phase1_result.get("row", {})
    summary = {
        "phase": "1",
        "saved_at": _utc_now(),
        "csv_path": phase1_result.get("csv_path"),
        "row": row,
        "prefill": phase1_result.get("prefill"),
        "decode": phase1_result.get("decode"),
        "ppl": phase1_result.get("ppl"),
        "flop_stats": {
            "empirical_flops_fwd": phase1_result.get("flop_stats", {}).get(
                "empirical_flops_fwd"
            ),
            "empirical_gflops_fwd": phase1_result.get("flop_stats", {}).get(
                "empirical_gflops_fwd"
            ),
        },
    }
    write_json("phase1_summary.json", summary)
    artifacts = {
        "summary": "results/phase1_summary.json",
        "csv": "results/phase1_dense_baselines.csv",
        "status": "results/phase_status.json",
    }
    mark_phase_complete(
        "1",
        artifacts=artifacts,
        metrics={
            "empirical_flops_fwd": row.get("empirical_flops_fwd"),
            "prefill_ms_mean": row.get("prefill_ms_mean"),
            "decode_ms_per_token_mean": row.get("decode_ms_per_token_mean"),
            "ppl_wikitext2": row.get("ppl_wikitext2"),
            "ppl_ptb": row.get("ppl_ptb"),
            "ppl_c4": row.get("ppl_c4"),
        },
        notes="Dense baseline. Skip re-run on new Colab unless FORCE_RERUN_PHASE1=True.",
        config=cfg,
    )
    print_section("Phase 1 artifacts saved (git these files)")
    for k, v in artifacts.items():
        print_kv(k, v)
    return artifacts


def should_skip_phase(phase: str, *, force: bool = False, config: Optional[Config] = None) -> bool:
    """Return True if phase is complete and force-rerun is not requested."""
    if force:
        return False
    return is_phase_complete(phase, config)


def print_git_save_instructions() -> None:
    """Print how to commit artifacts from Colab so the next session can skip work."""
    print_section("Save progress to GitHub (do this after each phase)")
    print(
        """
  In Colab (from SpectraLite root):

    %cd /content/Execution
    !git add SpectraLite/results/*.json SpectraLite/results/*.csv SpectraLite/notebooks/works.ipynb
    !git config user.email "you@example.com"
    !git config user.name "Your Name"
    !git commit -m "Record Phase N results"
    !git push

  Or: File → Save a copy in GitHub for the notebook, AND commit the results/ files.

  Next Colab session:
    1) git pull
    2) Run Stage 0 (deps)
    3) Run Session Restore — completed phases are skipped
    4) Continue from the next incomplete phase
""".strip(
            "\n"
        )
    )
