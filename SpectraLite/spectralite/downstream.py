"""lm-eval downstream evaluation helpers (Phase 8)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from spectralite.utils import get_logger, print_kv, print_section

logger = get_logger(__name__)

# Plan §8 core zero-shot suite (OPT-125M friendly).
DEFAULT_ZERO_SHOT_TASKS: tuple[str, ...] = (
    "piqa",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "boolq",
    "openbookqa",
)


def _pick_acc(metrics: dict[str, Any]) -> Optional[float]:
    """Prefer normalized accuracy when present."""
    for key in (
        "acc_norm,none",
        "acc_norm",
        "acc,none",
        "acc",
        "exact_match,none",
        "exact_match",
    ):
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    # Fallback: first numeric-looking value
    for k, v in metrics.items():
        if k.endswith("_stderr") or k.endswith(",stderr"):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def wrap_hf_lm(
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int | str = 8,
) -> Any:
    """Wrap an in-memory HF causal LM for lm-eval."""
    from lm_eval.models.huggingface import HFLM

    # Newer lm-eval accepts the live module via pretrained=
    try:
        return HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    except TypeError:
        return HFLM(model, tokenizer=tokenizer, batch_size=batch_size)


def run_lm_eval(
    model: Any,
    tokenizer: Any,
    *,
    tasks: Sequence[str] = DEFAULT_ZERO_SHOT_TASKS,
    num_fewshot: int = 0,
    batch_size: int | str = 8,
    limit: Optional[int | float] = None,
    method: str = "model",
) -> dict[str, Any]:
    """Run EleutherAI lm-eval zero-shot suite; return compact metrics dict."""
    from lm_eval import simple_evaluate

    print_section(f"lm-eval — {method}")
    print_kv("Tasks", ", ".join(tasks))
    print_kv("Few-shot", num_fewshot)
    print_kv("Batch size", batch_size)
    if limit is not None:
        print_kv("Limit", limit)

    lm = wrap_hf_lm(model, tokenizer, batch_size=batch_size)
    raw = simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
        bootstrap_iters=0,
    )
    task_results = raw.get("results", {}) if isinstance(raw, dict) else {}
    per_task: dict[str, Any] = {}
    scores: list[float] = []
    for task in tasks:
        metrics = task_results.get(task, {}) or {}
        acc = _pick_acc(metrics)
        per_task[task] = {
            "acc": acc,
            "raw": {k: v for k, v in metrics.items() if not str(k).endswith("stderr")},
        }
        if acc is not None:
            scores.append(acc)
            print_kv(f"{task}", f"{acc:.4f}")
        else:
            print_kv(f"{task}", "n/a")

    avg = float(sum(scores) / len(scores)) if scores else float("nan")
    print_kv("zero_shot_avg", f"{avg:.4f}" if scores else "n/a")
    return {
        "method": method,
        "tasks": list(tasks),
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "limit": limit,
        "per_task": per_task,
        "zero_shot_avg": avg,
    }
