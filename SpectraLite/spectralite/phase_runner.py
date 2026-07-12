"""Self-contained helpers so any phase cell can run alone on Colab."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from spectralite.utils import get_logger, print_section

logger = get_logger(__name__)

DEFAULT_REPO = "https://github.com/PrabinDevkota/Execution.git"
DEFAULT_ROOT = Path("/content/Execution")
DEFAULT_PKG = DEFAULT_ROOT / "SpectraLite"


def sync_repo_and_imports(
    *,
    root: Path = DEFAULT_ROOT,
    repo_url: str = DEFAULT_REPO,
    hard_reset: bool = True,
) -> Path:
    """Clone/fetch latest GitHub code and put SpectraLite on ``sys.path``.

    On Colab, local-only commits often block ``git pull --ff-only``. When
    ``hard_reset=True``, reset to ``origin/main`` (safe: results live on GitHub).
    """
    pkg = root / "SpectraLite"
    if not (pkg / "spectralite").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", repo_url, str(root)])
    else:
        subprocess.check_call(["git", "-C", str(root), "fetch", "origin"])
        if hard_reset:
            subprocess.check_call(
                ["git", "-C", str(root), "reset", "--hard", "origin/main"]
            )
        else:
            try:
                subprocess.check_call(
                    ["git", "-C", str(root), "pull", "--ff-only", "origin", "main"]
                )
            except subprocess.CalledProcessError:
                logger.warning("ff-only pull failed; hard-resetting to origin/main")
                subprocess.check_call(
                    ["git", "-C", str(root), "reset", "--hard", "origin/main"]
                )

    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    # Drop cached spectralite modules so newly pulled files import cleanly
    for name in list(sys.modules):
        if name == "spectralite" or name.startswith("spectralite."):
            del sys.modules[name]

    print_section("Repo synced")
    print(f"  PACKAGE_ROOT = {pkg}")
    print(f"  phase2.py exists = {(pkg / 'spectralite' / 'phase2.py').is_file()}")
    return pkg


def ensure_model_tokenizer(
    cfg: Any,
    model: Any = None,
    tokenizer: Any = None,
) -> tuple[Any, Any]:
    """Return in-memory model/tokenizer or load from Hugging Face."""
    from spectralite.model_loader import load_model_and_tokenizer

    if model is not None and tokenizer is not None:
        print_section("Using in-memory model")
        try:
            print("  device:", next(model.parameters()).device)
        except Exception:  # noqa: BLE001
            pass
        return model, tokenizer

    print_section("Loading model from Hugging Face")
    return load_model_and_tokenizer(config=cfg)


def bootstrap_phase(
    *,
    hard_reset: bool = True,
    install_deps: bool = False,
    require_cuda: bool = True,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """One-shot: sync git → (optional) deps → config → model.

    Call this at the top of every phase cell so that phase can run alone.
    """
    pkg = sync_repo_and_imports(hard_reset=hard_reset)

    from spectralite import default_config, set_seed
    from spectralite.colab_setup import ensure_environment, in_colab
    from spectralite.artifacts import print_progress_dashboard

    if install_deps or in_colab():
        # On Colab always ensure Colab-safe deps; cheap if already installed
        ensure_environment(pull=False, install=True, require_cuda_on_colab=require_cuda)

    cfg = default_config()
    cfg.ensure_directories()
    set_seed(cfg.seed)
    print_progress_dashboard(cfg)

    model, tokenizer = ensure_model_tokenizer(cfg, model=model, tokenizer=tokenizer)
    return {
        "pkg": pkg,
        "cfg": cfg,
        "model": model,
        "tokenizer": tokenizer,
    }
