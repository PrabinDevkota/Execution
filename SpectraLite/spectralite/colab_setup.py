"""One-shot Colab / environment bootstrap for SpectraLite.

Use from any phase notebook::

    from spectralite.colab_setup import ensure_environment
    ensure_environment()

Or run the notebook ``notebooks/Colab_Bootstrap.ipynb`` once per new runtime.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_URL_DEFAULT = "https://github.com/PrabinDevkota/Execution.git"
CLONE_ROOT_DEFAULT = Path("/content/Execution")
PACKAGE_NAME = "SpectraLite"


def in_colab() -> bool:
    """Return True when running inside Google Colab."""
    return "google.colab" in sys.modules


def find_repo_root(
    extra: Optional[list[Path]] = None,
    *,
    clone_root: Path = CLONE_ROOT_DEFAULT,
) -> Path:
    """Locate the SpectraLite package root (directory that contains ``spectralite/``)."""
    candidates: list[Path] = []
    if extra:
        candidates.extend(extra)
    cwd = Path.cwd().resolve()
    candidates.extend(
        [
            cwd,
            cwd.parent,
            clone_root / PACKAGE_NAME,
            Path(__file__).resolve().parent.parent,
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "spectralite").is_dir() and (
            (candidate / "requirements-colab.txt").is_file()
            or (candidate / "requirements.txt").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "SpectraLite repo root not found. On Colab, run ensure_environment() "
        "or notebooks/Colab_Bootstrap.ipynb first."
    )


def add_to_syspath(repo_root: Path) -> None:
    """Prepend ``repo_root`` to ``sys.path`` if missing."""
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def clone_or_pull(
    *,
    repo_url: str = REPO_URL_DEFAULT,
    clone_root: Path = CLONE_ROOT_DEFAULT,
    pull: bool = True,
) -> Path:
    """Clone the GitHub repo on Colab, or fast-forward pull if it already exists.

    Returns:
        Path to the SpectraLite package root.
    """
    package_root = clone_root / PACKAGE_NAME
    if not package_root.is_dir():
        clone_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", repo_url, str(clone_root)])
    elif pull:
        subprocess.check_call(
            ["git", "-C", str(clone_root), "pull", "--ff-only"],
        )
    return package_root


def install_dependencies(repo_root: Path, *, colab: Optional[bool] = None) -> Path:
    """Install the correct requirements file for this environment.

    On Colab, uses ``requirements-colab.txt`` (no torch) to preserve CUDA wheels.
    Locally, uses ``requirements.txt``.

    Returns:
        Path of the requirements file that was installed.
    """
    use_colab = in_colab() if colab is None else colab
    req_name = "requirements-colab.txt" if use_colab else "requirements.txt"
    requirements = repo_root / req_name
    if not requirements.is_file():
        raise FileNotFoundError(f"Missing dependency file: {requirements}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
    )
    return requirements


def verify_torch(*, require_cuda_on_colab: bool = True) -> dict[str, object]:
    """Import torch and report CUDA status.

    Args:
        require_cuda_on_colab: If True and running on Colab without CUDA, raise.

    Returns:
        Dict with version / cuda flags.
    """
    import torch

    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
    }
    if in_colab() and require_cuda_on_colab and not info["cuda_available"]:
        raise RuntimeError(
            "Colab CUDA is not available. "
            "Runtime → Change runtime type → GPU (A100) → "
            "Disconnect and delete runtime → reconnect → re-run bootstrap."
        )
    if in_colab() and "+cpu" in str(info["torch_version"]).lower():
        raise RuntimeError(
            "CPU-only torch detected on Colab. "
            "Delete the runtime and re-run bootstrap without installing torch."
        )
    return info


def ensure_environment(
    *,
    repo_url: str = REPO_URL_DEFAULT,
    clone_root: Path = CLONE_ROOT_DEFAULT,
    pull: bool = True,
    install: bool = True,
    require_cuda_on_colab: bool = True,
) -> Path:
    """Full bootstrap: clone/pull → sys.path → deps → torch/CUDA check.

    Returns:
        SpectraLite package root path.
    """
    if in_colab():
        repo_root = clone_or_pull(repo_url=repo_url, clone_root=clone_root, pull=pull)
    else:
        repo_root = find_repo_root(clone_root=clone_root)

    add_to_syspath(repo_root)

    req_path = None
    if install:
        req_path = install_dependencies(repo_root)

    torch_info = verify_torch(require_cuda_on_colab=require_cuda_on_colab)

    print("SpectraLite environment ready")
    print(f"  repo_root      : {repo_root}")
    print(f"  requirements   : {req_path}")
    print(f"  torch          : {torch_info['torch_version']}")
    print(f"  cuda_available : {torch_info['cuda_available']}")
    print(f"  gpu            : {torch_info['gpu_name']}")
    return repo_root
