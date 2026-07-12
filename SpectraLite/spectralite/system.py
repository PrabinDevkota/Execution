"""System and runtime environment introspection for SpectraLite."""

from __future__ import annotations

import platform
import sys
from typing import Any, Optional

from spectralite.utils import format_bytes, print_kv, print_section


def python_version() -> str:
    """Return the running Python version string."""
    return platform.python_version()


def pytorch_version() -> str:
    """Return the installed PyTorch version, or ``unavailable``."""
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "unavailable"


def cuda_available() -> bool:
    """Return whether PyTorch can see a CUDA device."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def cuda_version() -> str:
    """Return the CUDA version reported by PyTorch, if any."""
    try:
        import torch

        if torch.version.cuda is None:
            return "n/a (CPU build or CUDA unavailable)"
        return str(torch.version.cuda)
    except ImportError:
        return "unavailable"


def cudnn_version() -> str:
    """Return the cuDNN version reported by PyTorch, if any."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "n/a"
        version = torch.backends.cudnn.version()
        return str(version) if version is not None else "n/a"
    except Exception:
        return "n/a"


def gpu_name(device_index: int = 0) -> str:
    """Return the CUDA device name for ``device_index``.

    Args:
        device_index: CUDA ordinal.

    Returns:
        GPU marketing name, or a CPU / unavailable message.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "CPU (CUDA unavailable)"
        if device_index >= torch.cuda.device_count():
            return f"invalid device index {device_index}"
        return torch.cuda.get_device_name(device_index)
    except ImportError:
        return "unavailable"


def gpu_count() -> int:
    """Return the number of visible CUDA devices."""
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def gpu_memory_total_bytes(device_index: int = 0) -> Optional[int]:
    """Return total GPU memory in bytes for ``device_index``, or ``None``."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        return int(props.total_memory)
    except Exception:
        return None


def resolve_device() -> str:
    """Return ``cuda`` if available, otherwise ``cpu``."""
    return "cuda" if cuda_available() else "cpu"


def collect_environment_info(device_index: int = 0) -> dict[str, Any]:
    """Collect a structured environment report for logging / notebooks.

    Args:
        device_index: Primary CUDA device to describe.

    Returns:
        Dictionary of environment fields.
    """
    total = gpu_memory_total_bytes(device_index)
    return {
        "python_version": python_version(),
        "pytorch_version": pytorch_version(),
        "cuda_available": cuda_available(),
        "cuda_version": cuda_version(),
        "cudnn_version": cudnn_version(),
        "gpu_count": gpu_count(),
        "gpu_name": gpu_name(device_index),
        "gpu_memory_total": format_bytes(total) if total is not None else "n/a",
        "gpu_memory_total_bytes": total,
        "device": resolve_device(),
        "platform": platform.platform(),
        "processor": platform.processor() or "n/a",
        "executable": sys.executable,
    }


def print_environment_report(device_index: int = 0) -> dict[str, Any]:
    """Print and return the Phase 0 environment verification report.

    Args:
        device_index: Primary CUDA device to describe.

    Returns:
        Same dictionary produced by :func:`collect_environment_info`.
    """
    info = collect_environment_info(device_index=device_index)
    print_section("Environment Verification")
    print_kv("Python Version", info["python_version"])
    print_kv("PyTorch Version", info["pytorch_version"])
    print_kv("CUDA Version", info["cuda_version"])
    print_kv("cuDNN Version", info["cudnn_version"])
    print_kv("GPU Name", info["gpu_name"])
    print_kv("GPU Memory", info["gpu_memory_total"])
    print_kv("Torch CUDA Available", info["cuda_available"])
    print_kv("Device", info["device"])
    print_kv("GPU Count", info["gpu_count"])
    print_kv("Platform", info["platform"])
    return info
