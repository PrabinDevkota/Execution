"""GPU memory helpers for SpectraLite experiments."""

from __future__ import annotations

from typing import Any, Optional

from spectralite.utils import format_bytes, print_kv, print_section


def is_cuda_available() -> bool:
    """Return whether CUDA is available through PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def synchronize(device_index: int = 0) -> None:
    """Synchronize the selected CUDA device (no-op on CPU).

    Args:
        device_index: CUDA ordinal to synchronize.
    """
    if not is_cuda_available():
        return
    import torch

    torch.cuda.synchronize(device_index)


def memory_allocated_bytes(device_index: int = 0) -> int:
    """Return bytes currently allocated by PyTorch on the device.

    Args:
        device_index: CUDA ordinal.

    Returns:
        Allocated bytes, or ``0`` if CUDA is unavailable.
    """
    if not is_cuda_available():
        return 0
    import torch

    return int(torch.cuda.memory_allocated(device_index))


def memory_reserved_bytes(device_index: int = 0) -> int:
    """Return bytes reserved by the PyTorch caching allocator.

    Args:
        device_index: CUDA ordinal.

    Returns:
        Reserved bytes, or ``0`` if CUDA is unavailable.
    """
    if not is_cuda_available():
        return 0
    import torch

    return int(torch.cuda.memory_reserved(device_index))


def memory_total_bytes(device_index: int = 0) -> Optional[int]:
    """Return total physical GPU memory in bytes.

    Args:
        device_index: CUDA ordinal.

    Returns:
        Total bytes, or ``None`` if CUDA is unavailable.
    """
    if not is_cuda_available():
        return None
    import torch

    return int(torch.cuda.get_device_properties(device_index).total_memory)


def memory_free_bytes(device_index: int = 0) -> Optional[int]:
    """Estimate free GPU memory as ``total - reserved``.

    Args:
        device_index: CUDA ordinal.

    Returns:
        Approximate free bytes, or ``None`` if CUDA is unavailable.
    """
    total = memory_total_bytes(device_index)
    if total is None:
        return None
    return int(total - memory_reserved_bytes(device_index))


def empty_cache(device_index: int = 0) -> None:
    """Release unused cached blocks back to the driver (best-effort).

    Args:
        device_index: CUDA ordinal (used only to gate availability).
    """
    if not is_cuda_available():
        return
    import torch

    # device_index retained for API symmetry with other helpers.
    _ = device_index
    torch.cuda.empty_cache()


def snapshot(device_index: int = 0, label: str = "") -> dict[str, Any]:
    """Capture a structured GPU memory snapshot.

    Args:
        device_index: CUDA ordinal.
        label: Optional human-readable stage name (e.g. ``after_load``).

    Returns:
        Dictionary with allocated / reserved / free / total fields.
    """
    if is_cuda_available():
        synchronize(device_index)

    total = memory_total_bytes(device_index)
    allocated = memory_allocated_bytes(device_index)
    reserved = memory_reserved_bytes(device_index)
    free = memory_free_bytes(device_index)

    return {
        "label": label,
        "cuda_available": is_cuda_available(),
        "device_index": device_index,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "free_bytes": free,
        "total_bytes": total,
        "allocated": format_bytes(allocated),
        "reserved": format_bytes(reserved),
        "free": format_bytes(free) if free is not None else "n/a",
        "total": format_bytes(total) if total is not None else "n/a",
    }


def print_memory_snapshot(
    snap: Optional[dict[str, Any]] = None,
    *,
    device_index: int = 0,
    label: str = "GPU Memory",
) -> dict[str, Any]:
    """Print a GPU memory snapshot and return it.

    Args:
        snap: Existing snapshot from :func:`snapshot`. If ``None``, a fresh
            snapshot is taken.
        device_index: CUDA ordinal used when ``snap`` is ``None``.
        label: Section title / snapshot label.

    Returns:
        The snapshot dictionary that was printed.
    """
    if snap is None:
        snap = snapshot(device_index=device_index, label=label)

    title = snap.get("label") or label
    print_section(str(title))
    if not snap["cuda_available"]:
        print_kv("Status", "CUDA unavailable — CPU mode")
        return snap

    print_kv("Allocated", snap["allocated"])
    print_kv("Reserved", snap["reserved"])
    print_kv("Free (approx)", snap["free"])
    print_kv("Total", snap["total"])
    return snap


def print_memory_timeline(snapshots: list[dict[str, Any]]) -> None:
    """Print a multi-stage GPU memory comparison table.

    Args:
        snapshots: Ordered snapshots (e.g. before load / after load / after
            inference).
    """
    print_section("GPU Memory Timeline")
    if not snapshots:
        print("  (no snapshots)")
        return

    if not snapshots[0].get("cuda_available", False):
        print_kv("Status", "CUDA unavailable — CPU mode")
        for snap in snapshots:
            print_kv(str(snap.get("label", "stage")), "n/a")
        return

    for snap in snapshots:
        label = str(snap.get("label", "stage"))
        print(
            f"  {label:<28} "
            f"alloc={snap['allocated']:<12} "
            f"reserved={snap['reserved']:<12} "
            f"free={snap['free']}"
        )
