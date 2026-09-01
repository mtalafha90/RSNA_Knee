"""Keeping a long run inside its memory budget, and reporting what it used."""
from __future__ import annotations

from pathlib import Path
import ctypes
import gc
import torch


def _memory_state(runtime) -> dict[str, float]:
    """Return process, host and CUDA memory telemetry in GiB."""
    status: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"VmRSS", "VmHWM"}:
                status[key] = int(value.strip().split()[0])
    except (FileNotFoundError, OSError, ValueError):
        pass

    available_kib = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break
    except (FileNotFoundError, OSError, ValueError):
        pass

    gib = float(1024**2)
    result = {
        "rss_gib": float(status.get("VmRSS", 0)) / gib,
        "rss_peak_gib": float(status.get("VmHWM", 0)) / gib,
        "system_available_gib": float(available_kib) / gib,
        "cuda_allocated_gib": 0.0,
        "cuda_reserved_gib": 0.0,
        "cuda_peak_allocated_gib": 0.0,
        "cuda_peak_reserved_gib": 0.0,
    }
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        divisor = float(1024**3)
        result.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated(runtime.device)
                / divisor,
                "cuda_reserved_gib": torch.cuda.memory_reserved(runtime.device)
                / divisor,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(
                    runtime.device
                )
                / divisor,
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(
                    runtime.device
                )
                / divisor,
            }
        )
    return result


def _format_memory_state(state: dict[str, float]) -> str:
    return (
        f"rss={state['rss_gib']:.2f}GiB "
        f"rss_peak={state['rss_peak_gib']:.2f}GiB "
        f"host_available={state['system_available_gib']:.2f}GiB "
        f"cuda={state['cuda_allocated_gib']:.2f}/"
        f"{state['cuda_reserved_gib']:.2f}GiB "
        f"cuda_peak={state['cuda_peak_allocated_gib']:.2f}/"
        f"{state['cuda_peak_reserved_gib']:.2f}GiB"
    )


def _trim_host_memory() -> None:
    """Release unreachable Python objects and return free glibc arenas."""
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


WEIGHT_DECAY = 1e-4


HEAD_LR = 1e-4


GRAD_CLIP = 1.0

