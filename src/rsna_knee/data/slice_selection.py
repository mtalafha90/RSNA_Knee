"""Choosing which slices represent a series, and the position features that describe them."""
from __future__ import annotations

from dataclasses import dataclass
from torch import nn
import math
import numpy as np
import torch
import torch.nn.functional as F

from ..imaging.dicom_io import _centers


DENSE_SLICES = 32


BASE_SLICES = 16


def _extra_centers(base: np.ndarray, dense: np.ndarray, count: int) -> np.ndarray:
    """Pick deterministic dense-grid centres not already used by the base path."""
    base_list = [int(x) for x in np.asarray(base).reshape(-1)]
    dense_list = [int(x) for x in np.asarray(dense).reshape(-1)]
    used = set(base_list)
    extras = [x for x in dense_list if x not in used]
    if len(extras) < count:
        # Short series can quantize several requested centres to the same frame.
        # Repetition is preferable to inventing coordinates outside the scan.
        for x in dense_list:
            extras.append(x)
            if len(extras) >= count:
                break
    while len(extras) < count:
        extras.append(base_list[len(extras) % len(base_list)])
    return np.asarray(extras[:count], dtype=np.int64)


def _position_basis(position: torch.Tensor) -> torch.Tensor:
    """Deterministic 8-D continuous through-plane coordinate basis."""
    z = position.float().clamp(0.0, 1.0)
    return torch.stack(
        [
            z,
            z.square(),
            torch.sin(math.pi * z),
            torch.cos(math.pi * z),
            torch.sin(2.0 * math.pi * z),
            torch.cos(2.0 * math.pi * z),
            torch.sin(4.0 * math.pi * z),
            torch.cos(4.0 * math.pi * z),
        ],
        dim=-1,
    )


POSITION_BASIS = 8


DEFAULT_GRID_SIZE = 3


TOKEN_DROPOUT = 0.05


def slice_centres(
    n_frames: int,
    *,
    gap: int = 1,
    center_offset: int = 0,
    base_slices: int = BASE_SLICES,
    dense_slices: int = DENSE_SLICES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 32 centres whose first 16 exactly reproduce the B34 centres."""
    if dense_slices < base_slices:
        raise ValueError("dense_slices must be >= base_slices")
    base = _centers(
        n_frames,
        base_slices,
        gap,
        center_offset=center_offset,
        jitter=0,
    )
    dense = _centers(
        n_frames,
        dense_slices,
        gap,
        center_offset=center_offset,
        jitter=0,
    )
    extras = _extra_centers(base, dense, dense_slices - base_slices)
    combined = np.concatenate([base, extras]).astype(np.int64, copy=False)
    denom = float(max(n_frames - 1, 1))
    normalized_position = combined.astype(np.float32) / denom
    return combined, normalized_position

