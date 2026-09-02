"""Turning a volume into the model's input: slice centres, 2.5D triplets, one constant-area resize."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..data.slice_selection import slice_centres
from .crop_policy import CROP_FRACTION
from .dicom_io import _normalise_volume


PADDING_MODE = "reflect"


RESIZE_POLICY = "constant_area_aspect_rectangular"


STRIDE_ALIGNMENT = 32


SQUARE_IMAGE_SIZE = 448


def native_centre_crop(triplets: np.ndarray, fraction: float) -> np.ndarray:
    x = np.asarray(triplets)
    if x.ndim != 4:
        raise ValueError(f"B37 expected [S,C,H,W], got {x.shape}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    crop_h = max(2, min(h, int(round(h * float(fraction)))))
    crop_w = max(2, min(w, int(round(w * float(fraction)))))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return x[..., top : top + crop_h, left : left + crop_w]


REFERENCE_SIDE = SQUARE_IMAGE_SIZE


REFERENCE_AREA = REFERENCE_SIDE * REFERENCE_SIDE


def prepare_square_triplets(
    raw: np.ndarray,
    *,
    image_size: int = SQUARE_IMAGE_SIZE,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = CROP_FRACTION,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return 32 B36-style triplets after one native-crop -> 448 resize.

    Percentile normalization intentionally uses the complete native volume before
    cropping, preserving the historical B20 normalization support while avoiding
    B20's resize->crop->resize sequence.
    """
    if int(image_size) != SQUARE_IMAGE_SIZE:
        raise ValueError(f"B37 output size must remain {SQUARE_IMAGE_SIZE}")
    if int(gap) < 1:
        raise ValueError("B37 2.5D gap must be positive")

    normalized = _normalise_volume(raw)
    centers, position = slice_centres(
        len(normalized),
        gap=int(gap),
        center_offset=int(center_offset),
    )
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    index = np.clip(
        centers[:, None] + offsets[None, :],
        0,
        len(normalized) - 1,
    )
    triplets = normalized[index].astype(np.float32, copy=False)
    cropped = native_centre_crop(triplets, float(crop_fraction))
    tensor = torch.from_numpy(np.ascontiguousarray(cropped))
    resized = F.interpolate(
        tensor,
        size=(SQUARE_IMAGE_SIZE, SQUARE_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized, position


def constant_area_shape(
    height: int,
    width: int,
    *,
    reference_area: int = REFERENCE_AREA,
    alignment: int = STRIDE_ALIGNMENT,
) -> dict[str, int | float]:
    """Return isotropic resized and minimally stride-aligned rectangular geometry."""
    h, w = int(height), int(width)
    area = int(reference_area)
    stride = int(alignment)
    if h < 1 or w < 1:
        raise ValueError("B42 requires non-empty in-plane dimensions")
    if area < 4:
        raise ValueError("B42 reference area must be positive")
    if stride < 1:
        raise ValueError("B42 alignment must be positive")
    scale = math.sqrt(float(area) / float(h * w))
    resized_h = max(2, int(round(h * scale)))
    resized_w = max(2, int(round(w * scale)))
    aligned_h = int(math.ceil(resized_h / stride) * stride)
    aligned_w = int(math.ceil(resized_w / stride) * stride)
    pad_h = aligned_h - resized_h
    pad_w = aligned_w - resized_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return {
        "source_height": h,
        "source_width": w,
        "scale": float(scale),
        "resized_height": resized_h,
        "resized_width": resized_w,
        "aligned_height": aligned_h,
        "aligned_width": aligned_w,
        "pad_top": top,
        "pad_bottom": bottom,
        "pad_left": left,
        "pad_right": right,
        "anatomical_pixels": int(resized_h * resized_w),
        "tensor_pixels": int(aligned_h * aligned_w),
    }


def resize_to_constant_area(
    triplets: np.ndarray,
    *,
    reference_area: int = REFERENCE_AREA,
    alignment: int = STRIDE_ALIGNMENT,
) -> torch.Tensor:
    """Resize [S,C,H,W] once with one scale and reflection-pad only to stride."""
    x = np.asarray(triplets, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"B42 expected [S,C,H,W], got {x.shape}")
    if int(x.shape[1]) != 3:
        raise ValueError("B42 requires three-channel 2.5D triplets")
    geometry = constant_area_shape(
        int(x.shape[-2]),
        int(x.shape[-1]),
        reference_area=int(reference_area),
        alignment=int(alignment),
    )
    tensor = torch.from_numpy(np.ascontiguousarray(x))
    resized = F.interpolate(
        tensor,
        size=(int(geometry["resized_height"]), int(geometry["resized_width"])),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    padding = (
        int(geometry["pad_left"]),
        int(geometry["pad_right"]),
        int(geometry["pad_top"]),
        int(geometry["pad_bottom"]),
    )
    if any(padding):
        # Reflection padding needs every per-side pad to be smaller than the input axis.
        if max(padding[:2]) >= resized.shape[-1] or max(padding[2:]) >= resized.shape[-2]:
            raise ValueError("B42 reflection padding is invalid for this resized geometry")
        resized = F.pad(resized, padding, mode=PADDING_MODE)
    return resized


def prepare_series_triplets(
    raw: np.ndarray,
    *,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = CROP_FRACTION,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return 32 native-aspect B42 triplets at approximately constant pixel area."""
    return series_triplets_from_normalised(
        _normalise_volume(raw),
        gap=gap,
        center_offset=center_offset,
        crop_fraction=crop_fraction,
    )


def series_triplets_from_normalised(
    normalized: np.ndarray,
    *,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = CROP_FRACTION,
) -> tuple[torch.Tensor, np.ndarray]:
    """One view, from a volume that has already been normalised.

    `prepare_series_triplets` is this plus the normalisation, and calls it, so
    the two cannot drift. Splitting them lets a caller normalise a series once
    and then build each test-time view from that -- which matters at inference,
    where three views of a fourteen-series study held at once is roughly 3.2 GiB
    of resized pixels, against about 0.6 for the normalised volumes they come
    from.
    """
    if int(gap) < 1:
        raise ValueError("B42 2.5D gap must be positive")
    centers, position = slice_centres(
        len(normalized), gap=int(gap), center_offset=int(center_offset)
    )
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    index = np.clip(centers[:, None] + offsets[None, :], 0, len(normalized) - 1)
    triplets = normalized[index].astype(np.float32, copy=False)
    cropped = native_centre_crop(triplets, float(crop_fraction))
    images = resize_to_constant_area(cropped)
    return images, position

