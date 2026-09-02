"""Building one study's test-time views without holding all of them at once.

The training dataset materialises every TTA view of every series before the
model sees any of them: `[views, series, slices, 3, H, W]`. At the fourteen
series this data actually contains that is roughly 3.2 GiB of resized pixels for
one study, doubled when two shards run together.

That is a host-memory risk the archive diagnosed after B39 and B41 both passed
their visible notebooks and then died on the hidden set with no traceback. The
fix is to keep the normalised native volumes -- about a fifth of the size -- and
build one view from them at a time.

`series_triplets_from_normalised` is the same function `prepare_series_triplets`
calls after normalising, so a streamed view and a materialised one are the same
tensor, not an equivalent one.
"""
from __future__ import annotations

import numpy as np
import torch

from ..imaging.dicom_io import _normalise_volume, find_series_dir, read_dicom_series
from ..imaging.physical_scale import resample_volume_inplane
from ..imaging.triplets import series_triplets_from_normalised

PLANE_NAMES = {"sagittal": "Sagittal", "coronal": "Coronal", "axial": "Axial"}


def load_normalised_study(
    data_root, split: str, uid: str, records: list[dict], *, physical_scale_policy=None
) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    """Decode and normalise each of a study's series once, keeping only the arrays.

    A series that cannot be found or decoded is **dropped**, not fatal. A study
    with five readable series out of six is still a real prediction from five;
    ending the run over it would forfeit every other study as well.
    """
    normalised: list[np.ndarray] = []
    kept: list[dict] = []
    dropped: list[dict] = []

    for record in records:
        series_uid = str(record["series_uid"])
        path = find_series_dir(data_root, split, str(uid), series_uid)
        if path is None:
            dropped.append({"series_uid": series_uid, "error": "series directory not found"})
            continue
        try:
            if physical_scale_policy is None:
                raw = read_dicom_series(path)
            else:
                raw, stats = read_dicom_series(path, return_stats=True)
                raw, _ = resample_volume_inplane(
                    raw,
                    source_spacing_mm=stats.get("pixel_spacing_mm"),
                    plane=PLANE_NAMES[str(record["plane"]).lower()],
                    policy=physical_scale_policy,
                )
            normalised.append(np.asarray(_normalise_volume(raw), dtype=np.float32))
            kept.append(record)
            del raw
        except Exception as error:  # noqa: BLE001
            dropped.append(
                {"series_uid": series_uid, "error": f"{type(error).__name__}: {error}"}
            )
    return normalised, kept, dropped


def build_study_view(
    normalised: list[np.ndarray],
    records: list[dict],
    *,
    center_offset: int,
    gap: int,
    crop_fraction: float,
    device,
):
    """One complete study view, moved to the device as each series is built.

    The host never holds the whole view: each series goes to the device and its
    CPU copy is released before the next one is built.
    """
    if len(normalised) != len(records) or not records:
        raise ValueError("normalised-series and record surfaces disagree")

    volumes, positions, shapes = [], [], []
    for volume in normalised:
        image, position = series_triplets_from_normalised(
            volume,
            gap=int(gap),
            center_offset=int(center_offset),
            crop_fraction=float(crop_fraction),
        )
        if image.ndim != 4:
            raise RuntimeError(f"streamed view has shape {tuple(image.shape)}, expected 4 dims")
        shapes.append((int(image.shape[-2]), int(image.shape[-1])))
        volumes.append(image.to(device, non_blocking=True))
        positions.append(torch.from_numpy(position))
        del image

    position = torch.stack(positions).to(device, non_blocking=True)
    present = torch.ones(len(records), dtype=torch.float32).to(device, non_blocking=True)
    meta = torch.tensor(
        [[int(r["plane_id"]), int(r["fluid_id"]), int(r["fat_id"])] for r in records],
        dtype=torch.long,
    ).to(device, non_blocking=True)
    return volumes, position, present, meta, shapes


__all__ = ["build_study_view", "load_normalised_study"]
