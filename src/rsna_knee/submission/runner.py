"""Inferring one shard of the test set, on one device, without giving up on it.

Two behaviours here exist because the archive's equivalents did not have them,
and three hidden runs died as a result.

**An unreadable study does not end the run.** It gets a constant prediction and
the run carries on. One bad study out of 1,300 should not forfeit the other
1,299, and on a hidden set nobody can see which one it was.

**The runtime projection cannot raise.** It is the mean of the last five studies
times the remaining count times a safety factor, so one slow early study in a
650-study shard forecasts past the budget and kills a run that would in fact
have finished. It is printed, and that is all.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..constants import TARGETS
from ..training.memory import _trim_host_memory
from .views import build_study_view, load_normalised_study

ON_UNREADABLE_RAISE = "raise"
ON_UNREADABLE_FALLBACK = "fallback"
ON_UNREADABLE_MODES = (ON_UNREADABLE_RAISE, ON_UNREADABLE_FALLBACK)

# A study nothing could be read from still needs a row, and every target gets
# the same number. ROC AUC sees only ordering, so one constant leaves those
# studies tied among themselves and contributes neither way -- the honest
# outcome for a study the model never saw.
FALLBACK_PROBABILITY = 0.5

TIMING_SAFETY_FACTOR = 1.35


def projected_remaining_seconds(
    durations: list[float], *, remaining: int, safety_factor: float = TIMING_SAFETY_FACTOR
) -> float:
    """A deliberately pessimistic estimate, from the last five complete studies."""
    if remaining < 0:
        raise ValueError("remaining must be non-negative")
    if not durations:
        return 180.0
    window = np.asarray(durations[-5:], dtype=np.float64)
    return float(window.mean() * int(remaining) * float(safety_factor))


@torch.inference_mode()
def infer_one_study(
    uid: str,
    records: list[dict],
    *,
    model,
    data_root,
    split: str,
    tta_offsets: tuple[int, ...],
    gap: int,
    crop_fraction: float,
    device: torch.device,
    physical_scale_policy=None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[dict]]:
    """Average the sigmoid probabilities across TTA views, one view at a time."""
    normalised, kept, dropped = load_normalised_study(
        data_root, split, uid, records, physical_scale_policy=physical_scale_policy
    )
    if not normalised:
        raise RuntimeError(f"study {uid} has no readable MRI series")

    view_probabilities: list[torch.Tensor] = []
    shapes: list[tuple[int, int]] = []
    for position, offset in enumerate(tta_offsets):
        volumes, slice_position, present, meta, view_shapes = build_study_view(
            normalised, kept, center_offset=offset, gap=gap,
            crop_fraction=crop_fraction, device=device,
        )
        if position == 0:
            shapes = view_shapes
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(volumes, present, meta, slice_position)
        view_probabilities.append(torch.sigmoid(out.logits.float()).cpu())
        del volumes, slice_position, present, meta, out

    probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
    if probability.shape != (1, len(TARGETS)) or not torch.isfinite(probability).all():
        raise RuntimeError(f"study {uid} produced invalid probabilities")
    return probability.numpy()[0], shapes, dropped


def infer_shard(
    *,
    rank: int,
    indices: list[int],
    uids: list[str],
    index: dict,
    model,
    data_root,
    split: str = "test",
    tta_offsets: tuple[int, ...] = (-1, 0, 1),
    gap: int = 1,
    crop_fraction: float = 0.90,
    device: torch.device | None = None,
    on_unreadable: str = ON_UNREADABLE_RAISE,
    fallback_probability: float = FALLBACK_PROBABILITY,
    max_hours: float = 8.25,
    reserve_minutes: float = 30.0,
    global_started: float | None = None,
    physical_scale_policy=None,
) -> tuple[list[tuple], list[dict]]:
    """Infer the studies at `indices`, in order, on one device."""
    if on_unreadable not in ON_UNREADABLE_MODES:
        raise ValueError(f"on_unreadable must be one of {ON_UNREADABLE_MODES}")
    device = torch.device(f"cuda:{int(rank)}") if device is None else device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    started_at = time.monotonic() if global_started is None else global_started

    rows: list[tuple] = []
    failures: list[dict] = []
    durations: list[float] = []

    for position, study_index in enumerate(indices):
        uid = str(uids[study_index])
        started = time.monotonic()

        def one_study(uid=uid, study_index=study_index):
            records = index.get(uid, [])
            if not records:
                raise RuntimeError(f"study {uid} has no series with a recognised plane")
            return infer_one_study(
                uid, records, model=model, data_root=data_root, split=split,
                tta_offsets=tta_offsets, gap=gap, crop_fraction=crop_fraction,
                device=device, physical_scale_policy=physical_scale_policy,
            )

        if on_unreadable == ON_UNREADABLE_RAISE:
            probability, shapes, dropped = one_study()
        else:
            probability = shapes = dropped = None
            try:
                probability, shapes, dropped = one_study()
            except torch.OutOfMemoryError as first:
                print(
                    f"[submit gpu{rank}] {uid} out of memory, retrying once with an "
                    f"empty cache: {first}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                try:
                    probability, shapes, dropped = one_study()
                except Exception as second:  # noqa: BLE001
                    error = second
            except Exception as only:  # noqa: BLE001
                error = only

            if probability is None:
                probability = np.full(len(TARGETS), float(fallback_probability), np.float32)
                shapes, dropped = [], []
                failures.append(
                    {"index": int(study_index), "study_uid": uid,
                     "error": f"{type(error).__name__}: {error}"}
                )
                print(
                    f"[submit gpu{rank}] {uid} unreadable, predicting "
                    f"{fallback_probability} for all targets: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

        if dropped:
            print(
                f"[submit gpu{rank}] {uid}: dropped {len(dropped)}/"
                f"{len(index.get(uid, []))} unreadable series",
                flush=True,
            )
        rows.append((int(study_index), uid, probability, shapes))
        _trim_host_memory()
        durations.append(time.monotonic() - started)

        completed = position + 1
        if completed % 10 == 0 or completed == len(indices):
            projected = projected_remaining_seconds(
                durations, remaining=len(indices) - completed
            )
            available = float(max_hours) * 3600.0 - float(reserve_minutes) * 60.0 - (
                time.monotonic() - started_at
            )
            note = "" if projected <= available else "  OVER BUDGET (telemetry only)"
            print(
                f"[submit gpu{rank}] {completed}/{len(indices)} "
                f"elapsed={sum(durations)/60.0:.1f} min "
                f"projected_remaining={projected/60.0:.1f} min{note}",
                flush=True,
            )

    return rows, failures


__all__ = [
    "FALLBACK_PROBABILITY",
    "ON_UNREADABLE_FALLBACK",
    "ON_UNREADABLE_MODES",
    "ON_UNREADABLE_RAISE",
    "infer_one_study",
    "infer_shard",
    "projected_remaining_seconds",
]
