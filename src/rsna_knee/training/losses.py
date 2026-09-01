"""The per-study loss, and moving one study onto the device."""
from __future__ import annotations

import torch

from ..runtime import autocast
from .supervision import confidence_weighted_bce


def move_study_to_device(item: dict, device) -> tuple:
    volumes = [volume.to(device, non_blocking=True) for volume in item["volumes"]]
    return (
        volumes,
        item["slice_position"].to(device, non_blocking=True),
        item["present"].to(device, non_blocking=True),
        item["series_meta"].to(device, non_blocking=True),
        item["target"].to(device, non_blocking=True).unsqueeze(0),
        item["weight"].to(device, non_blocking=True).unsqueeze(0),
    )


def study_losses(model, runtime, tensors, multiplier_t, aux_weight: float):
    volumes, position, present, meta, target, weight = tensors
    with autocast(runtime):
        out = model(volumes, present, meta, position)
        combined = confidence_weighted_bce(
            out.logits, target, weight, multiplier_t
        )
        local = confidence_weighted_bce(
            out.local_logits, target, weight, multiplier_t
        )
        total = combined + float(aux_weight) * local
    return out, total, combined, local


def study_supervision_mass(weight: torch.Tensor, target_multiplier: torch.Tensor) -> float:
    w = weight.reshape(-1, weight.shape[-1]).to(dtype=torch.float32, device="cpu")
    m = target_multiplier.to(dtype=torch.float32, device="cpu")
    return float((w * m[None, :]).sum().item())


def batch_scales(items: list[dict], multiplier_cpu: torch.Tensor) -> list[float]:
    masses = [study_supervision_mass(item["weight"], multiplier_cpu) for item in items]
    total = float(sum(masses))
    if total > 0:
        return [float(mass / total) for mass in masses]
    # Historical confidence_weighted_bce returns a graph-connected zero when an
    # entire batch has no usable cells. Equal zero-loss contributions preserve
    # that behavior and still expose every study's MRI to the forward path.
    return [1.0 / len(items)] * len(items)

