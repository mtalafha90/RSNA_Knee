"""Resampling to a common physical scale, so a millimetre means the same thing on every scanner."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F


MISSING_SPACING_ACTION = "legacy_resize"


PHYSICAL_ROUTING_MODE = "dual"


def physical_policy_digest(policy: dict) -> str:
    payload = dict(policy)
    payload.pop("policy_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


PHYSICAL_SCALE_POLICY = "inplane_median_spacing_fov_v1"


def validate_physical_scale_policy(policy: dict) -> None:
    if str(policy.get("policy_name")) != PHYSICAL_SCALE_POLICY:
        raise ValueError(f"B10 policy_name must be {PHYSICAL_SCALE_POLICY!r}")
    if str(policy.get("routing_mode")) != PHYSICAL_ROUTING_MODE:
        raise ValueError("B10 physical policy must be derived from historical dual routing")
    if str(policy.get("normalization_scope")) != "in_plane_only":
        raise ValueError("B10-v1 normalizes in-plane geometry only")
    if str(policy.get("missing_spacing_action")) != MISSING_SPACING_ACTION:
        raise ValueError(
            f"B10 missing-spacing action must remain {MISSING_SPACING_ACTION!r}"
        )
    if bool(policy.get("uses_gold_labels", True)):
        raise ValueError("B10 physical-scale policy must certify zero gold-label use")
    planes = policy.get("planes")
    if not isinstance(planes, dict):
        raise ValueError("B10 physical-scale policy is missing planes")
    for plane in ("Sagittal", "Coronal", "Axial"):
        part = planes.get(plane)
        if not isinstance(part, dict):
            raise ValueError(f"B10 physical policy missing {plane}")
        for field in ("target_spacing_mm", "target_fov_mm"):
            values = np.asarray(part.get(field, []), dtype=float).reshape(-1)
            if len(values) != 2 or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"invalid B10 {plane} {field}: {part.get(field)!r}")
    expected = physical_policy_digest(policy)
    recorded = str(policy.get("policy_sha256", ""))
    if recorded and recorded != expected:
        raise ValueError("B10 physical policy SHA256 does not match its contents")


def resample_volume_inplane(
    volume: np.ndarray,
    *,
    source_spacing_mm: tuple[float, float] | list[float] | None,
    plane: str,
    policy: dict,
) -> tuple[np.ndarray, bool]:
    """Resample to canonical mm/pixel, then center crop/pad to canonical physical FOV."""
    validate_physical_scale_policy(policy)
    if source_spacing_mm is None:
        return np.asarray(volume, dtype=np.float32), False
    source = np.asarray(source_spacing_mm, dtype=float).reshape(-1)
    if len(source) != 2 or not np.isfinite(source).all() or np.any(source <= 0):
        return np.asarray(volume, dtype=np.float32), False
    if plane not in policy["planes"]:
        raise ValueError(f"B10 policy does not contain plane {plane!r}")

    target_spacing = np.asarray(policy["planes"][plane]["target_spacing_mm"], dtype=float)
    target_fov = np.asarray(policy["planes"][plane]["target_fov_mm"], dtype=float)

    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3 or len(v) == 0:
        raise ValueError(f"expected [S,H,W] volume, got {v.shape}")
    source_h, source_w = int(v.shape[1]), int(v.shape[2])
    new_h = max(1, int(round(source_h * source[0] / target_spacing[0])))
    new_w = max(1, int(round(source_w * source[1] / target_spacing[1])))

    tensor = torch.from_numpy(v).unsqueeze(1)
    if (new_h, new_w) != (source_h, source_w):
        tensor = F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    target_h = max(1, int(round(target_fov[0] / target_spacing[0])))
    target_w = max(1, int(round(target_fov[1] / target_spacing[1])))
    current_h, current_w = int(tensor.shape[-2]), int(tensor.shape[-1])

    crop_h = min(current_h, target_h)
    crop_w = min(current_w, target_w)
    src_r = max(0, (current_h - crop_h) // 2)
    src_c = max(0, (current_w - crop_w) // 2)
    tensor = tensor[..., src_r : src_r + crop_h, src_c : src_c + crop_w]

    pad_h = target_h - crop_h
    pad_w = target_w - crop_w
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        tensor = F.pad(tensor, (left, right, top, bottom), value=0.0)

    return tensor.squeeze(1).cpu().numpy().astype(np.float32, copy=False), True

