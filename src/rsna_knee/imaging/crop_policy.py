"""The frozen crop: 90% of the native matrix, centred."""
from __future__ import annotations

import numpy as np


CROP_FRACTION = 0.90


CROP_FOCUS_VERSION = "joint_focus_center_crop_only_v1"


DEFAULT_CROP_FOCUS_POLICY = {
    "version": CROP_FOCUS_VERSION,
    "crop_fraction": 0.90,
}


def validate_crop_focus_policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise ValueError("crop focus policy must be a dictionary")
    version = str(policy.get("version", CROP_FOCUS_VERSION))
    if version != CROP_FOCUS_VERSION:
        raise ValueError(
            f"unsupported crop focus version {version!r}; expected {CROP_FOCUS_VERSION!r}"
        )
    crop = float(policy.get("crop_fraction", 0.90))
    if not 0.70 <= crop <= 1.0:
        raise ValueError("crop focus crop_fraction must be in [0.70, 1.0]")
    return {"version": version, "crop_fraction": crop}


def crop_focus_policy(config: dict) -> dict:
    if bool(config.get("b20_crop_focus_enabled", False)) is not True:
        raise ValueError("B20 requires b20_crop_focus_enabled=true")
    policy = {
        "version": str(config.get("b20_crop_focus_version", CROP_FOCUS_VERSION)),
        "crop_fraction": float(
            config.get(
                "b20_crop_focus_crop_fraction",
                DEFAULT_CROP_FOCUS_POLICY["crop_fraction"],
            )
        ),
    }
    policy = validate_crop_focus_policy(policy)
    for key, expected in DEFAULT_CROP_FOCUS_POLICY.items():
        if key == "version":
            if policy[key] != expected:
                raise ValueError(f"B20-v1 freezes {key}={expected!r}")
        elif not np.isclose(float(policy[key]), float(expected), atol=1e-12, rtol=0):
            raise ValueError(f"B20-v1 freezes {key}={expected}; got {policy[key]}")
    return policy

