"""Turning report labels into targets and weights, and balancing the twelve findings."""
from __future__ import annotations

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import argparse
import json
import numpy as np
import pandas as pd
import random
import time
import torch
import yaml

from ..constants import TARGETS
from ..data.dataset import DatasetConfig


def make_dataset_config(config: dict, root: str | Path, *, train: bool, tta_offsets: tuple[int, ...] = ()) -> DatasetConfig:
    return DatasetConfig(
        data_root=str(root), split="train",
        n_slices=int(config.get("b7_n_slices", 16)),
        image_size=int(config.get("b7_image_size", 224)),
        noise_std=float(config.get("b7_noise_std", 0.02)) if train else 0.0,
        slice_dropout=float(config.get("b7_slice_dropout", 0.08)) if train else 0.0,
        triplet_gap=int(config.get("b7_triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom", False)) if train else bool(config.get("strict_dicom_inference", True)),
        tta_center_offsets=tuple(int(x) for x in tta_offsets),
        train_gap_choices=tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])),
        center_jitter=int(config.get("b7_center_jitter", 2)) if train else 0,
        rotation_deg=float(config.get("b7_rotation_deg", 5.0)) if train else 0.0,
        translate_frac=float(config.get("b7_translate_frac", 0.03)) if train else 0.0,
        scale_jitter=float(config.get("b7_scale_jitter", 0.05)) if train else 0.0,
        gamma_jitter=float(config.get("b7_gamma_jitter", 0.12)) if train else 0.0,
        bias_field_strength=float(config.get("b7_bias_field_strength", 0.08)) if train else 0.0,
        series_cache_mb=int(config.get("series_cache_mb_per_worker", 256)),
    )


def target_balance_multipliers(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(TARGETS):
        raise ValueError("weights must have shape [N,12]")
    mass = weights.sum(axis=0)
    valid = mass > 0
    if not valid.all():
        missing = [TARGETS[j] for j in np.flatnonzero(~valid)]
        raise ValueError(f"B7 has no usable supervision for target(s): {missing}")
    mean_mass = float(mass.mean())
    return (mean_mass / mass).astype(np.float32)


def confidence_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    target_multiplier: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != weight.shape:
        raise ValueError("logits/target/weight shapes must match")
    if logits.ndim != 2 or logits.shape[1] != len(TARGETS):
        raise ValueError("B7 logits must have shape [B,12]")
    multiplier = torch.as_tensor(target_multiplier, dtype=logits.dtype, device=logits.device)
    if multiplier.shape != (len(TARGETS),):
        raise ValueError("target_multiplier must have shape [12]")
    effective = weight * multiplier[None, :]
    denominator = effective.sum()
    if float(denominator.detach().item()) <= 0:
        return logits.sum() * 0.0
    cell = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (cell * effective).sum() / denominator.clamp_min(1e-8)


SUPERVISION_VARIANT = "b7_b5_init_b6_asymmetric_weak_v1"


POSITIVE_WEIGHT = 0.50


POSITIVE_TARGET = 0.85


NEGATIVE_TARGET = 0.05


NEGATIVE_WEIGHT = 1.00


MIN_CONFIDENCE = 0.75

