"""Reading and fingerprinting the base checkpoint a run starts from."""
from __future__ import annotations

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import argparse
import hashlib
import json
import math
import numpy as np
import time
import torch

from .data.labels import PHASE9_VERSION, REPORT_ONLY_STUDIES
from .model.encoder import attach_dinov3_encoder, encoder_state_sha256, freeze_encoder
from .model.study_hierarchy import build_study_hierarchy


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


EXPECTED_BASE_EPOCHS = 2


EXPECTED_BASE_CELLS = 34010


EXPECTED_BASE_ARM = "llm_fill"


EXPECTED_BASE_SERIES = 24035


PHASE9_EXPERIMENT = "phase9_matched_b34_b6_vs_phase8_supervision"


PHASE9_ARMS = ("control", "candidate", "llm_fill")


PHASE9_EXPECTED_REPORT_ONLY_SERIES = 24035


PHASE9_FIXED_EPOCHS = 2


def require_base_checkpoint(payload: dict) -> None:
    if str(payload.get("arm")) != EXPECTED_BASE_ARM:
        raise ValueError("B35 Phase A requires the full llm_fill B34 base checkpoint")
    if int(payload.get("completed_epochs", -1)) != EXPECTED_BASE_EPOCHS:
        raise ValueError("B35 requires the completed fixed-E2 B34 base")
    if bool(payload.get("fixed_endpoint")) is not True:
        raise ValueError("B35 base checkpoint is not marked as a fixed endpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B35 base checkpoint unexpectedly used expert labels")
    if int(payload.get("report_only_studies_exposed", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("B35 base checkpoint used a different report-only population")
    if int(payload.get("training_series", -1)) != EXPECTED_BASE_SERIES:
        raise ValueError("B35 base checkpoint used a different MRI series surface")
    if int(payload.get("training_supervision_cells", -1)) != EXPECTED_BASE_CELLS:
        raise ValueError(
            "B35 Phase A is pinned to the 34,010-cell full LLM-fill checkpoint"
        )
    if int(payload.get("encoder_trainable_stages", -1)) != 1:
        raise ValueError("B35 Phase A requires the measured one-stage fine-tuned base")


def load_base_checkpoint(
    path: str | Path,
    *,
    expected_arm: str | None = None,
    device: torch.device | str = "cpu",
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != PHASE9_EXPERIMENT or payload.get("phase9_version") != PHASE9_VERSION:
        raise ValueError("not a Phase-9 matched-supervision checkpoint")
    arm = str(payload.get("arm", ""))
    if arm not in PHASE9_ARMS:
        raise ValueError("Phase-9 checkpoint has invalid arm")
    if expected_arm is not None and arm != str(expected_arm):
        raise ValueError(f"Phase-9 checkpoint arm mismatch: expected {expected_arm}, got {arm}")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != PHASE9_FIXED_EPOCHS:
        raise ValueError("Phase-9 checkpoint must be complete fixed-E2")
    if payload.get("validation_used_for_checkpoint_selection") is not False:
        raise ValueError("Phase-9 validation unexpectedly used for checkpoint selection")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(payload.get("gold_labels_used", True)):
        raise ValueError("Phase-9 checkpoint unexpectedly used gold labels")
    if int(payload.get("report_only_studies_exposed", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("Phase-9 MRI exposure population changed")
    if int(payload.get("training_series", -1)) != PHASE9_EXPECTED_REPORT_ONLY_SERIES:
        raise ValueError("Phase-9 MRI series exposure changed")
    if payload.get("stochastic_path_matched_after_model_construction") is not True:
        raise ValueError("Phase-9 checkpoint lacks matched RNG reset")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or not final_sha:
        raise ValueError("Phase-9 checkpoint is missing an encoder fingerprint")
    if int(payload.get("encoder_trainable_stages", 0)) == 0 and initial_sha != final_sha:
        raise ValueError("Phase-9 checkpoint encoder fingerprint changed")

    spec = payload["model_spec"]
    model = build_study_hierarchy(spec, pretrained_weights=False)
    # A DINO checkpoint stores a replacement encoder rather than the original
    # ConvNeXtSliceEncoder.  Recreate that module before loading its state so
    # strict loading and the final fingerprint verify the checkpoint actually
    # used at training time.  `pretrained_weights=False` is deliberate: the
    # checkpoint supplies every encoder weight and evaluation must not download
    # or substitute a newer external model.
    encoder_source = str(
        spec.get("encoder_source", payload.get("encoder_source", "report-aligned"))
    )
    if encoder_source == "dinov3":
        encoder_description = payload.get("encoder", {})
        variant = (
            encoder_description.get("variant")
            if isinstance(encoder_description, dict)
            else None
        ) or spec.get("dinov3_variant") or "tiny"
        attach_dinov3_encoder(model, variant=str(variant), pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    # The state dict holds the encoder as it ended, which equals the initial
    # fingerprint only when the encoder was frozen throughout.
    if encoder_state_sha256(model.encoder) != final_sha:
        raise ValueError("Phase-9 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload

