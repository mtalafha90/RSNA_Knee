"""Assembling the report-only training surface and fingerprinting what went into it."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from ..checkpoints import EXPECTED_BASE_CELLS, sha256_file
from ..data.labels import REPORT_ONLY_STUDIES, load_fill_merged_export, prepare_all_report_only_supervision, rescale_label_confidence
from ..data.tables import gold_mask, load_train_csv


def build_report_only_surface(
    *,
    data_root: Path,
    labels_root: str | Path,
    config: dict,
    domain_rows: pd.DataFrame,
    base_payload: dict,
    expected_cells: int = EXPECTED_BASE_CELLS,
) -> tuple:
    """Return the split-aligned B48 weak-label surface without any gold rows.

    `expected_cells` pins how many cells the report labels supervise. Its default
    is the base checkpoint's own count, so an export edited between runs still
    trips the guard, which is what it is for.

    A **deliberate** change of teacher changes that count legitimately -- filling
    only negated cells, for instance, supervises 25,524 rather than 34,010 -- and
    such a run must state the number it expects rather than have the guard
    quietly relaxed for everyone. The count reaches the checkpoint either way, so
    which surface trained a model is recoverable from the model.
    """
    train = load_train_csv(data_root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("B48 requires the complete 4,407-study training release")
    gold_uids = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))

    frame, fill_policy, fill_audit = load_fill_merged_export(labels_root)
    all_uids, all_targets, all_weights, supervision = prepare_all_report_only_supervision(train, frame)
    all_uids = [str(uid) for uid in all_uids]
    if len(all_uids) != REPORT_ONLY_STUDIES:
        raise ValueError("B48 requires all 4,349 report-only studies before the split")
    observed_cells = int((all_weights > 0).sum())
    if observed_cells != int(expected_cells):
        raise ValueError(
            f"the weak supervision surface has {observed_cells:,} cells, not the "
            f"{int(expected_cells):,} this run expects. If the label export was "
            "edited or replaced by accident, that is what this guard is for. If "
            "the teacher was changed on purpose, say so: pass "
            f"--expected-cells {observed_cells}"
        )
    if set(all_uids).intersection(gold_uids):
        raise RuntimeError("B48 report-only supervision includes an official gold study")
    if int(fill_audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B48 requires zero B6 overrides")
    if list(fill_audit.get("excluded_targets", [])):
        raise ValueError("B48 requires all 12 targets")

    split_uids = set(domain_rows["StudyInstanceUID"])
    if split_uids != set(all_uids):
        missing = sorted(set(all_uids).difference(split_uids))
        extra = sorted(split_uids.difference(set(all_uids)))
        raise RuntimeError(
            "B48 domain split/report-only population mismatch "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    all_targets, confidence = rescale_label_confidence(all_targets, all_weights, config)
    base_confidence = base_payload.get("label_confidence", {})
    for key in ("positive_target", "negative_target"):
        if key in base_confidence and not np.isclose(
            float(confidence[key]), float(base_confidence[key]), atol=1e-12, rtol=0
        ):
            raise ValueError(f"B48 label confidence mismatch for {key}")
    supervision = {**supervision, "expected_cells": int(expected_cells)}
    lookup = {uid: index for index, uid in enumerate(all_uids)}
    return (
        train,
        all_uids,
        all_targets.astype(np.float32),
        all_weights.astype(np.float32),
        lookup,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


FILL_ARTIFACT_FILES = ("training_targets.csv", "policy.json", "audit.json")


def indices_for_split(all_uids: list[str], rows: pd.DataFrame, split: str) -> np.ndarray:
    selected = set(rows.loc[rows["split"].eq(str(split)), "StudyInstanceUID"].astype(str))
    result = np.asarray([index for index, uid in enumerate(all_uids) if uid in selected], dtype=np.int64)
    if len(result) != len(selected) or not len(result):
        raise RuntimeError(f"B48 split {split!r} has missing or zero report-only UIDs")
    return result


def config_sha256(config: dict) -> str:
    return sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str))


def uid_sha256(uids: list[str]) -> str:
    return sha256_text("\n".join(str(uid) for uid in uids) + "\n")


def fill_artifacts(labels_root: str | Path) -> dict[str, str]:
    """Fingerprint each fill-only label artifact consumed by B48.

    ``load_fill_merged_export`` enforces the no-overwrite rules but does not
    itself pin artifact digests.  Recording these inputs prevents a paired
    evaluation from silently using a later-edited report-only label export.
    """
    root = Path(labels_root).resolve()
    result: dict[str, str] = {}
    for name in FILL_ARTIFACT_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"B48 fill-only artifact is missing: {path}")
        result[name] = sha256_file(path)
    return result

