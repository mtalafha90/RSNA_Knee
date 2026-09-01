"""Report-derived labels: the merged export, its confidences, and what silence means."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from ..constants import TARGETS
from .tables import gold_mask
from ..training.supervision import MIN_CONFIDENCE, NEGATIVE_TARGET, NEGATIVE_WEIGHT, POSITIVE_TARGET, POSITIVE_WEIGHT, SUPERVISION_VARIANT


REPORT_ONLY_STUDIES = 4349


PHASE9_VERSION = "phase9_matched_b34_b6_vs_phase8_supervision_v1"


def load_fill_merged_export(root: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """Read a fill-only merged export and insist it overrode nothing.

    The merge is usable precisely because it preserves every parser call, which
    is what keeps the frozen specificity intact. An export claiming otherwise is
    a different experiment and must not be trained on under this name.
    """
    root = Path(root)
    targets_path = root / "training_targets.csv"
    policy_path = root / "policy.json"
    audit_path = root / "audit.json"
    for path in (targets_path, policy_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"fill-merged export is missing {path}")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if int(audit.get("base_cells_overridden", -1)) != 0:
        raise ValueError(
            "this export overrode base parser cells; fill-only is what makes the "
            "frozen specificity carry through, so it is required here"
        )
    if int(audit.get("gold_rows_in_training_targets", -1)) != 0:
        raise ValueError("fill-merged export does not certify zero gold rows")
    return pd.read_csv(targets_path), policy, audit


def prepare_all_report_only_supervision(
    train_df: pd.DataFrame,
    supervision_frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Create B7-style targets/weights without dropping zero-cell studies."""
    train = train_df.copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    frame = supervision_frame.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("supervision table contains duplicate StudyInstanceUID values")

    gold = gold_mask(train)
    gold_uids = set(train.loc[gold, "StudyInstanceUID"])
    overlap = gold_uids.intersection(set(frame["StudyInstanceUID"]))
    if overlap:
        raise ValueError(f"Phase 9 supervision contains {len(overlap)} gold UID(s)")

    non_gold = train.loc[~gold, ["StudyInstanceUID"]].copy()
    if len(non_gold) != REPORT_ONLY_STUDIES:
        raise ValueError("Phase 9 requires exactly 4,349 report-only studies")
    expected = set(non_gold["StudyInstanceUID"])
    actual = set(frame["StudyInstanceUID"])
    if expected != actual:
        raise ValueError(
            f"Phase 9 report-only UID mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}"
        )

    ordered = non_gold.merge(frame, on="StudyInstanceUID", how="left", validate="one_to_one")
    n = len(ordered)
    y = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros((n, len(TARGETS)), dtype=np.float32)
    per_target: dict[str, dict] = {}

    for j, target in enumerate(TARGETS):
        state_col = f"{target}__state"
        conf_col = f"{target}__confidence"
        if state_col not in ordered or conf_col not in ordered:
            raise ValueError(f"supervision table missing state/confidence for {target}")
        state = ordered[state_col].fillna("").astype(str).to_numpy()
        conf = pd.to_numeric(ordered[conf_col], errors="coerce").fillna(0.0).to_numpy(float)
        positive = (state == "positive") & (conf >= MIN_CONFIDENCE)
        negative = (state == "negated") & (conf >= MIN_CONFIDENCE)
        y[positive, j] = POSITIVE_TARGET
        y[negative, j] = NEGATIVE_TARGET
        w[positive, j] = POSITIVE_WEIGHT
        w[negative, j] = NEGATIVE_WEIGHT
        per_target[target] = {
            "positive_cells": int(positive.sum()),
            "negative_cells": int(negative.sum()),
            "usable_cells": int((positive | negative).sum()),
            "base_weight_sum": float(w[:, j].sum()),
        }

    active = w.sum(axis=1) > 0
    summary = {
        "report_only_rows": n,
        "active_studies": int(active.sum()),
        "inactive_studies_zero_usable_cells": int((~active).sum()),
        "usable_cells": int((w > 0).sum()),
        "positive_cells": int(((w > 0) & (y > 0.5)).sum()),
        "negative_cells": int(((w > 0) & (y < 0.5)).sum()),
        "targets": per_target,
        "zero_weight_studies_retained_in_mri_exposure": True,
    }
    return ordered["StudyInstanceUID"].astype(str).tolist(), y, w, summary


EXPORTED_NEGATIVE_TARGET = 0.05


# What the frozen export contains, and what a rescale replaces.
EXPORTED_POSITIVE_TARGET = 0.85


NEGATIVE_TARGET_KEY = "label_confidence_negative_target"


# A cell counts as positive or negative by which side of this it falls, in the
# training loop and in the audit counts alike. Targets must stay on their own
# side of it or those counts quietly start describing something else.
DECISION_BOUNDARY = 0.5


# Deliberately not the b7_* keys: those describe the export and are frozen.
POSITIVE_TARGET_KEY = "label_confidence_positive_target"


def rescale_label_confidence(
    targets: np.ndarray, weights: np.ndarray, config: dict
) -> tuple[np.ndarray, dict]:
    """Move the supervised targets onto the configured confidences.

    Only cells that carry weight are touched, so unsupervised cells keep
    whatever placeholder the export gave them. Returns the new targets and a
    record of what happened, for the training payload.
    """
    positive_target = float(config.get(POSITIVE_TARGET_KEY, EXPORTED_POSITIVE_TARGET))
    negative_target = float(config.get(NEGATIVE_TARGET_KEY, EXPORTED_NEGATIVE_TARGET))

    for name, value in (("positive", positive_target), ("negative", negative_target)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} target must lie between 0 and 1, not {value}")
    if positive_target <= DECISION_BOUNDARY:
        raise ValueError(
            f"positive target must stay above {DECISION_BOUNDARY}, not {positive_target}; "
            "below it, positive cells would be counted as negatives"
        )
    if negative_target >= DECISION_BOUNDARY:
        raise ValueError(
            f"negative target must stay below {DECISION_BOUNDARY}, not {negative_target}; "
            "above it, negative cells would be counted as positives"
        )

    changed = not (
        np.isclose(positive_target, EXPORTED_POSITIVE_TARGET)
        and np.isclose(negative_target, EXPORTED_NEGATIVE_TARGET)
    )
    record = {
        "changed": bool(changed),
        # The identity the contract asks for. A rescaled run is not B7-v1
        # supervision and must not be recorded as though it were.
        "supervision_policy": (
            f"{SUPERVISION_VARIANT}_retargeted_pos{positive_target:g}_neg{negative_target:g}"
            if changed
            else SUPERVISION_VARIANT
        ),
        "positive_target": positive_target,
        "negative_target": negative_target,
        "exported_positive_target": EXPORTED_POSITIVE_TARGET,
        "exported_negative_target": EXPORTED_NEGATIVE_TARGET,
        "measured_positive_agreement": 0.690,
        "measured_negative_agreement": 0.964,
        "measurement_source": "tools.label_audit over the 58 expert-labelled studies",
    }
    if not changed:
        return targets, record

    rescaled = np.array(targets, dtype=targets.dtype, copy=True)
    supervised = weights > 0
    positives = supervised & (targets > DECISION_BOUNDARY)
    negatives = supervised & (targets < DECISION_BOUNDARY)
    rescaled[positives] = positive_target
    rescaled[negatives] = negative_target

    record["positive_cells_rescaled"] = int(positives.sum())
    record["negative_cells_rescaled"] = int(negatives.sum())
    return rescaled, record

