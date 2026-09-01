"""The scanner-grouped split: whole scanner models held out, never half of one."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from ..checkpoints import sha256_file


SPLIT_EXCLUDED = "excluded_prior_surface"


SPLIT_TRAIN = "train"


SPLIT_SEEN_SCANNERS = "validation_seen_scanners"


SPLIT_UNSEEN_SCANNERS = "validation_unseen_scanners"


PARENT_TRAIN_SPLIT = "train"


ALLOWED_SPLITS = {
    SPLIT_TRAIN,
    SPLIT_SEEN_SCANNERS,
    SPLIT_UNSEEN_SCANNERS,
    SPLIT_EXCLUDED,
}


def verify_selection_split(rows: pd.DataFrame) -> None:
    """Enforce B50's boundary around rows spent by B48/B49."""
    required = {
        "StudyInstanceUID",
        "scanner_profile",
        "parent_b48_split",
        "b50_split",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"B50 selection split rows are missing columns: {missing}")
    if rows["StudyInstanceUID"].astype(str).duplicated().any():
        raise ValueError("B50 selection split contains duplicate study UIDs")
    labels = set(rows["b50_split"].astype(str))
    unknown = sorted(labels.difference(ALLOWED_SPLITS))
    if unknown:
        raise ValueError(f"B50 selection split contains unknown labels: {unknown}")

    parent = rows["parent_b48_split"].astype(str)
    b50 = rows["b50_split"].astype(str)
    prior = ~parent.eq(PARENT_TRAIN_SPLIT)
    if not b50.loc[prior].eq(SPLIT_EXCLUDED).all():
        raise ValueError("B50 would reuse a B48/B49 validation row instead of excluding it")
    if b50.loc[~prior].eq(SPLIT_EXCLUDED).any():
        raise ValueError("B50 left a parent-training row outside its fresh split")

    train_profiles = set(rows.loc[b50.eq(SPLIT_TRAIN), "scanner_profile"].astype(str))
    seen_profiles = set(rows.loc[b50.eq(SPLIT_SEEN_SCANNERS), "scanner_profile"].astype(str))
    unseen_profiles = set(rows.loc[b50.eq(SPLIT_UNSEEN_SCANNERS), "scanner_profile"].astype(str))
    if not train_profiles or not seen_profiles or not unseen_profiles:
        raise ValueError("B50 requires non-empty train, seen, and unseen groups")
    if unseen_profiles.intersection(train_profiles) or unseen_profiles.intersection(seen_profiles):
        raise ValueError("B50 unseen-scanner profiles straddle training or seen validation")
    if not seen_profiles.issubset(train_profiles):
        missing_profiles = sorted(seen_profiles.difference(train_profiles))
        raise ValueError(
            "B50 seen-scanner validation contains profiles absent from B50 training: "
            f"{missing_profiles[:5]}"
        )


def load_selection_gate(path: str | Path) -> tuple[dict, "pd.DataFrame", dict]:
    """Read the fresh B50 gate, not the split B48 and B49 already spent.

    B50 must not be selected on the B48/B49 scanner surface. Those rows have
    been inspected twice, and the project's own governance records that split as
    spent for new architecture selection. The fresh gate is built only from the
    parent's former `train` rows, with every former B48/B49 validation row
    excluded outright.

    The gate stores its assignment under `b50_split` with an
    `excluded_prior_surface` label the parent format has no equivalent for, so
    it is verified here and then renamed to the `split` column the shared index
    helper expects. Requiring this file rather than accepting either format is
    deliberate: a trainer that silently accepted the spent split would let the
    boundary be crossed by a path argument.

    Every one of the 4,349 report-only rows is returned, the spent ones still
    carrying their `excluded_prior_surface` label. They are excluded by never
    being asked for -- only the `train` split is selected -- rather than by being
    deleted here. Deleting them would defeat the shared surface check that every
    report-only study is accounted for by exactly one split, which is a guard
    worth keeping.
    """
    import pandas as pd

    # SPLIT_EXCLUDED and verify_selection_split are defined in this module.
    # They used to live in a separate one and were imported here lazily to
    # break a cycle that no longer exists.

    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    payload_path = directory / "b50_selection_split.json"
    rows_path = directory / "b50_selection_split_by_study.csv"
    if not payload_path.exists() or not rows_path.exists():
        raise FileNotFoundError(
            f"B50 requires its fresh selection gate at {directory}. Build it once "
            "with developments/scripts/prepare_b50_ordered_slice_gate.sh; the "
            "B48/B49 domain_split.json is spent and must not be used here."
        )

    payload = json.loads(payload_path.read_text())
    rows = pd.read_csv(rows_path)
    verify_selection_split(rows)

    meta = {
        "path": str(payload_path),
        "sha256": sha256_file(payload_path),
        "rows_sha256": sha256_file(rows_path),
        "version": payload.get("version"),
        "salt": payload.get("salt"),
    }
    rows = rows.copy()
    rows["split"] = rows["b50_split"].astype(str)
    if not (rows["split"] == SPLIT_EXCLUDED).any():
        raise ValueError(
            "B50 gate marks no rows as spent by B48/B49, which cannot be right: "
            "the parent's validation rows must all carry excluded_prior_surface"
        )
    return payload, rows, meta

