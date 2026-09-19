"""Loaders for the competition metadata, with the known traps handled.

Two traps are handled here so that no other code has to remember them:

1. `hand_labels.csv` is delimited with ", " rather than ",", so every column
   name and every identifier carries a leading space. Joining it against the
   other files without stripping that whitespace silently matches nothing.
2. One row of `hand_labels.csv` is corrupt: its identifier is truncated to
   `1.2.826.0.1.3680043.8.498`. It is dropped.

Import from a script or notebook:

    from scripts.load_data import LABELS, load_reports, load_gold, load_series
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Exact spelling and order as they appear in sample_submission.csv. The
# apostrophe in "Baker's" and the spaces in the multi-word names are load
# bearing: the submission must reproduce them character for character.
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# The identifier of the one malformed row in hand_labels.csv.
CORRUPT_STUDY_ID = "1.2.826.0.1.3680043.8.498"


def _read_hand_labels(data_dir: Path) -> pd.DataFrame:
    """Read hand_labels.csv, undoing its ", " delimiting."""
    frame = pd.read_csv(data_dir / "hand_labels.csv", skipinitialspace=True)
    frame.columns = [column.strip() for column in frame.columns]
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].str.strip()
    return frame[frame["StudyInstanceUID"] != CORRUPT_STUDY_ID].copy()


def load_reports(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per study, with the report in its original language and in English.

    Columns: StudyInstanceUID, report_original, report_english.

    `report_english` is missing for the single study whose translation row is
    corrupt, and is identical to `report_original` for reports that were
    already written in English.
    """
    train = pd.read_csv(data_dir / "train.csv")[["StudyInstanceUID", "Report"]]
    train = train.rename(columns={"Report": "report_original"})

    english = _read_hand_labels(data_dir)[["StudyInstanceUID", "Report"]]
    english = english.rename(columns={"Report": "report_english"})
    english["report_english"] = english["report_english"].str.strip()

    return train.merge(english, on="StudyInstanceUID", how="left")


def load_gold(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The hand-labelled studies only: the gold standard for the label extractor.

    Returns StudyInstanceUID plus the twelve label columns. Labels may still be
    missing individually — six of these studies are only partially labelled —
    so callers must mask per label rather than dropping whole rows.
    """
    frame = _read_hand_labels(data_dir)
    labelled = frame[frame[LABELS].notna().any(axis=1)]
    return labelled[["StudyInstanceUID", *LABELS]].reset_index(drop=True)


def load_series(split: str = "train", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Series-level metadata for "train" or "test".

    `Fat_Suppression` is dropped: it is identical to `Fluid_Sensitive` in every
    one of the 24,371 training series, so keeping both invites the mistake of
    treating them as two independent signals.
    """
    frame = pd.read_csv(data_dir / f"{split}_series.csv")
    if {"Fluid_Sensitive", "Fat_Suppression"} <= set(frame.columns):
        if frame["Fluid_Sensitive"].equals(frame["Fat_Suppression"]):
            frame = frame.drop(columns=["Fat_Suppression"])
    return frame


def empty_submission(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """A submission frame for the test studies, every probability at 0.5."""
    frame = pd.read_csv(data_dir / "test.csv")[["StudyInstanceUID"]]
    for label in LABELS:
        frame[label] = 0.5
    return frame


if __name__ == "__main__":
    reports = load_reports()
    gold = load_gold()
    series = load_series()
    print(f"reports : {len(reports):,} studies, "
          f"{reports.report_english.isna().sum()} without an English translation")
    print(f"gold    : {len(gold):,} studies, "
          f"{int(gold[LABELS].notna().sum().sum()):,} label cells present")
    print(f"series  : {len(series):,} rows, columns {list(series.columns)}")
    print(f"submission columns match sample: "
          f"{list(empty_submission().columns) == list(pd.read_csv(DATA_DIR / 'sample_submission.csv').columns)}")
