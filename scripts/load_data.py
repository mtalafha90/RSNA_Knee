"""Loaders for the competition metadata.

`data/hand_labels.csv` is deliberately **not** used anywhere in this project.
Its provenance is unknown, its English text is a translation rather than the
source report, and a mistranslation would corrupt the labels silently. Nothing
here reads it. If a translated corpus is wanted later, it should be built from
`train.csv` by a process we control and can re-run.

Import from a script or notebook:

    from scripts.load_data import LABELS, load_train, load_gold, load_series
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


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """All 4,407 training studies: identifier, report, and the twelve labels.

    The labels are missing for all but 58 studies. That is the shape of the
    problem, not a loading error.
    """
    return pd.read_csv(data_dir / "train.csv")


def load_gold(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The ground truth: every study for which we have trustworthy labels.

    Prefers `gold_labels.csv` — the enlarged set produced by hand-labelling a
    worklist — and falls back to the 58 studies labelled in `train.csv` when it
    does not exist yet.

    Those original 58 all carry twelve labels and every one has at least one
    positive finding. With no all-negative study among them, that set can
    measure how well an extractor finds abnormalities but says nothing about
    how often it invents them on a normal report. Enlarging it is what
    `make_worklist.py` and `merge_gold.py` are for.
    """
    enlarged = data_dir / "gold_labels.csv"
    if enlarged.exists():
        return pd.read_csv(enlarged)[["StudyInstanceUID", *LABELS]]

    train = load_train(data_dir)
    gold = train[train[LABELS].notna().all(axis=1)]
    return gold[["StudyInstanceUID", *LABELS]].reset_index(drop=True)


def load_unlabelled(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The 4,349 studies with a report but no labels — the extraction target."""
    train = load_train(data_dir)
    return train[train[LABELS].isna().all(axis=1)].reset_index(drop=True)


def load_series(split: str = "train", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Series-level metadata for "train" or "test".

    `Fat_Suppression` is dropped when it duplicates `Fluid_Sensitive`, which it
    does in every one of the 24,371 training series. Keeping both invites the
    mistake of treating them as two independent signals.
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
    train, gold, unlabelled = load_train(), load_gold(), load_unlabelled()
    sample_columns = list(pd.read_csv(DATA_DIR / "sample_submission.csv").columns)
    print(f"train      : {len(train):,} studies")
    print(f"gold       : {len(gold):,} studies, all twelve labels each")
    print(f"unlabelled : {len(unlabelled):,} studies to extract labels for")
    print(f"series     : {len(load_series()):,} rows, "
          f"columns {list(load_series().columns)}")
    print(f"submission columns match the sample: "
          f"{list(empty_submission().columns) == sample_columns}")
