"""Merge a filled worklist into the gold set, and report what it bought you.

Validates the hand-entered labels, combines them with the 58 already in
train.csv, writes data/gold_labels.csv, and measures how much the confidence
interval narrowed — which is the whole point of the exercise.

Usage:
    python scripts/merge_gold.py data/worklist.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.load_data import LABELS, load_gold  # noqa: E402
from scripts.metrics import macro_auc  # noqa: E402


def interval_width(n_studies: int, truth: np.ndarray, seed: int = 0) -> float:
    """Expected width of the 95% macro-AUC interval at a given gold-set size.

    Estimated against a random-scoring model, so the number reflects how much
    the sample size constrains the measurement rather than how good any
    particular extractor is.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(600):
        index = rng.integers(0, len(truth), n_studies)
        sampled = truth[index]
        value = macro_auc(sampled, rng.random(sampled.shape))
        if not np.isnan(value):
            draws.append(value)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(high - low)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worklist", type=Path)
    parser.add_argument("--out", default=Path("data/gold_labels.csv"), type=Path)
    args = parser.parse_args()

    filled = (
        pd.read_excel(args.worklist)
        if args.worklist.suffix in {".xlsx", ".xlsm"}
        else pd.read_csv(args.worklist)
    )

    missing_columns = set(LABELS) - set(filled.columns)
    if missing_columns:
        print(f"Worklist is missing label columns: {sorted(missing_columns)}")
        return 1

    complete = filled[filled[LABELS].notna().all(axis=1)].copy()
    skipped = len(filled) - len(complete)
    if complete.empty:
        print("No fully labelled rows found. Fill all twelve columns for each study.")
        return 1

    for label in LABELS:
        complete[label] = pd.to_numeric(complete[label], errors="coerce")

    bad = complete[~complete[LABELS].isin([0, 1]).all(axis=1)]
    if len(bad):
        print(f"{len(bad)} rows contain values other than 0 and 1. First few:")
        print(bad[["StudyInstanceUID", *LABELS]].head().to_string(index=False))
        return 1

    existing = load_gold()[["StudyInstanceUID", *LABELS]]
    overlap = set(existing.StudyInstanceUID) & set(complete.StudyInstanceUID)
    if overlap:
        print(f"Dropping {len(overlap)} studies already labelled in train.csv.")
        complete = complete[~complete.StudyInstanceUID.isin(overlap)]

    gold = pd.concat(
        [existing, complete[["StudyInstanceUID", *LABELS]]], ignore_index=True
    )
    gold[LABELS] = gold[LABELS].astype(int)
    gold.to_csv(args.out, index=False)

    truth = gold[LABELS].to_numpy()
    findings = truth.sum(axis=1)
    normals = int((findings == 0).sum())

    print(f"\nGold set: {len(existing)} existing + {len(complete)} new = {len(gold)} studies.")
    if skipped:
        print(f"({skipped} worklist rows were incomplete and left out.)")
    print(f"Normal studies (no positive finding): {normals}"
          f"{'  — still none, so specificity remains unmeasurable' if normals == 0 else ''}")
    print("\nPositives per label:")
    for label in LABELS:
        total = int(gold[label].sum())
        flag = "  <- thin" if min(total, len(gold) - total) < 10 else ""
        print(f"  {label:18s} {total:4d} / {len(gold)}{flag}")

    before = interval_width(len(existing), truth)
    after = interval_width(len(gold), truth)
    print(f"\n95% interval width: {before:.3f} at {len(existing)} studies "
          f"-> {after:.3f} at {len(gold)}.")
    print(f"Two extractors must now differ by about {after / 2:.2f} "
          "before the difference is real.")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
