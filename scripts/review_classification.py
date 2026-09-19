"""Score a candidate label set against the 58 gold studies.

Given a spreadsheet or CSV holding the twelve labels for training studies, this
answers the only question that matters about it: how well would those labels
serve as the training target for the imaging model?

The headline number is the macro-AUC the labels would score if submitted
directly as predictions. Because the labels are binary, each label's AUC is
just the mean of its sensitivity and specificity.

Usage:
    python scripts/review_classification.py data/train_classified.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.load_data import LABELS, load_gold, load_train  # noqa: E402
from scripts.metrics import bootstrap_macro_auc  # noqa: E402


def read_candidate(path: Path, sheet: str | None) -> pd.DataFrame:
    """Read a candidate label set from a spreadsheet, a CSV, or a verdicts file.

    A .jsonl file is taken to be the output of extract_labels.py and its
    verdicts are mapped to probabilities on the way in.
    """
    if path.suffix == ".jsonl":
        from scripts.extract_labels import verdicts_to_probabilities

        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {"StudyInstanceUID": record["StudyInstanceUID"],
                 **verdicts_to_probabilities(record["verdicts"])}
            )
        return pd.DataFrame(rows)
    if path.suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def score_against_gold(candidate: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Per-label confusion counts and the resulting binary AUC."""
    merged = gold.merge(
        candidate[["StudyInstanceUID", *LABELS]],
        on="StudyInstanceUID",
        suffixes=("_gold", "_pred"),
    )
    rows = []
    for label in LABELS:
        truth = merged[f"{label}_gold"].astype(int)
        predicted = merged[f"{label}_pred"].astype(int)
        tp = int(((truth == 1) & (predicted == 1)).sum())
        fn = int(((truth == 1) & (predicted == 0)).sum())
        fp = int(((truth == 0) & (predicted == 1)).sum())
        tn = int(((truth == 0) & (predicted == 0)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        rows.append(
            {
                "label": label,
                "gold positives": tp + fn,
                "predicted": tp + fp,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "AUC": (sensitivity + specificity) / 2,
            }
        )
    return pd.DataFrame(rows)


def score_by_language(candidate: pd.DataFrame, language_column: str) -> pd.DataFrame:
    """How many findings the labels assign, broken down by report language.

    A language whose studies are almost all finding-free is a language the
    extractor cannot read. That is worse than leaving those studies unlabelled,
    because an all-negative target actively teaches the imaging model that the
    abnormalities in those scans are absent.
    """
    frame = candidate.copy()
    frame["findings"] = frame[LABELS].sum(axis=1)
    grouped = frame.groupby(language_column).agg(
        studies=("findings", "size"),
        mean_findings=("findings", "mean"),
        finding_free=("findings", lambda values: (values == 0).mean()),
    )
    return grouped.sort_values("studies", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--language-column", default="Language")
    parser.add_argument("--out", default=Path("reports/classification_review.md"), type=Path)
    args = parser.parse_args()

    candidate = read_candidate(args.candidate, args.sheet)
    gold = load_gold()
    train = load_train()

    missing = set(LABELS) - set(candidate.columns)
    if missing:
        print(f"Candidate is missing label columns: {sorted(missing)}")
        return 1

    merged = gold.merge(
        candidate[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID",
        suffixes=("_gold", "_pred"),
    )
    truth = merged[[f"{label}_gold" for label in LABELS]].to_numpy().astype(int)
    predicted = merged[[f"{label}_pred" for label in LABELS]].to_numpy().astype(float)
    interval = bootstrap_macro_auc(truth, predicted)

    is_binary = set(pd.unique(predicted.ravel())) <= {0.0, 1.0}
    covered = len(set(train.StudyInstanceUID) & set(candidate.StudyInstanceUID))

    lines = [
        "# Classification review",
        "",
        f"Candidate: `{args.candidate.name}` — {len(candidate):,} rows, "
        f"covering {covered:,} of the {len(train):,} training studies. "
        f"{len(merged)} gold studies matched.",
        "",
        f"## Macro-AUC: **{interval['macro_auc']:.3f}**  "
        f"(95% interval {interval['ci_low']:.3f} to {interval['ci_high']:.3f})",
        "",
        "0.500 is a coin flip. The interval comes from resampling studies, and it is "
        f"{interval['ci_width']:.3f} wide — so a rival label set must beat this by "
        f"more than about {interval['ci_width'] / 2:.2f} before the difference is "
        "real rather than noise.",
        "",
    ]

    if is_binary:
        lines += [
            "The candidate is binary 0/1, so each AUC below is the mean of that "
            "label's sensitivity and specificity. Graded confidences would score "
            "higher at identical decisions — see `docs/findings-02-classification-review.md`.",
            "",
            score_against_gold(candidate, gold).to_markdown(index=False, floatfmt=".3f"),
            "",
        ]

    if args.language_column in candidate.columns:
        by_language = score_by_language(candidate, args.language_column)
        lines += [
            "## Findings assigned, by report language",
            "",
            by_language.to_markdown(floatfmt=".2f"),
            "",
        ]
        unreadable = by_language[by_language.finding_free > 0.75]
        if len(unreadable):
            affected = int(unreadable.studies.sum())
            lines += [
                f"**{len(unreadable)} languages look unreadable to this extractor**, "
                f"covering {affected:,} studies ({affected / len(candidate):.1%}): "
                f"{', '.join(unreadable.index)}. More than three quarters of their "
                "studies come out with no finding at all.",
                "",
            ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
