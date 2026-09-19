"""Characterise the radiology reports and the 58-study gold set.

The reports are the training signal for 98% of the data, so their language mix
and the representativeness of the gold set decide how the label extractor must
be built and how far its measured accuracy can be trusted.

Usage:
    python scripts/analyse_reports.py [--out reports/report_analysis.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.load_data import LABELS, load_gold, load_train  # noqa: E402

# Language identification is approximate. Short reports and closely related
# languages are the usual failure cases, so small counts should not be trusted
# individually.
MIN_COUNT_TO_REPORT = 20

# Detector codes that are almost certainly misfires on a related language.
# Extremaduran, Aragonese, Galician and Latin do not appear in radiology.
IMPLAUSIBLE_CODES = {"ext", "an", "gl", "la"}


def detect_languages(reports: pd.Series) -> pd.Series:
    """Identify the language of each report, or return an empty series."""
    try:
        import py3langid
    except ImportError:
        print("py3langid is not installed; skipping language detection.")
        print("Install it with: pip install py3langid")
        return pd.Series(dtype="object")
    # Only the opening of each report is needed, and it keeps this quick.
    return reports.astype(str).map(lambda text: py3langid.classify(text[:2000])[0])


def language_section(train: pd.DataFrame, gold: pd.DataFrame) -> list[str]:
    if "lang" not in train.columns:
        return []

    lines = ["## Language of the reports", ""]
    counts = train.lang.value_counts()
    lines += [
        f"{len(counts)} languages detected across {len(train):,} studies.",
        "",
        "| Language | Studies | Share | Gold studies |",
        "| --- | ---: | ---: | ---: |",
    ]
    gold_counts = gold.lang.value_counts() if "lang" in gold.columns else pd.Series(dtype=int)
    for code, count in counts.items():
        if count < MIN_COUNT_TO_REPORT:
            continue
        note = " ⚠" if code in IMPLAUSIBLE_CODES else ""
        lines.append(
            f"| `{code}`{note} | {count:,} | {count / len(train):.1%} "
            f"| {int(gold_counts.get(code, 0))} |"
        )
    tail = counts[counts < MIN_COUNT_TO_REPORT]
    if len(tail):
        lines.append(
            f"| *{len(tail)} rarer* | {tail.sum():,} | {tail.sum() / len(train):.1%} | — |"
        )

    missing = sorted(set(counts.index) - set(gold_counts.index))
    if missing:
        share = counts[missing].sum() / len(train)
        lines += [
            "",
            f"**{len(missing)} languages have no gold example at all** "
            f"(`{'`, `'.join(missing)}`), covering {counts[missing].sum():,} studies "
            f"({share:.1%}). Extraction quality on those is unmeasurable.",
        ]

    lines += [
        "",
        "Entries marked ⚠ are near-certain misidentifications of a related "
        "language, which is a reminder that these counts are approximate.",
        "",
    ]
    return lines


def gold_section(gold: pd.DataFrame) -> list[str]:
    findings = gold[LABELS].sum(axis=1)
    lines = [
        "## The gold set",
        "",
        f"{len(gold)} studies, each carrying all twelve labels — "
        f"{len(gold) * len(LABELS):,} label cells in total.",
        "",
        "| Label | Positive | Rate |",
        "| --- | ---: | ---: |",
    ]
    for label in sorted(LABELS, key=lambda name: -gold[name].mean()):
        lines.append(f"| {label} | {int(gold[label].sum())} | {gold[label].mean():.1%} |")

    all_negative = int((findings == 0).sum())
    lines += [
        "",
        f"Findings per study: mean {findings.mean():.2f}, median "
        f"{findings.median():.0f}, maximum {int(findings.max())}. "
        f"Studies with no positive finding: **{all_negative}**.",
        "",
    ]
    if all_negative == 0:
        lines += [
            "Not one of these studies is normal. The gold set can therefore show "
            "whether an extractor finds the abnormalities that are present, but "
            "it cannot show how often the extractor invents findings in a normal "
            "report — and specificity is exactly what a macro-AUC metric "
            "punishes. Any claim about false positives needs a separate, "
            "deliberately normal sample.",
            "",
        ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=Path("reports/report_analysis.md"), type=Path)
    args = parser.parse_args()

    train = load_train()
    gold = load_gold()

    languages = detect_languages(train.Report)
    if not languages.empty:
        train = train.assign(lang=languages)
        gold = gold.merge(train[["StudyInstanceUID", "lang"]], on="StudyInstanceUID", how="left")

    lengths = train.Report.astype(str).str.len()
    report = [
        "# Report analysis",
        "",
        f"{len(train):,} studies, of which {len(gold)} are labelled "
        f"({len(gold) / len(train):.1%}).",
        "",
        f"Report length in characters: median {int(lengths.median()):,}, "
        f"shortest {lengths.min():,}, longest {lengths.max():,}.",
        "",
    ]
    report += language_section(train, gold)
    report += gold_section(gold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
