"""Build a spreadsheet of studies to hand-label, to enlarge the gold set.

The gold set is 58 studies and every one of them is abnormal. That is too few
to tell two extractors apart, and with no normal studies it cannot measure how
often an extractor invents findings. Both problems are fixed by labelling more
reports.

Selection is a random sample stratified by report language, not a hand-picked
one. Two reasons. Picking studies that "look normal" would bias the sample and
make the resulting specificity meaningless. And stratifying by language
guarantees every language gets gold examples — French currently has none, so
extraction quality on those 81 studies is unmeasurable.

Studies already labelled in train.csv are excluded.

Usage:
    python scripts/make_worklist.py --n 120 --out data/worklist.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.load_data import LABELS, load_gold, load_train  # noqa: E402

# Every language gets at least this many studies, however rare it is, so that
# none is left with an unmeasurable extraction quality.
MIN_PER_LANGUAGE = 6

# Bosnian and Croatian are one language for this purpose, and the detector's
# rarest codes — Extremaduran, Aragonese, Galician, Latin — are misreadings of
# Spanish or Italian rather than real strata. Grouping them stops the sample
# spending its floor allocation on detector artefacts.
LANGUAGE_GROUPS = {
    "bs": "hbs",
    "hr": "hbs",
    "ext": "other",
    "an": "other",
    "gl": "other",
    "la": "other",
}


def detect_languages(reports: pd.Series) -> pd.Series:
    try:
        import py3langid
    except ImportError:
        print("py3langid is not installed; sampling without stratifying.")
        print("Install it with: pip install py3langid")
        return pd.Series("unknown", index=reports.index)
    detected = reports.astype(str).map(lambda text: py3langid.classify(text[:2000])[0])
    return detected.map(lambda code: LANGUAGE_GROUPS.get(code, code))


def allocate(counts: pd.Series, total: int) -> dict[str, int]:
    """Split `total` studies across languages, proportionally but with a floor."""
    languages = list(counts.index)
    allocation = {language: min(MIN_PER_LANGUAGE, counts[language]) for language in languages}

    remaining = total - sum(allocation.values())
    if remaining > 0:
        headroom = {lang: counts[lang] - allocation[lang] for lang in languages}
        available = sum(headroom.values())
        for language in languages:
            if available <= 0:
                break
            share = round(remaining * counts[language] / counts.sum())
            allocation[language] += min(share, headroom[language])

    # Trim if rounding overshot, taking from the largest groups first.
    while sum(allocation.values()) > total:
        biggest = max(allocation, key=lambda lang: allocation[lang])
        if allocation[biggest] <= MIN_PER_LANGUAGE:
            break
        allocation[biggest] -= 1

    return allocation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120, help="How many studies to select.")
    parser.add_argument("--out", default=Path("data/worklist.xlsx"), type=Path)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    train = load_train()
    already_labelled = set(load_gold().StudyInstanceUID)
    pool = train[~train.StudyInstanceUID.isin(already_labelled)].copy()
    pool["Language"] = detect_languages(pool.Report)

    counts = pool.Language.value_counts()
    allocation = allocate(counts, args.n)

    selected = pd.concat(
        [
            pool[pool.Language == language].sample(
                n=count, random_state=args.seed + index
            )
            for index, (language, count) in enumerate(allocation.items())
            if count > 0
        ]
    )

    worklist = pd.DataFrame(
        {
            "StudyInstanceUID": selected.StudyInstanceUID.values,
            "Language": selected.Language.values,
            "Report": selected.Report.values,
        }
    )
    for label in LABELS:
        worklist[label] = ""
    worklist["notes"] = ""

    # Shuffle, so that working through the file in order does not mean working
    # through one language at a time — which would let fatigue land unevenly.
    worklist = worklist.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix in {".xlsx", ".xlsm"}:
        worklist.to_excel(args.out, index=False)
    else:
        worklist.to_csv(args.out, index=False)

    print(f"{len(worklist)} studies selected, excluding the {len(already_labelled)} already labelled.\n")
    print(f"{'language':>10}  {'in corpus':>10}  {'selected':>9}")
    for language, count in counts.items():
        chosen = allocation.get(language, 0)
        if chosen:
            print(f"{language:>10}  {count:>10,}  {chosen:>9}")
    print(f"\nWritten to {args.out}")
    print("\nFill every one of the twelve columns with 1 or 0 — do not leave blanks,")
    print("and do not skip a study because it looks normal: the normal ones are")
    print("exactly what the gold set is missing. Then run:")
    print(f"  python scripts/merge_gold.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
