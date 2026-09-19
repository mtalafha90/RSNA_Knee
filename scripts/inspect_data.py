"""Sniff the schema of whatever competition CSVs are present.

Nothing here assumes a particular file name or column name. The point is to
discover the real schema of the downloaded data before any modelling code is
written, so that later scripts can be written against facts rather than
guesses.

Usage:
    python scripts/inspect_data.py [--raw-dir data/raw] [--out reports/schema_report.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# A column is treated as free text once its typical length exceeds this.
TEXT_LENGTH_THRESHOLD = 80

# A column is treated as categorical below this many distinct values.
CATEGORICAL_MAX_CARDINALITY = 30

# Show at most this many distinct values for a categorical column.
MAX_VALUES_SHOWN = 15


def classify_column(series: pd.Series) -> str:
    """Return a coarse role for a column: id, binary, categorical, text or numeric."""
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    name_hint = series.name.lower() if isinstance(series.name, str) else ""
    looks_like_id = name_hint.endswith("uid") or name_hint.endswith("_id") or name_hint == "id"
    if looks_like_id:
        return "id"

    unique_values = set(pd.unique(non_null))
    if unique_values <= {0, 1, 0.0, 1.0, True, False}:
        return "binary"

    if pd.api.types.is_numeric_dtype(non_null):
        return "numeric"

    if pd.api.types.is_string_dtype(non_null):
        typical_length = non_null.astype(str).str.len().median()
        if typical_length > TEXT_LENGTH_THRESHOLD:
            return "text"

    if non_null.nunique() <= CATEGORICAL_MAX_CARDINALITY:
        return "categorical"

    # Nearly every value distinct, and not numeric: almost certainly an identifier.
    if non_null.nunique() > 0.95 * len(non_null):
        return "id"

    return "other"


def describe_column(frame: pd.DataFrame, column: str) -> list[str]:
    """Produce the report lines for a single column."""
    series = frame[column]
    role = classify_column(series)
    null_count = int(series.isna().sum())
    null_share = null_count / len(frame) if len(frame) else 0.0

    lines = [
        f"- **`{column}`** — role: *{role}*, dtype: `{series.dtype}`, "
        f"distinct: {series.nunique(dropna=True):,}, "
        f"missing: {null_count:,} ({null_share:.1%})"
    ]

    non_null = series.dropna()
    if non_null.empty:
        return lines

    if role == "binary":
        positives = int(non_null.astype(float).sum())
        lines.append(f"  - positives: {positives:,} ({positives / len(non_null):.2%} of non-missing)")
    elif role == "numeric":
        stats = non_null.describe()
        lines.append(
            f"  - min {stats['min']:.4g}, median {non_null.median():.4g}, "
            f"max {stats['max']:.4g}, mean {stats['mean']:.4g}"
        )
    elif role == "categorical":
        counts = non_null.value_counts().head(MAX_VALUES_SHOWN)
        rendered = ", ".join(f"`{value}` ({count:,})" for value, count in counts.items())
        overflow = non_null.nunique() - len(counts)
        if overflow > 0:
            rendered += f", … and {overflow:,} more"
        lines.append(f"  - values: {rendered}")
    elif role == "text":
        lengths = non_null.astype(str).str.len()
        lines.append(
            f"  - length in characters: min {lengths.min():,}, "
            f"median {int(lengths.median()):,}, max {lengths.max():,}"
        )
        excerpt = str(non_null.iloc[0])[:300].replace("\n", " ⏎ ")
        lines.append(f"  - first value (truncated): `{excerpt}`")
    else:
        examples = ", ".join(f"`{value}`" for value in non_null.unique()[:3])
        lines.append(f"  - examples: {examples}")

    return lines


def describe_file(path: Path) -> list[str]:
    """Produce the report section for a single CSV or Parquet file."""
    lines = [f"## `{path.name}`", ""]
    try:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception as error:  # noqa: BLE001 - report and carry on with other files
        lines += [f"Could not be read: `{error}`", ""]
        return lines

    size_mb = path.stat().st_size / 1024**2
    lines += [
        f"{len(frame):,} rows × {len(frame.columns):,} columns ({size_mb:.1f} MB on disk).",
        "",
    ]

    for column in frame.columns:
        lines += describe_column(frame, column)

    roles = {column: classify_column(frame[column]) for column in frame.columns}
    binary_columns = [column for column, role in roles.items() if role == "binary"]
    id_columns = [column for column, role in roles.items() if role == "id"]

    if len(binary_columns) >= 8:
        lines += ["", "### Looks like the label table", ""]
        prevalence = frame[binary_columns].mean().sort_values(ascending=False)
        lines += ["| label | prevalence | positives |", "| --- | ---: | ---: |"]
        for label, rate in prevalence.items():
            lines.append(f"| {label} | {rate:.2%} | {int(frame[label].sum()):,} |")
        findings_per_study = frame[binary_columns].sum(axis=1)
        lines += [
            "",
            f"Findings per study: mean {findings_per_study.mean():.2f}, "
            f"median {findings_per_study.median():.0f}, max {findings_per_study.max():.0f}. "
            f"Studies with no positive finding: {(findings_per_study == 0).sum():,} "
            f"({(findings_per_study == 0).mean():.1%}).",
        ]

    if len(id_columns) >= 2:
        parent, child = id_columns[0], id_columns[1]
        children_per_parent = frame.groupby(parent)[child].nunique()
        lines += [
            "",
            f"### Nesting: `{child}` within `{parent}`",
            "",
            f"{frame[parent].nunique():,} distinct `{parent}` values, "
            f"{frame[child].nunique():,} distinct `{child}` values.",
            f"Per `{parent}`: min {children_per_parent.min()}, "
            f"median {children_per_parent.median():.0f}, "
            f"mean {children_per_parent.mean():.2f}, max {children_per_parent.max()}.",
        ]

    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--out", default="reports/schema_report.md", type=Path)
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.raw_dir.rglob("*")
        if path.suffix in {".csv", ".parquet"} and path.is_file()
    )
    if not files:
        print(f"No CSV or Parquet files found under {args.raw_dir}/.")
        print("Place the competition metadata files there and run this script again.")
        return 1

    report = ["# Schema report", "", f"Files found under `{args.raw_dir}/`: {len(files)}.", ""]
    for path in files:
        report += describe_file(path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
