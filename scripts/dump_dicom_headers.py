"""Summarise DICOM headers into one small CSV that is safe to share.

Run this where the full imaging data lives. It reads headers only, never pixel
data, so it is fast, and it writes a single CSV of a few megabytes that can be
committed to this repository. That lets the modelling code be written against
the real scanner mix, sequence mix and geometry without moving the images.

Direct patient identifiers are deliberately not collected. The competition data
is already de-identified; this simply avoids copying anything that looks like an
identifier into a shared file.

Usage:
    python scripts/dump_dicom_headers.py --dicom-root /path/to/train_series \
        --out data/raw/dicom_headers_sample.csv --studies 300
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pydicom

# Header fields worth knowing before designing the preprocessing pipeline.
TAGS = [
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "BodyPartExamined",
    "Laterality",
    "ImageLaterality",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "FlipAngle",
    "EchoTrainLength",
    "NumberOfAverages",
    "PixelBandwidth",
    "Rows",
    "Columns",
    "BitsStored",
    "PhotometricInterpretation",
    "RescaleIntercept",
    "RescaleSlope",
    "WindowCenter",
    "WindowWidth",
    "PatientSex",
    "PatientAge",
    "InstanceNumber",
    "NumberOfFrames",
]

# Multi-valued fields that are flattened into their own columns.
VECTOR_TAGS = ["PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient"]


def stringify(value: object) -> str:
    """Render a DICOM value as a compact, CSV-friendly string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(item) for item in value)
    return str(value).strip().replace("\n", " ")


def find_series(dicom_root: Path) -> dict[tuple[str, str], list[Path]]:
    """Group .dcm files by (study directory, series directory).

    The documented layout is <root>/<StudyInstanceUID>/<SeriesInstanceUID>/<SOP>.dcm,
    so the two parent directory names are used as the identifiers. Any other
    nesting still groups correctly, it simply labels the groups differently.
    """
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in dicom_root.rglob("*.dcm"):
        series_dir = path.parent
        study_dir = series_dir.parent
        grouped.setdefault((study_dir.name, series_dir.name), []).append(path)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/raw/dicom_headers_sample.csv"), type=Path)
    parser.add_argument(
        "--studies",
        type=int,
        default=300,
        help="Number of studies to sample; 0 means every study.",
    )
    parser.add_argument(
        "--slices-per-series",
        type=int,
        default=2,
        help="Slices read per series: the first, then evenly spaced through the stack.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dicom_root.is_dir():
        print(f"Not a directory: {args.dicom_root}")
        return 1

    print(f"Scanning {args.dicom_root} for DICOM files …")
    grouped = find_series(args.dicom_root)
    if not grouped:
        print("No .dcm files found. Check --dicom-root.")
        return 1

    studies = sorted({study for study, _ in grouped})
    print(f"Found {len(grouped):,} series across {len(studies):,} studies.")

    if args.studies and args.studies < len(studies):
        random.Random(args.seed).shuffle(studies)
        studies = sorted(studies[: args.studies])
        print(f"Sampling {len(studies):,} studies.")
    chosen = {study for study in studies}

    columns = (
        ["study_dir", "series_dir", "slices_in_series", "file_name"]
        + TAGS
        + [f"{tag}_{index}" for tag in VECTOR_TAGS for index in range(6)]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    failures = 0

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for (study, series), paths in sorted(grouped.items()):
            if study not in chosen:
                continue
            paths.sort()
            count = max(1, min(args.slices_per_series, len(paths)))
            step = max(1, len(paths) // count)
            selected = paths[::step][:count]

            for path in selected:
                row = {
                    "study_dir": study,
                    "series_dir": series,
                    "slices_in_series": len(paths),
                    "file_name": path.name,
                }
                try:
                    dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                except Exception:  # noqa: BLE001 - a broken file must not stop the scan
                    failures += 1
                    continue

                for tag in TAGS:
                    row[tag] = stringify(getattr(dataset, tag, None))
                for tag in VECTOR_TAGS:
                    values = getattr(dataset, tag, None) or []
                    for index in range(6):
                        row[f"{tag}_{index}"] = (
                            stringify(values[index]) if index < len(values) else ""
                        )

                writer.writerow(row)
                rows_written += 1

            if rows_written and rows_written % 2000 == 0:
                print(f"  … {rows_written:,} rows")

    size_mb = args.out.stat().st_size / 1024**2
    print(f"Wrote {rows_written:,} rows to {args.out} ({size_mb:.1f} MB).")
    if failures:
        print(f"{failures:,} files could not be read and were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
