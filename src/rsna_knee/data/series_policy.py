"""Which of a study's series the model may look at, and the fingerprint that pins the rule."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
import json
import numpy as np
import pandas as pd

from ..constants import PLANES, PLANE_TO_ID
from .tables import build_series_index


FROZEN_SERIES_SIGNATURE = "5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376"


def series_signature(index: dict[str, list[dict]], studies: Iterable[str]) -> str:
    rows = []
    for uid in studies:
        key = str(uid)
        rows.append(
            {
                "StudyInstanceUID": key,
                "series": [
                    {
                        "series_uid": str(r["series_uid"]),
                        "plane_id": int(r["plane_id"]),
                        "fluid_id": int(r["fluid_id"]),
                        "fat_id": int(r["fat_id"]),
                    }
                    for r in index.get(key, [])
                ],
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SERIES_POLICY_NAME = "all_repaired_anatomical_series_v1"


def _flag_id(value) -> int:
    """0=unknown, 1=false, 2=true."""
    if pd.isna(value):
        return 0
    return 2 if bool(value) else 1


def load_series_policy(policy_path: str | Path) -> dict:
    path = Path(policy_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("policy") != SERIES_POLICY_NAME:
        raise ValueError("B12 series policy name mismatch")
    if payload.get("uses_gold_labels") is not False:
        raise ValueError("B12 series policy does not certify label-free derivation")
    if payload.get("viability_passed") is not True:
        raise ValueError("B12 series viability audit did not pass")
    if int(payload.get("b6_active_studies", -1)) != 3120 or int(payload.get("b6_usable_cells", -1)) != 14123:
        raise ValueError("B12 series policy was not frozen on the retained B7.1 supervision surface")
    return payload


def build_series_index_all_planes(
    series_df: pd.DataFrame,
    studies: Iterable[str],
) -> dict[str, list[dict]]:
    """Return every repaired series with a recognized anatomical plane.

    No fluid/structural winner is selected. Ordering is deterministic only for
    reproducibility; the B12 model has no series-position embedding.
    """
    work = series_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    work["SeriesInstanceUID"] = work["SeriesInstanceUID"].astype(str)
    grouped = {uid: part for uid, part in work.groupby("StudyInstanceUID", sort=False)}
    empty = work.iloc[0:0]
    result: dict[str, list[dict]] = {}
    for study in studies:
        uid = str(study)
        records: list[dict] = []
        part = grouped.get(uid, empty)
        for _, row in part.iterrows():
            plane = str(row.get("Anatomical_Plane", ""))
            plane_id = PLANE_TO_ID.get(plane, 0)
            if plane_id == 0:
                continue
            records.append(
                {
                    "series_uid": str(row["SeriesInstanceUID"]),
                    "plane": plane,
                    "plane_id": int(plane_id),
                    "fluid_id": _flag_id(row.get("Fluid_Sensitive")),
                    "fat_id": _flag_id(row.get("Fat_Suppression")),
                }
            )
        records.sort(
            key=lambda x: (
                int(x["plane_id"]),
                int(x["fluid_id"]),
                int(x["fat_id"]),
                str(x["series_uid"]),
            )
        )
        result[uid] = records
    return result


def audit_series_surface(
    series_df: pd.DataFrame,
    studies: Iterable[str],
) -> tuple[dict, dict[str, list[dict]]]:
    uids = [str(x) for x in studies]
    variable = build_series_index_all_planes(series_df, uids)
    legacy = build_series_index(series_df, uids, mode="dual")

    counts = np.asarray([len(variable[uid]) for uid in uids], dtype=np.int64)
    legacy_counts = []
    missing_legacy = 0
    extra_counts = []
    for uid in uids:
        legacy_set = {str(v) for v in legacy[uid].values() if v}
        variable_set = {str(r["series_uid"]) for r in variable[uid]}
        missing_legacy += len(legacy_set.difference(variable_set))
        legacy_counts.append(len(legacy_set))
        extra_counts.append(len(variable_set.difference(legacy_set)))
    legacy_counts_arr = np.asarray(legacy_counts, dtype=np.int64)
    extra_arr = np.asarray(extra_counts, dtype=np.int64)

    study_set = set(uids)
    relevant = series_df.loc[series_df["StudyInstanceUID"].astype(str).isin(study_set)]
    recognized = relevant["Anatomical_Plane"].isin(PLANES)
    unknown_plane = int((~recognized).sum())
    eligible_total = int(counts.sum())
    legacy_total = int(legacy_counts_arr.sum())
    extra_total = int(extra_arr.sum())
    extra_fraction = float(extra_total / max(legacy_total, 1))
    study_extra_fraction = float((extra_arr > 0).mean()) if len(extra_arr) else 0.0

    quantile_levels = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    q = np.quantile(counts, quantile_levels).tolist() if len(counts) else [0.0] * len(quantile_levels)
    summary = {
        "policy": SERIES_POLICY_NAME,
        "studies": int(len(uids)),
        "series_rows_for_studies": int(len(relevant)),
        "eligible_recognized_plane_series": eligible_total,
        "excluded_unknown_plane_series": unknown_plane,
        "historical_dual_unique_series": legacy_total,
        "extra_series_retained": extra_total,
        "extra_series_fraction_vs_historical_unique": extra_fraction,
        "studies_with_extra_series": int((extra_arr > 0).sum()),
        "studies_with_extra_series_fraction": study_extra_fraction,
        "studies_with_zero_eligible_series": int((counts == 0).sum()),
        "historical_selected_series_missing_from_b12": int(missing_legacy),
        "series_per_study": {
            "min": int(counts.min()) if len(counts) else 0,
            "mean": float(counts.mean()) if len(counts) else 0.0,
            "q25": float(q[1]),
            "median": float(q[2]),
            "q75": float(q[3]),
            "q90": float(q[4]),
            "q95": float(q[5]),
            "q99": float(q[6]),
            "max": int(counts.max()) if len(counts) else 0,
        },
    }
    summary["viability_passed"] = bool(
        summary["studies_with_zero_eligible_series"] == 0
        and summary["historical_selected_series_missing_from_b12"] == 0
        and extra_fraction >= 0.05
        and study_extra_fraction >= 0.10
    )
    summary["viability_requirements"] = {
        "zero_studies_without_eligible_series": True,
        "zero_historical_selected_series_missing": True,
        "minimum_extra_series_fraction_vs_historical_unique": 0.05,
        "minimum_fraction_studies_with_extra_series": 0.10,
    }
    summary["series_signature_sha256"] = series_signature(variable, uids)
    return summary, variable

