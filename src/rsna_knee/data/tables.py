"""The competition tables, and the metadata repair that fills their blanks from the DICOM."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from ..constants import PLANES, TARGETS
from ..imaging.dicom_io import find_series_dir
from ..imaging.dicom_metadata import read_series_metadata


def backfill_series_metadata(
    series_df: pd.DataFrame,
    data_root: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Independently repair missing plane, fluid and fat-suppression metadata."""
    df = series_df.copy()
    missing_plane = df["Anatomical_Plane"].astype(str).str.strip().eq("")
    missing_fluid = df["Fluid_Sensitive"].isna()
    missing_fat = df["Fat_Suppression"].isna()
    needs = missing_plane | missing_fluid | missing_fat
    targets = df.index[needs]
    if limit is not None:
        targets = targets[: int(limit)]

    stats = {
        "rows_needing_metadata": int(needs.sum()),
        "missing_plane": int(missing_plane.sum()),
        "missing_fluid": int(missing_fluid.sum()),
        "missing_fat_suppression": int(missing_fat.sum()),
        "inspected": len(targets),
        "repaired_plane": 0,
        "repaired_fluid": 0,
        "repaired_fat_suppression": 0,
    }
    for index in targets:
        row = df.loc[index]
        series_dir = find_series_dir(data_root, split, str(row["StudyInstanceUID"]), str(row["SeriesInstanceUID"]))
        if series_dir is None:
            continue
        metadata = read_series_metadata(series_dir)
        if missing_plane.loc[index] and metadata["Anatomical_Plane"] is not None:
            df.at[index, "Anatomical_Plane"] = metadata["Anatomical_Plane"]
            stats["repaired_plane"] += 1
        if missing_fluid.loc[index] and metadata["Fluid_Sensitive"] is not None:
            df.at[index, "Fluid_Sensitive"] = bool(metadata["Fluid_Sensitive"])
            stats["repaired_fluid"] += 1
        if missing_fat.loc[index] and metadata["Fat_Suppression"] is not None:
            df.at[index, "Fat_Suppression"] = bool(metadata["Fat_Suppression"])
            stats["repaired_fat_suppression"] += 1
    return df, stats


def gold_mask(df: pd.DataFrame) -> pd.Series:
    return df[TARGETS].notna().any(axis=1)


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def normalise_plane(values: pd.Series) -> pd.Series:
    mapping = {
        "sagittal": "Sagittal", "sag": "Sagittal", "sagital": "Sagittal",
        "coronal": "Coronal", "cor": "Coronal",
        "axial": "Axial", "ax": "Axial", "transverse": "Axial",
    }
    text = values.astype("string").str.strip().str.lower()
    return text.map(mapping).fillna("").astype(str)


FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}


TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}


def _rank_indices(score: np.ndarray) -> list[int]:
    return np.argsort(-score, kind="mergesort").astype(int).tolist()


def load_train_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, {"StudyInstanceUID", "Report", *TARGETS}, "train.csv")
    if df["StudyInstanceUID"].isna().any():
        raise ValueError("train.csv contains missing StudyInstanceUID values")
    df = df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("train.csv contains duplicate StudyInstanceUID values")
    return df


def coerce_bool(values: pd.Series, *, preserve_unknown: bool = False) -> pd.Series:
    """Parse metadata flags without converting missing/unknown values to True.

    ``preserve_unknown=True`` returns pandas' nullable Boolean dtype so DICOM
    metadata can repair unknown sequence flags before routing. The public helper
    keeps the historical conservative default of mapping unknown values to False.
    """
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    if pd.api.types.is_bool_dtype(values):
        result.loc[values.notna()] = values.loc[values.notna()].astype(bool)
    elif pd.api.types.is_numeric_dtype(values):
        known = values.notna()
        result.loc[known] = values.loc[known].astype(float).ne(0.0)
    else:
        text = values.astype("string").str.strip().str.lower()
        result.loc[text.isin(TRUE_TOKENS)] = True
        result.loc[text.isin(FALSE_TOKENS)] = False
    return result if preserve_unknown else result.fillna(False).astype(bool)


def _select_from_study(part: pd.DataFrame, mode: str) -> dict[str, str | None]:
    if mode not in {"best", "dual"}:
        raise ValueError("stream mode must be 'best' or 'dual'")
    result: dict[str, str | None] = {}
    for plane in PLANES:
        p = part.loc[part["Anatomical_Plane"].eq(plane)].reset_index(drop=True)
        key = plane.lower()
        if p.empty:
            if mode == "dual":
                result[f"{key}_fluid"] = None
                result[f"{key}_structural"] = None
            else:
                result[key] = None
            continue

        fluid_flag = p["Fluid_Sensitive"].fillna(False).astype(bool).to_numpy()
        fat_flag = p["Fat_Suppression"].fillna(False).astype(bool).to_numpy()
        fluid_score = 2 * fluid_flag.astype(int) + 2 * fat_flag.astype(int)
        structural_score = 2 * (~fat_flag).astype(int) + (~fluid_flag).astype(int)

        if mode == "best":
            result[key] = p.at[_rank_indices(fluid_score + 0.25 * structural_score)[0], "SeriesInstanceUID"]
            continue
        if len(p) == 1:
            uid = p.at[0, "SeriesInstanceUID"]
            if fluid_score[0] >= structural_score[0]:
                result[f"{key}_fluid"], result[f"{key}_structural"] = uid, None
            else:
                result[f"{key}_fluid"], result[f"{key}_structural"] = None, uid
            continue

        fluid_idx = _rank_indices(fluid_score)[0]
        fluid_uid = p.at[fluid_idx, "SeriesInstanceUID"]
        structural_idx = _rank_indices(structural_score)[0]
        if p.at[structural_idx, "SeriesInstanceUID"] == fluid_uid:
            for candidate in _rank_indices(structural_score)[1:]:
                if p.at[candidate, "SeriesInstanceUID"] != fluid_uid:
                    structural_idx = candidate
                    break
        result[f"{key}_fluid"] = fluid_uid
        result[f"{key}_structural"] = p.at[structural_idx, "SeriesInstanceUID"]
    return result


def load_series_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"}
    _require_columns(df, required, "series CSV")
    if df[["StudyInstanceUID", "SeriesInstanceUID"]].isna().any().any():
        raise ValueError("series CSV contains missing study/series UID values")
    df = df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
    df["Fluid_Sensitive"] = coerce_bool(df["Fluid_Sensitive"], preserve_unknown=True)
    df["Fat_Suppression"] = coerce_bool(df["Fat_Suppression"], preserve_unknown=True)
    df["Anatomical_Plane"] = normalise_plane(df["Anatomical_Plane"])
    if df[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any():
        raise ValueError("series CSV contains duplicate study/series rows")
    return df


def build_series_index(series_df: pd.DataFrame, studies: Iterable[str], mode: str = "dual") -> dict[str, dict[str, str | None]]:
    if mode not in {"best", "dual"}:
        raise ValueError("stream mode must be 'best' or 'dual'")
    work = series_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    grouped = {uid: part for uid, part in work.groupby("StudyInstanceUID", sort=False)}
    empty = work.iloc[0:0]
    return {str(uid): _select_from_study(grouped.get(str(uid), empty), mode) for uid in studies}

