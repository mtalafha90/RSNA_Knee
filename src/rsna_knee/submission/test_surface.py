"""The test studies, their series, and whether their pixels are on this machine."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..data.coverage import study_coverage
from ..data.series_policy import audit_series_surface
from ..data.tables import backfill_series_metadata, load_series_csv


def load_test_csv(path: str | Path) -> pd.DataFrame:
    """The test manifest: study UIDs and nothing else that matters.

    Deliberately not `load_train_csv`, which requires the twelve label columns.
    A test table carrying labels would mean the wrong file was passed.
    """
    frame = pd.read_csv(path)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError(f"{path} has no StudyInstanceUID column")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path} lists a study more than once")
    if not len(frame):
        raise ValueError(f"{path} contains no studies")
    return frame


def build_test_surface(config: dict, data_root: str | Path) -> dict:
    """Every test study, the series it may be read from, and the coverage check.

    A study whose series all have an unrecognised plane ends up with none. On
    the visible three-study example that cannot happen; across a hidden set it
    is likely, so the count is returned rather than raised on, and the caller
    decides according to its unreadable-study policy.
    """
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)

    test = load_test_csv(root / settings.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].tolist()

    series = load_series_csv(root / settings.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    _summary, index = audit_series_surface(series, uids)

    counts = [len(index.get(uid, [])) for uid in uids]
    without_series = [uid for uid, count in zip(uids, counts) if count == 0]

    return {
        "settings": settings,
        "root": root,
        "uids": uids,
        "index": index,
        "counts": counts,
        "studies_without_series": without_series,
        "metadata_repair": metadata_stats,
        # Reported, not raised on. Refusing to start because one hidden study's
        # folders are absent would forfeit the other 1,299; the launcher decides
        # by its unreadable-study policy, and records what it found either way.
        "coverage": study_coverage(root, index, split="test"),
    }


__all__ = ["build_test_surface", "load_test_csv"]
