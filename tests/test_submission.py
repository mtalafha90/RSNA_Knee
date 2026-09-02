"""A model trained here must be submittable from here, and identically.

Until now every inference launcher lived in the archive this package was
extracted from, so a run here had to be carried back there to be scored --
through a path that failed three hidden runs, each on something the three
visible example studies could not reveal.

The property that makes a rebuilt path safe rather than merely convenient is
that it shows the model the same tensors. A streamed view differing from a
materialised one by a rounding step would still produce a plausible submission
and a plausible score.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pydicom")

from rsna_knee.constants import TARGETS  # noqa: E402
from rsna_knee.imaging.dicom_io import _normalise_volume  # noqa: E402
from rsna_knee.imaging.triplets import (  # noqa: E402
    prepare_series_triplets,
    series_triplets_from_normalised,
)
from rsna_knee.submission import runner  # noqa: E402
from rsna_knee.submission.runner import (  # noqa: E402
    FALLBACK_PROBABILITY,
    ON_UNREADABLE_FALLBACK,
    ON_UNREADABLE_RAISE,
    infer_shard,
    projected_remaining_seconds,
)
from rsna_knee.submission.test_surface import load_test_csv  # noqa: E402
from rsna_knee.submission.views import build_study_view  # noqa: E402

CPU = torch.device("cpu")


# --- the streamed view is the materialised one ----------------------------


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_a_streamed_view_is_the_same_tensor_not_an_equivalent_one(offset):
    """Checked with torch.equal, not a tolerance: a resize differing by a
    rounding step would still produce a plausible submission."""
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(41, 73, 121)).astype(np.float32)

    materialised, positions = prepare_series_triplets(
        raw, gap=1, center_offset=offset, crop_fraction=0.90
    )
    streamed, streamed_positions = series_triplets_from_normalised(
        _normalise_volume(raw), gap=1, center_offset=offset, crop_fraction=0.90
    )
    assert torch.equal(streamed, materialised)
    assert np.array_equal(streamed_positions, positions)


def test_prepare_series_triplets_calls_the_split_function():
    """So the two cannot drift apart when one is edited."""
    import inspect

    assert "series_triplets_from_normalised(" in inspect.getsource(prepare_series_triplets)


def test_a_study_view_keeps_its_series_in_order_and_rectangular():
    rng = np.random.default_rng(21)
    normalised = [
        _normalise_volume(rng.normal(size=shape).astype(np.float32))
        for shape in ((40, 73, 121), (36, 64, 64), (44, 101, 83))
    ]
    records = [
        {"series_uid": f"s{i}", "plane": "Sagittal", "plane_id": i + 1,
         "fluid_id": i + 2, "fat_id": i + 3}
        for i in range(3)
    ]
    volumes, position, present, meta, shapes = build_study_view(
        normalised, records, center_offset=0, gap=1, crop_fraction=0.90, device=CPU
    )

    assert len(volumes) == 3 and len(shapes) == 3
    assert len({tuple(v.shape[-2:]) for v in volumes}) > 1, "three shapes must not collapse"
    assert torch.equal(
        meta, torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=torch.long)
    )
    assert torch.equal(present, torch.ones(3))
    assert position.shape[0] == 3

    for series, volume in enumerate(volumes):
        expected, _ = series_triplets_from_normalised(
            normalised[series], gap=1, center_offset=0, crop_fraction=0.90
        )
        assert torch.equal(volume, expected)


def test_a_view_whose_records_do_not_match_is_refused():
    with pytest.raises(ValueError, match="disagree"):
        build_study_view([], [{"series_uid": "s"}], center_offset=0, gap=1,
                         crop_fraction=0.9, device=CPU)


# --- the shard survives what a hidden set contains -------------------------


class _Output:
    def __init__(self, logits):
        self.logits = logits


def _model(value=2.0):
    def call(volumes, present, meta, position):
        return _Output(torch.full((1, len(TARGETS)), float(value)))
    return call


def _shard(monkeypatch, uids, index, *, mode, failing=None):
    failing = failing or {}

    def fake_infer(uid, records, **kwargs):
        if uid in failing:
            raise failing[uid]
        return np.full(len(TARGETS), 0.9, np.float32), [(64, 48)], []

    monkeypatch.setattr(runner, "infer_one_study", fake_infer)
    return infer_shard(
        rank=0, indices=list(range(len(uids))), uids=uids, index=index,
        model=_model(), data_root="/nowhere", device=CPU, on_unreadable=mode,
    )


def test_one_unreadable_study_does_not_forfeit_the_others(monkeypatch):
    uids = [f"study-{i}" for i in range(4)]
    index = {uid: [{"series_uid": "s"}] for uid in uids}
    rows, failures = _shard(
        monkeypatch, uids, index, mode=ON_UNREADABLE_FALLBACK,
        failing={"study-2": FileNotFoundError("gone")},
    )
    assert [row[0] for row in rows] == [0, 1, 2, 3]
    assert len(failures) == 1 and failures[0]["study_uid"] == "study-2"
    assert np.allclose(rows[2][2], FALLBACK_PROBABILITY)
    assert not np.allclose(rows[0][2], FALLBACK_PROBABILITY)


def test_strict_mode_still_ends_the_run(monkeypatch):
    uids = ["a", "b"]
    index = {uid: [{"series_uid": "s"}] for uid in uids}
    with pytest.raises(FileNotFoundError):
        _shard(monkeypatch, uids, index, mode=ON_UNREADABLE_RAISE,
               failing={"b": FileNotFoundError("gone")})


def test_a_study_with_no_recognised_plane_falls_back_rather_than_raising(monkeypatch):
    rows, failures = _shard(monkeypatch, ["a", "b"], {"a": [{"series_uid": "s"}]},
                            mode=ON_UNREADABLE_FALLBACK)
    assert len(rows) == 2
    assert failures[0]["study_uid"] == "b"
    assert "recognised plane" in failures[0]["error"]


def test_an_unknown_policy_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="on_unreadable must be one of"):
        _shard(monkeypatch, ["a"], {"a": [{"series_uid": "s"}]}, mode="carry_on")


# --- the runtime projection is telemetry ----------------------------------


def test_the_projection_is_pessimistic_and_never_raises():
    assert projected_remaining_seconds([], remaining=100) == 180.0
    assert projected_remaining_seconds([1.0] * 5, remaining=10) == pytest.approx(13.5)


def test_a_shard_over_budget_still_finishes(monkeypatch):
    """The failure this avoids: one slow early study killing a run that would
    have finished."""
    uids = [f"study-{i}" for i in range(12)]
    index = {uid: [{"series_uid": "s"}] for uid in uids}

    def fake_infer(uid, records, **kwargs):
        return np.full(len(TARGETS), 0.9, np.float32), [(64, 48)], []

    monkeypatch.setattr(runner, "infer_one_study", fake_infer)
    rows, failures = infer_shard(
        rank=0, indices=list(range(12)), uids=uids, index=index, model=_model(),
        data_root="/nowhere", device=CPU, max_hours=1e-9, reserve_minutes=0.0,
    )
    assert len(rows) == 12 and failures == []


def test_the_runner_has_no_raise_on_the_budget_path():
    import inspect

    source = inspect.getsource(infer_shard)
    assert "telemetry only" in source
    assert "cannot finish inside" not in source


# --- the test table --------------------------------------------------------


def test_a_test_table_with_duplicate_studies_is_refused(tmp_path):
    import pandas as pd

    path = tmp_path / "test.csv"
    pd.DataFrame({"StudyInstanceUID": ["a", "a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than once"):
        load_test_csv(path)


def test_an_empty_test_table_is_refused(tmp_path):
    import pandas as pd

    path = tmp_path / "test.csv"
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no studies"):
        load_test_csv(path)


def test_a_table_without_study_uids_says_so(tmp_path):
    import pandas as pd

    path = tmp_path / "test.csv"
    pd.DataFrame({"something_else": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no StudyInstanceUID"):
        load_test_csv(path)


# --- how many GPUs a real submission needs --------------------------------
#
# On two T4s the hidden set takes about four hours; on one, about nine against a
# nine-hour ceiling, so it does not finish. The archive's launcher refuses below
# two for that reason -- and as a result its submission path could only ever be
# exercised on Kaggle, where each of its three failures cost a submission slot.
# The refusal is kept and made waivable, so a smoke test is possible without
# making an unfinishable run possible by accident.


def _no_gpus(monkeypatch):
    from rsna_knee.submission import launcher

    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    return launcher


def test_a_full_run_refuses_a_single_device(monkeypatch, tmp_path):
    launcher = _no_gpus(monkeypatch)
    with pytest.raises(RuntimeError, match="needs two GPUs"):
        launcher.generate_submission(
            {}, data_root=tmp_path, checkpoint=tmp_path / "c.pt",
            base_checkpoint=tmp_path / "b.pt",
        )


def test_a_limited_run_waives_it_on_its_own(monkeypatch, tmp_path):
    """A smoke test names its own size, which an accidental full run cannot."""
    launcher = _no_gpus(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        launcher.generate_submission(
            {}, data_root=tmp_path, checkpoint=tmp_path / "c.pt",
            base_checkpoint=tmp_path / "b.pt", limit=3,
        )
    assert "needs two GPUs" not in str(excinfo.value)


def test_the_waiver_can_also_be_stated_outright(monkeypatch, tmp_path):
    launcher = _no_gpus(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        launcher.generate_submission(
            {}, data_root=tmp_path, checkpoint=tmp_path / "c.pt",
            base_checkpoint=tmp_path / "b.pt", allow_single_gpu=True,
        )
    assert "needs two GPUs" not in str(excinfo.value)


def test_the_manifest_records_whether_it_was_a_full_run():
    """So a score can never be read against a submission that scored 3 studies."""
    import inspect

    from rsna_knee.submission.launcher import generate_submission

    source = inspect.getsource(generate_submission)
    assert '"full_run": len(devices) >= 2 and limit is None' in source
