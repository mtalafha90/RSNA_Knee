"""The second reading, and the ceiling that says whether it can resolve anything.

The report-derived validation surface is what selects the epoch and what the
0.834998 baseline was measured on. It is also the surface that measured
`+0.011219` for a change the hidden test returned `-0.001` for. The 58
expert-labelled studies are the only expert-truth proxy this project has, and
its audit said `-0.002432` -- inconclusive at that size, but pointing the right
way where the 548-study report surface did not.

These tests cover the arithmetic that makes such an audit readable: the AUC
ceiling two prediction sets impose on each other, and the reading rule applied
to a delta. Both are written to be checkable without a GPU or the competition
data, because the parts that need those are the parts that already exist.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")

from rsna_knee.constants import TARGETS  # noqa: E402
from rsna_knee.expert_audit import (  # noqa: E402
    EXPERT_RESOLUTION,
    SUPPORT_DELTA,
    VETO_DELTA,
    discordant_pair_fraction,
    expert_scores,
    read_predictions,
    read_the_delta,
    write_predictions,
)

STUDIES = 58


def _truth(seed: int = 0) -> np.ndarray:
    """Hard expert labels with both classes present in every target."""
    rng = np.random.default_rng(seed)
    truth = (rng.random((STUDIES, len(TARGETS))) > 0.6).astype(np.float64)
    truth[0, :] = 1.0
    truth[1, :] = 0.0
    return truth


# --- the ceiling ------------------------------------------------------------


def test_identical_predictions_cannot_differ_at_all():
    """The property the ceiling exists to express."""
    truth = _truth()
    rng = np.random.default_rng(1)
    predictions = rng.random(truth.shape)
    result = discordant_pair_fraction(truth, predictions, predictions)
    assert result["ceiling"] == 0.0


def test_a_reversed_ordering_has_a_ceiling_of_one():
    truth = _truth()
    rng = np.random.default_rng(2)
    predictions = rng.random(truth.shape)
    result = discordant_pair_fraction(truth, predictions, -predictions)
    assert result["ceiling"] == pytest.approx(1.0)


def test_the_ceiling_really_does_bound_the_auc_difference():
    """Not asserted -- checked against measured AUCs on many random pairs.

    This is the claim the whole module rests on: a delta larger than its own
    ceiling is arithmetically impossible, so it means a bug rather than a
    result.
    """
    truth = _truth(3)
    rng = np.random.default_rng(4)
    for _ in range(30):
        left = rng.random(truth.shape)
        # Correlated rather than independent, which is the realistic case: two
        # checkpoints from the same recipe order most pairs identically.
        right = left + rng.normal(scale=rng.uniform(0.01, 2.0), size=truth.shape)
        ceiling = discordant_pair_fraction(truth, left, right)["ceiling"]
        delta = abs(
            expert_scores(truth, left)["macro_auc"]
            - expert_scores(truth, right)["macro_auc"]
        )
        assert delta <= ceiling + 1e-9, f"delta {delta} exceeded ceiling {ceiling}"


def test_two_nearly_identical_checkpoints_have_a_tiny_ceiling():
    """B48 and B49 were judged against +0.010 with ceilings of 0.0015 and 0.0024.

    Neither could have passed whatever its mechanism did. Reporting a delta
    without its ceiling is how that goes unnoticed.
    """
    truth = _truth(5)
    rng = np.random.default_rng(6)
    left = rng.random(truth.shape)
    right = left + rng.normal(scale=1e-6, size=truth.shape)
    assert discordant_pair_fraction(truth, left, right)["ceiling"] < 0.01


def test_a_target_with_one_class_reports_nan_rather_than_a_number():
    truth = _truth(7)
    truth[:, 0] = 1.0
    rng = np.random.default_rng(8)
    result = discordant_pair_fraction(truth, rng.random(truth.shape), rng.random(truth.shape))
    assert np.isnan(result["per_target_ceiling"][TARGETS[0]])
    assert np.isfinite(result["ceiling"]), "the macro must survive one dead target"


def test_mismatched_shapes_are_refused():
    truth = _truth()
    with pytest.raises(ValueError, match="same shape"):
        discordant_pair_fraction(truth, truth, truth[:10])


# --- the reading rule, fixed before any B53 number existed -----------------


def test_a_clear_expert_loss_is_a_veto():
    reading = read_the_delta(-0.05, ceiling=0.2)
    assert reading.startswith("VETO")
    assert "B15" in reading


def test_a_clear_expert_gain_is_supported():
    assert read_the_delta(0.02, ceiling=0.2).startswith("SUPPORTED")


def test_a_small_delta_is_silence_not_failure():
    reading = read_the_delta(-0.002432, ceiling=0.03)
    assert reading.startswith("INCONCLUSIVE")
    assert "not a" in reading and "failure" in reading


def test_b50s_actual_expert_delta_reads_as_inconclusive():
    """The number that preceded B51's -0.001 hidden result."""
    assert read_the_delta(-0.002432, ceiling=0.0307).startswith("INCONCLUSIVE")


def test_a_delta_larger_than_its_ceiling_is_reported_as_a_bug():
    reading = read_the_delta(0.05, ceiling=0.01)
    assert reading.startswith("IMPOSSIBLE")
    assert "bug" in reading


def test_the_thresholds_are_consistent_with_the_stated_resolution():
    assert VETO_DELTA < 0 < SUPPORT_DELTA
    assert SUPPORT_DELTA <= EXPERT_RESOLUTION, (
        "a support threshold inside the noise band would confirm nothing"
    )


# --- prediction files -------------------------------------------------------


def test_predictions_round_trip(tmp_path):
    uids = [f"study-{i}" for i in range(STUDIES)]
    rng = np.random.default_rng(9)
    probabilities = rng.random((STUDIES, len(TARGETS)))
    path = write_predictions(tmp_path / "b53_expert58_predictions.csv", uids, probabilities)
    assert np.allclose(read_predictions(path, uids), probabilities)


def test_a_prediction_file_from_another_surface_is_refused(tmp_path):
    """Comparing against misaligned rows would silently produce a wrong delta."""
    uids = [f"study-{i}" for i in range(STUDIES)]
    rng = np.random.default_rng(10)
    path = write_predictions(tmp_path / "p.csv", uids, rng.random((STUDIES, len(TARGETS))))
    with pytest.raises(ValueError, match="not aligned"):
        read_predictions(path, list(reversed(uids)))


def test_a_prediction_file_with_the_wrong_columns_is_refused(tmp_path):
    import pandas as pd  # noqa: PLC0415

    path = tmp_path / "wrong.csv"
    pd.DataFrame({"StudyInstanceUID": ["a"], "not_a_target": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="prediction columns changed"):
        read_predictions(path, ["a"])


# --- scoring ----------------------------------------------------------------


def test_a_perfect_ranking_scores_one_and_a_reversed_one_scores_zero():
    truth = _truth(11)
    assert expert_scores(truth, truth)["macro_auc"] == pytest.approx(1.0)
    assert expert_scores(truth, 1.0 - truth)["macro_auc"] == pytest.approx(0.0)


def test_every_expert_cell_is_scored():
    """Unlike the report surface, nothing here is unsupervised."""
    truth = _truth(12)
    rng = np.random.default_rng(13)
    scores = expert_scores(truth, rng.random(truth.shape))
    assert scores["targets_defined"] == len(TARGETS)
