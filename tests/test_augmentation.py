"""Augmentation, checked on the real decoding path rather than in the abstract.

This package exists partly because of a bug worth remembering. The pipeline this
code came from carried nine augmentation settings in its config, a flag that set
them, and a trainer that printed `augment=True`. The dataset never read those
fields. Two independent causes -- a hard-coded `train=False` in a base class, and
a subclass that wrote its own loader straight from DICOM to triplets -- meant a
27-hour run trained on byte-identical pixels every epoch while recording
`augmentation_enabled: true` in every checkpoint.

Nothing in that run would have revealed it. An augmentation test that does not
build the real dataset from real DICOM files and compare two draws proves
nothing, so most of this file does exactly that.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rsna_knee.data.augmentation import (  # noqa: E402
    DEFAULT_SLICE_JITTER,
    AugmentationPolicy,
    AugmentedStudyDataset,
    augment_series,
    verify_augmentation_reaches_pixels,
)
from rsna_knee.data.dataset import StudyDataset  # noqa: E402

from .conftest import CROP_POLICY, TARGET_COUNT  # noqa: E402


def _augmented(study_on_disk, config, *, policy=None, slice_jitter=DEFAULT_SLICE_JITTER, seed=2026):
    _root, records = study_on_disk
    return AugmentedStudyDataset(
        ["study-a"], records, config, crop_focus_policy=CROP_POLICY, center_offsets=(0,),
        targets=np.zeros((1, TARGET_COUNT), np.float32),
        weights=np.ones((1, TARGET_COUNT), np.float32),
        policy=policy, seed=seed, slice_jitter=slice_jitter,
    )


def _series(slices: int = 8, height: int = 24, width: int = 32) -> "torch.Tensor":
    ramp = torch.linspace(0.1, 0.9, width)
    return ramp[None, :].expand(height, width)[None, None].expand(slices, 3, height, width).clone()


# --- the policy -------------------------------------------------------------


def test_the_policy_is_read_from_the_config_not_written_here():
    """Invented numbers would make this a second change, not one."""
    policy = AugmentationPolicy.from_config(
        {
            "b7_rotation_deg": 7.5, "b7_translate_frac": 0.04, "b7_scale_jitter": 0.06,
            "b7_gamma_jitter": 0.15, "b7_bias_field_strength": 0.09,
            "b7_noise_std": 0.03, "b7_slice_dropout": 0.10,
        }
    )
    assert policy.rotation_deg == 7.5
    assert policy.slice_dropout == 0.10


def test_the_shipped_config_switches_on_seven_settings():
    """The config in this repository is what a run without overrides will use."""
    import yaml
    from pathlib import Path

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "training.yaml").read_text()
    )
    assert len(AugmentationPolicy.from_config(config).active()) == 7


def test_a_negative_setting_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        AugmentationPolicy(noise_std=-0.1)


def test_dropping_every_slice_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        AugmentationPolicy(slice_dropout=1.0)


def test_the_disabled_policy_knows_it_is_disabled():
    assert AugmentationPolicy.disabled().is_disabled()
    assert not AugmentationPolicy.from_config({}).is_disabled()


# --- the augmentation itself ------------------------------------------------


def test_augmentation_changes_the_pixels():
    before = _series()
    after = augment_series(before, AugmentationPolicy.from_config({}), torch.Generator().manual_seed(7))
    assert after.shape == before.shape
    assert not torch.allclose(after, before)


def test_the_disabled_policy_leaves_the_pixels_alone():
    before = _series()
    assert torch.equal(
        augment_series(before, AugmentationPolicy.disabled(), torch.Generator().manual_seed(7)),
        before,
    )


def test_the_same_seed_reproduces_the_same_distortion():
    policy = AugmentationPolicy.from_config({})
    first = augment_series(_series(), policy, torch.Generator().manual_seed(3))
    second = augment_series(_series(), policy, torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_different_seeds_distort_differently():
    """Otherwise the model still sees one fixed dataset, merely a warped one."""
    policy = AugmentationPolicy.from_config({})
    assert not torch.allclose(
        augment_series(_series(), policy, torch.Generator().manual_seed(3)),
        augment_series(_series(), policy, torch.Generator().manual_seed(4)),
    )


def test_it_never_touches_the_global_random_state():
    """A run's loader order and weight init must not shift when augmentation does."""
    torch.manual_seed(11)
    expected = torch.randn(4)
    torch.manual_seed(11)
    augment_series(_series(), AugmentationPolicy.from_config({}), torch.Generator().manual_seed(3))
    assert torch.equal(torch.randn(4), expected)


def test_the_output_stays_in_range():
    """Pixels are percentile-normalised into [0, 1] and must stay there."""
    policy = AugmentationPolicy.from_config({})
    for seed in range(20):
        after = augment_series(_series(), policy, torch.Generator().manual_seed(seed))
        assert float(after.min()) >= 0.0 and float(after.max()) <= 1.0
        assert torch.isfinite(after).all()


def test_slice_dropout_never_empties_a_series():
    """A blank study still carrying a real label teaches something false."""
    policy = AugmentationPolicy(
        rotation_deg=0.0, translate_frac=0.0, scale_jitter=0.0, gamma_jitter=0.0,
        bias_field_strength=0.0, noise_std=0.0, slice_dropout=0.999,
    )
    for seed in range(15):
        after = augment_series(_series(), policy, torch.Generator().manual_seed(seed))
        assert [i for i in range(after.shape[0]) if float(after[i].abs().sum()) > 0]


def test_there_is_no_left_right_flip():
    """Mirroring a knee swaps medial and lateral, which are separate findings.

    Read from the source rather than the behaviour: a flip added behind a policy
    flag that happens to be off would pass any behavioural check while still
    being one edit away from live.
    """
    import inspect

    source = inspect.getsource(augment_series)
    for primitive in ("torch.flip", "fliplr", "hflip", "[::-1]"):
        assert primitive not in source, f"{primitive} mirrors the image"


# --- the dataset ------------------------------------------------------------


def test_the_dataset_augments_the_decoded_pixels(study_on_disk, dataset_config):
    """The whole point, on the real path: two draws of one study must differ."""
    dataset = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}))
    dataset.set_epoch(1)
    first = dataset[0]
    dataset.set_epoch(2)
    second = dataset[0]

    assert len(first["volumes"]) == 2
    for position in range(2):
        assert not torch.equal(first["volumes"][position], second["volumes"][position])


def test_it_is_reproducible_at_a_fixed_epoch(study_on_disk, dataset_config):
    left = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}))
    right = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}))
    left.set_epoch(4)
    right.set_epoch(4)
    assert all(torch.equal(a, b) for a, b in zip(left[0]["volumes"], right[0]["volumes"]))


def test_a_different_run_seed_augments_differently(study_on_disk, dataset_config):
    left = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}), seed=2026)
    right = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}), seed=7)
    left.set_epoch(1)
    right.set_epoch(1)
    assert not torch.equal(left[0]["volumes"][0], right[0]["volumes"][0])


def test_without_a_policy_it_matches_the_plain_dataset(study_on_disk, dataset_config, study_dataset):
    """The control arm must be the unaugmented behaviour, not merely close to it."""
    off = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.disabled())
    off.set_epoch(3)
    assert all(torch.equal(a, b) for a, b in zip(off[0]["volumes"], study_dataset[0]["volumes"]))


def test_the_geometry_is_unchanged(study_on_disk, dataset_config, study_dataset):
    """The crop, the aspect ratio and the pixel area are the frozen contract.

    Augmentation runs after all of it and must not resize anything, or a run
    would differ in two things rather than one.
    """
    augmented = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}))
    augmented.set_epoch(1)
    plain = study_dataset[0]
    assert augmented[0]["geometry"] == plain["geometry"]
    for a, b in zip(augmented[0]["volumes"], plain["volumes"]):
        assert a.shape == b.shape


def test_slice_jitter_is_off_by_default(study_on_disk, dataset_config):
    """It changes which slices are chosen, which is a second change."""
    assert _augmented(
        study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({})
    ).slice_jitter == 0


def test_slice_jitter_moves_the_centres_when_asked(study_on_disk, dataset_config):
    plain = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.disabled(), slice_jitter=0)
    jittered = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.disabled(), slice_jitter=4)
    jittered.set_epoch(1)
    assert not torch.equal(jittered[0]["slice_position"], plain[0]["slice_position"])


def test_slice_jitter_restores_the_offsets_it_borrowed(study_on_disk, dataset_config):
    """It mutates center_offsets in place; leaving it shifted would drift."""
    dataset = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.disabled(), slice_jitter=3)
    before = dataset.center_offsets
    dataset.set_epoch(1)
    dataset[0]
    assert dataset.center_offsets == before


# --- the check that gates a run --------------------------------------------


def test_the_verification_passes_on_a_real_augmented_dataset(study_on_disk, dataset_config):
    dataset = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}))
    report = verify_augmentation_reaches_pixels(dataset)

    assert report["series_compared"] == 2
    assert report["series_that_changed"] == 2
    assert report["max_absolute_difference"] > 0


def test_the_verification_leaves_the_epoch_where_it_found_it(study_on_disk, dataset_config):
    dataset = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}))
    verify_augmentation_reaches_pixels(dataset)
    assert dataset.epoch == 0


def test_the_verification_catches_augmentation_that_does_nothing(
    study_on_disk, dataset_config, monkeypatch
):
    """The original failure, simulated: a live policy with no effect.

    A run must not be able to start in that state, because nothing later would
    reveal it.
    """
    dataset = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.from_config({}))
    monkeypatch.setattr(
        "rsna_knee.data.augmentation.augment_series",
        lambda series, policy, generator: series,
    )
    with pytest.raises(RuntimeError, match="did not reach the pixels"):
        verify_augmentation_reaches_pixels(dataset)


def test_the_verification_refuses_a_disabled_policy(study_on_disk, dataset_config):
    """Calling it on the control arm is a mistake, not a pass."""
    dataset = _augmented(study_on_disk, dataset_config, policy=AugmentationPolicy.disabled())
    with pytest.raises(RuntimeError, match="augmentation off"):
        verify_augmentation_reaches_pixels(dataset)


def test_the_plain_dataset_still_ignores_config_augmentation_fields(study_on_disk):
    """The original bug, pinned as a fact about the inherited dataset.

    `StudyDataset` writes its own loader and does not read the augmentation
    fields on its config. That is why `AugmentedStudyDataset` exists. If this
    ever starts failing, the base dataset has begun augmenting on its own and
    augmentation would be applied twice.
    """
    from rsna_knee.data.dataset import DatasetConfig

    root, records = study_on_disk

    def build(live: bool):
        config = DatasetConfig(
            data_root=str(root), split="train", n_slices=16, image_size=448, triplet_gap=1,
            noise_std=0.02 if live else 0.0, slice_dropout=0.08 if live else 0.0,
            rotation_deg=5.0 if live else 0.0, translate_frac=0.03 if live else 0.0,
            scale_jitter=0.05 if live else 0.0, gamma_jitter=0.12 if live else 0.0,
            bias_field_strength=0.08 if live else 0.0,
        )
        return StudyDataset(
            ["study-a"], records, config, crop_focus_policy=CROP_POLICY, center_offsets=(0,),
            targets=np.zeros((1, TARGET_COUNT), np.float32),
            weights=np.ones((1, TARGET_COUNT), np.float32),
        )

    on, off = build(True), build(False)
    assert on.config.rotation_deg == 5.0, "the config really does carry the setting"
    torch.manual_seed(0)
    first = on[0]["volumes"][0]
    torch.manual_seed(1)
    second = on[0]["volumes"][0]
    torch.manual_seed(2)
    plain = off[0]["volumes"][0]
    assert torch.equal(first, second) and torch.equal(first, plain)


# --- the control arm, which had never been run ----------------------------
#
# `--no-augment` is documented as reproducing the run this experiment is
# measured against. It crashed on every attempt: preflight called
# verify_augmentation_reaches_pixels unconditionally, and that function
# correctly refuses to verify augmentation that is switched off. The control arm
# of the experiment was unreachable.


def _plain_dataset(study_on_disk, dataset_config, policy):
    from rsna_knee.data.augmentation import AugmentedStudyDataset

    _root, records = study_on_disk
    return AugmentedStudyDataset(
        ["study-a"], records, dataset_config, crop_focus_policy=CROP_POLICY,
        center_offsets=(0,),
        targets=np.zeros((1, TARGET_COUNT), np.float32),
        weights=np.ones((1, TARGET_COUNT), np.float32),
        policy=policy, seed=2026,
    )


def test_the_control_arm_verifies_that_nothing_moved(study_on_disk, dataset_config):
    """The mirror check. Skipping it would leave the arm that most needs a
    guarantee as the only one without one."""
    from rsna_knee.data.augmentation import verify_augmentation_is_off

    dataset = _plain_dataset(study_on_disk, dataset_config, AugmentationPolicy.disabled())
    report = verify_augmentation_is_off(dataset)

    assert report["series_that_changed"] == 0
    assert report["max_absolute_difference"] == 0.0
    assert report["augmentation_enabled"] is False
    assert report["series_compared"] == 2


def test_the_control_check_refuses_an_augmenting_dataset(study_on_disk, dataset_config):
    """Each check answers one arm's question. Neither answers the other's."""
    from rsna_knee.data.augmentation import verify_augmentation_is_off

    dataset = _plain_dataset(
        study_on_disk, dataset_config, AugmentationPolicy.from_config({})
    )
    with pytest.raises(RuntimeError, match="was called with augmentation on"):
        verify_augmentation_is_off(dataset)


def test_a_control_arm_that_secretly_augments_is_refused(study_on_disk, dataset_config):
    """The failure this exists to catch: a control that is not one.

    The policy says disabled while the pixels still move between draws -- which
    is the same shape of defect as B52's, pointing the other way.
    """
    from rsna_knee.data.augmentation import verify_augmentation_is_off

    dataset = _plain_dataset(study_on_disk, dataset_config, AugmentationPolicy.disabled())
    inner = type(dataset).__getitem__
    draws = {"n": 0}

    class Wobbling(type(dataset)):
        def __getitem__(self, index):
            item = inner(self, index)
            draws["n"] += 1
            if draws["n"] % 2 == 0:
                item["volumes"] = [volume + 0.5 for volume in item["volumes"]]
            return item

    dataset.__class__ = Wobbling
    with pytest.raises(RuntimeError, match="not the control arm it claims"):
        verify_augmentation_is_off(dataset)


def test_preflight_checks_whichever_arm_is_running():
    """Dispatching on the policy is what makes --no-augment reachable at all."""
    import inspect

    from rsna_knee.training.loop import preflight

    source = inspect.getsource(preflight)
    assert "train_dataset.policy.is_disabled()" in source
    assert "verify_augmentation_is_off" in source
    assert "verify_augmentation_reaches_pixels" in source
