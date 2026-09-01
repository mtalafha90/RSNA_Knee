"""This package must compute exactly what the code it was extracted from does.

The package was migrated mechanically from a research archive: definitions were
copied verbatim, regrouped by purpose, and their names changed from experiment
numbers to descriptions. A rename that quietly changed a number would be worse
than no rename at all, because every measurement in the archive would stop being
comparable without anyone noticing.

So the migration is not trusted -- it is checked. When the original package is
importable, these tests run both side by side on identical inputs and compare
bit for bit. When it is not (the normal case, on any machine that only has this
repository), they skip, and the recorded result below stands as the evidence.

    25/25 checks passed, 2026-09-01, against cnn_cpc @ 10bec85

Set `RSNA_KNEE_ORIGINAL` to the original `developments/src` directory to re-run
the comparison for yourself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ORIGINAL = os.environ.get("RSNA_KNEE_ORIGINAL", "")

pytestmark = pytest.mark.skipif(
    not ORIGINAL or not (Path(ORIGINAL) / "rsna_knee" / "b53_augmented_training.py").is_file(),
    reason="set RSNA_KNEE_ORIGINAL to the original developments/src to compare",
)

CROP_POLICY = {"crop_fraction": 0.90, "policy": "b20_crop_focus_v1"}


@pytest.fixture(scope="module")
def original():
    """Import the original package under the same name, then restore ours.

    Both packages call themselves `rsna_knee`, so they cannot be live at once.
    Everything the comparison needs is pulled out here and the import state is
    put back, which keeps the swap contained to this fixture.
    """
    ours = {name: module for name, module in sys.modules.items()
            if name == "rsna_knee" or name.startswith("rsna_knee.")}
    for name in ours:
        del sys.modules[name]
    sys.path.insert(0, ORIGINAL)
    try:
        from rsna_knee import (  # noqa: PLC0415
            b12_variable_series, b34_training_only_context_scaffold,
            b35_target_spatial_residual, b42_constant_area_aspect_sparse_mil,
            b50_adapted_hierarchy_mil, b52_competition_training,
            b53_augmented_training, dataset, dicom, evaluation,
        )
        captured = {
            "dataset_cls": b42_constant_area_aspect_sparse_mil.B42ConstantAreaAspectDataset,
            "aug_dataset_cls": b53_augmented_training.B53AugmentedDataset,
            "policy_cls": b53_augmented_training.AugmentationPolicy,
            "augment": b53_augmented_training.augment_b42_series,
            "triplets": b42_constant_area_aspect_sparse_mil.preprocess_dense_triplets_b42,
            "normalise": dicom._normalise_volume,
            "centres": b35_target_spatial_residual.b35_centers,
            "base_cls": b34_training_only_context_scaffold.TrainingOnlyContextScaffoldKneeMILNet,
            "model_cls": b50_adapted_hierarchy_mil.B50AdaptedHierarchySparseMILResidual,
            "macro_auc": b52_competition_training.macro_auc,
            "masked": b52_competition_training.masked_binary_targets,
            "select": b52_competition_training.select_train_and_validation,
            "groups": b52_competition_training.b52_parameter_groups,
            "auc": evaluation.fast_auc,
            "series_index": b12_variable_series.build_variable_series_index,
            "config_cls": dataset.DatasetConfig,
        }
    finally:
        sys.path.remove(ORIGINAL)
        for name in [n for n in sys.modules if n == "rsna_knee" or n.startswith("rsna_knee.")]:
            del sys.modules[name]
        sys.modules.update(ours)
    return captured


@pytest.fixture(scope="module")
def ours():
    from rsna_knee import evaluation
    from rsna_knee.data import augmentation, dataset, series_policy, slice_selection
    from rsna_knee.imaging import dicom_io, triplets
    from rsna_knee.model import study_hierarchy, study_model
    from rsna_knee.training import loop

    return {
        "dataset_cls": dataset.StudyDataset,
        "aug_dataset_cls": augmentation.AugmentedStudyDataset,
        "policy_cls": augmentation.AugmentationPolicy,
        "augment": augmentation.augment_series,
        "triplets": triplets.prepare_series_triplets,
        "normalise": dicom_io._normalise_volume,
        "centres": slice_selection.slice_centres,
        "base_cls": study_hierarchy.StudyHierarchyNet,
        "model_cls": study_model.KneeStudyModel,
        "macro_auc": evaluation.macro_auc,
        "masked": evaluation.masked_binary_targets,
        "select": loop.select_train_and_validation,
        "groups": loop.parameter_groups,
        "auc": evaluation.fast_auc,
        "series_index": series_policy.build_series_index_all_planes,
        "config_cls": dataset.DatasetConfig,
    }


def _dataset(pack, study_on_disk, *, augmented=False, policy=None):
    root, records = study_on_disk
    config = pack["config_cls"](
        data_root=str(root), split="train", n_slices=16, image_size=448, triplet_gap=1
    )
    cls = pack["aug_dataset_cls"] if augmented else pack["dataset_cls"]
    extra = {"policy": policy, "seed": 2026} if augmented else {}
    return cls(
        ["study-a"], records, config, crop_focus_policy=CROP_POLICY, center_offsets=(0,),
        targets=np.zeros((1, 12), np.float32), weights=np.ones((1, 12), np.float32), **extra,
    )


# --- the pixels -------------------------------------------------------------


def test_intensity_normalisation_is_identical(original, ours):
    volume = np.random.default_rng(4).random((40, 48, 56)).astype(np.float32) * 900
    assert np.array_equal(original["normalise"](volume), ours["normalise"](volume))


@pytest.mark.parametrize("gap", [1, 2])
def test_slice_centres_are_identical(original, ours, gap):
    a = original["centres"](40, gap=gap, center_offset=0)
    b = ours["centres"](40, gap=gap, center_offset=0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_crop_resize_and_triplets_are_identical(original, ours):
    volume = np.random.default_rng(4).random((40, 48, 56)).astype(np.float32) * 900
    a = original["triplets"](volume, gap=1, center_offset=0, crop_fraction=0.90)
    b = ours["triplets"](volume, gap=1, center_offset=0, crop_fraction=0.90)
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_a_decoded_study_is_identical(original, ours, study_on_disk):
    a = _dataset(original, study_on_disk)[0]
    b = _dataset(ours, study_on_disk)[0]
    assert len(a["volumes"]) == len(b["volumes"]) == 2
    assert all(torch.equal(x, y) for x, y in zip(a["volumes"], b["volumes"]))
    assert torch.equal(a["slice_position"], b["slice_position"])
    assert a["geometry"] == b["geometry"]


# --- the augmentation -------------------------------------------------------


def test_the_policy_values_are_identical(original, ours):
    assert (
        original["policy_cls"].from_config({}).to_dict()
        == ours["policy_cls"].from_config({}).to_dict()
    )


@pytest.mark.parametrize("seed", [1, 7, 33])
def test_augmented_pixels_are_identical(original, ours, seed):
    series = torch.rand(8, 3, 24, 32, generator=torch.Generator().manual_seed(9))
    a = original["augment"](series, original["policy_cls"].from_config({}),
                            torch.Generator().manual_seed(seed))
    b = ours["augment"](series, ours["policy_cls"].from_config({}),
                        torch.Generator().manual_seed(seed))
    assert torch.equal(a, b)


def test_an_augmented_study_is_identical(original, ours, study_on_disk):
    a = _dataset(original, study_on_disk, augmented=True,
                 policy=original["policy_cls"].from_config({}))
    b = _dataset(ours, study_on_disk, augmented=True, policy=ours["policy_cls"].from_config({}))
    a.set_epoch(3)
    b.set_epoch(3)
    assert all(torch.equal(x, y) for x, y in zip(a[0]["volumes"], b[0]["volumes"]))


# --- the model --------------------------------------------------------------


def _models(original, ours):
    base_kwargs = dict(pretrained_weights=False, normalize_input=True, dropout=0.0)
    torch.manual_seed(0)
    base_a = original["base_cls"](16, **base_kwargs)
    torch.manual_seed(0)
    base_b = ours["base_cls"](16, **base_kwargs)

    kwargs = dict(grid_size=6, top_k=8, temperature=1.0, encoder_trainable_stages=1,
                  encoder_chunk_size=1, adapt_hierarchy=True)
    torch.manual_seed(1)
    model_a = original["model_cls"](base_a, **kwargs).eval()
    torch.manual_seed(1)
    model_b = ours["model_cls"](base_b, **kwargs).eval()
    model_b.load_state_dict(model_a.state_dict())
    return base_a, base_b, model_a, model_b


def test_the_state_dict_keys_are_identical(original, ours):
    """Checkpoint keys are attribute paths.

    Class names could be renamed freely; `self.encoder` could not, because the
    Phase-9 base checkpoint is loaded by `load_state_dict` and its keys are
    exactly these strings.
    """
    base_a, base_b, model_a, model_b = _models(original, ours)
    assert list(base_a.state_dict()) == list(base_b.state_dict())
    assert list(model_a.state_dict()) == list(model_b.state_dict())


def test_the_logits_are_identical(original, ours, study_on_disk):
    _base_a, _base_b, model_a, model_b = _models(original, ours)
    study = _dataset(ours, study_on_disk)[0]
    args = (list(study["volumes"]), study["present"], study["series_meta"],
            study["slice_position"])
    with torch.no_grad():
        out_a, out_b = model_a(*args), model_b(*args)
    assert torch.equal(out_a.logits, out_b.logits)
    assert torch.equal(out_a.local_logits, out_b.local_logits)


def test_the_optimiser_groups_are_identical(original, ours):
    _base_a, _base_b, model_a, model_b = _models(original, ours)
    rates = dict(head_lr=1e-4, encoder_lr_scale=0.10, hierarchy_lr_scale=0.05)
    summary = lambda groups: [  # noqa: E731
        (g["name"], g["lr"], sum(p.numel() for p in g["params"])) for g in groups
    ]
    assert summary(original["groups"](model_a, **rates)) == summary(ours["groups"](model_b, **rates))


# --- the maths --------------------------------------------------------------


def test_the_scoring_is_identical(original, ours):
    rng = np.random.default_rng(5)
    target = rng.choice([0.03, 0.97], size=(40, 12))
    weight = rng.choice([0.0, 0.9], size=(40, 12))
    prediction = rng.random((40, 12))

    assert np.array_equal(
        np.nan_to_num(original["masked"](target, weight), nan=-1),
        np.nan_to_num(ours["masked"](target, weight), nan=-1),
    )
    a = original["macro_auc"](target, weight, prediction)
    b = ours["macro_auc"](target, weight, prediction)
    assert a["macro_auc"] == b["macro_auc"]
    assert a["per_target_auc"] == b["per_target_auc"]

    truth = rng.integers(0, 2, 60).astype(float)
    score = rng.integers(0, 6, 60).astype(float)
    assert original["auc"](truth, score) == ours["auc"](truth, score)


def test_the_split_selection_is_identical(original, ours):
    import pandas as pd

    uids = [f"study-{index}" for index in range(30)]
    rows = pd.DataFrame([
        {"StudyInstanceUID": uid,
         "split": "train" if index % 3 else "validation_unseen_scanners"}
        for index, uid in enumerate(uids)
    ])
    a, b = original["select"](uids, rows), ours["select"](uids, rows)
    assert a[2] == b[2] and a[3] == b[3]
