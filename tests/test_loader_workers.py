"""Raising num_workers must not change what the model sees.

`num_workers: 0` decodes DICOM and augments on the main thread, between GPU
steps, so a fast card spends its time waiting for the CPU. Raising it is the
obvious fix -- but a baseline measured at 0 is only comparable with a run at 6
if the worker count changes throughput and nothing else.

Two things have to hold. The shuffle order must come from the explicitly seeded
generator rather than the global RNG, or every worker count would visit studies
in a different order. And the augmentation must draw from a generator keyed by
run seed, epoch and study index rather than the global state, or each worker
would inherit a copy of the same stream and hand out the same "random" numbers.

Both are design decisions in this package. This checks they actually hold.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pydicom")

from torch.utils.data import DataLoader  # noqa: E402

from rsna_knee.data.augmentation import AugmentationPolicy, AugmentedStudyDataset  # noqa: E402
from rsna_knee.data.dataset import DatasetConfig, collate_studies  # noqa: E402
from rsna_knee.runtime import resolve_runtime  # noqa: E402

from .conftest import CROP_POLICY, TARGET_COUNT, write_dicom_series  # noqa: E402


@pytest.fixture(scope="module")
def several_studies():
    """Enough studies that a shuffle has something to get wrong."""
    root = Path(tempfile.mkdtemp()) / "data"
    records: dict = {}
    for index in range(4):
        uid = f"study-{index}"
        records[uid] = []
        write_dicom_series(
            root / "train_series" / uid / "series-0",
            frames=36, rows=40, columns=32, seed=index + 1,
        )
        records[uid].append({
            "series_uid": "series-0", "plane": "Sagittal",
            "plane_id": 1, "fluid_id": 1, "fat_id": 1,
        })
    return root, records


def _pass(root, records, workers: int):
    uids = sorted(records)
    config = DatasetConfig(
        data_root=str(root), split="train", n_slices=16, image_size=448, triplet_gap=1
    )
    dataset = AugmentedStudyDataset(
        uids, records, config, crop_focus_policy=CROP_POLICY, center_offsets=(0,),
        targets=np.zeros((len(uids), TARGET_COUNT), np.float32),
        weights=np.ones((len(uids), TARGET_COUNT), np.float32),
        policy=AugmentationPolicy.from_config({}), seed=2026,
    )
    dataset.set_epoch(3)
    runtime = resolve_runtime({"num_workers": workers, "device": "cpu"})
    loader = DataLoader(
        dataset, batch_size=2, shuffle=True, drop_last=False,
        collate_fn=collate_studies, **runtime.loader_kwargs(seed=2026 + 29),
    )
    order, pixels = [], []
    for batch in loader:
        for item in batch:
            order.append(item["study_uid"])
            pixels.append(item["volumes"][0].clone())
    return order, pixels


def test_worker_count_changes_throughput_and_nothing_else(several_studies):
    """The property a baseline measured at workers=0 depends on."""
    root, records = several_studies
    single_order, single_pixels = _pass(root, records, 0)
    many_order, many_pixels = _pass(root, records, 2)

    assert single_order == many_order, (
        "the shuffle visited studies in a different order, so the loader's "
        "generator is not seeded independently of the worker count"
    )
    assert len(single_pixels) == len(many_pixels) == len(records)
    for position, (single, many) in enumerate(zip(single_pixels, many_pixels)):
        assert torch.equal(single, many), (
            f"study {many_order[position]} was augmented differently with workers; "
            "the augmentation is reading the global random state, so each worker "
            "inherits a copy of the same stream"
        )


def test_the_loader_generator_is_seeded_explicitly():
    """The structural half: a generator must reach the DataLoader at all.

    Without it the sampler falls back to the global RNG, and the shuffle order
    then depends on however much randomness the process consumed beforehand.
    """
    runtime = resolve_runtime({"num_workers": 0, "device": "cpu"})
    kwargs = runtime.loader_kwargs(seed=99)
    assert "generator" in kwargs
    assert kwargs["generator"].initial_seed() == 99


def test_workers_get_an_init_function_that_seeds_them():
    """Python and NumPy inside a worker start from the global seed otherwise."""
    runtime = resolve_runtime({"num_workers": 4, "device": "cpu"})
    kwargs = runtime.loader_kwargs(seed=1)
    assert callable(kwargs.get("worker_init_fn"))
    assert kwargs["num_workers"] == 4
