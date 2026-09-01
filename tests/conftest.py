"""Fixtures shared across the suite.

The DICOM writer is here rather than in one test file because several suites
need real files on disk. Decoding is where a run actually fails, and a mocked
reader would pass while the real path was broken.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TARGET_COUNT = 12
CROP_POLICY = {"crop_fraction": 0.90, "policy": "b20_crop_focus_v1"}


def write_dicom_series(
    directory: Path, *, frames: int = 36, rows: int = 48, columns: int = 40, seed: int = 1
) -> None:
    """One small but genuine DICOM series.

    Deliberately rectangular: constant-area resizing and ragged batching only
    differ from square handling when the input is not square.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)

    for index in range(frames):
        pixels = generator.integers(0, 2048, size=(rows, columns), dtype=np.uint16)
        pixels[rows // 4 : rows // 2, columns // 4 : columns // 2] += 7000

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        frame = Dataset()
        frame.file_meta = meta
        frame.SOPClassUID = meta.MediaStorageSOPClassUID
        frame.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        frame.Modality = "MR"
        frame.Rows, frame.Columns = rows, columns
        frame.BitsAllocated = frame.BitsStored = 16
        frame.HighBit = 15
        frame.PixelRepresentation = 0
        frame.SamplesPerPixel = 1
        frame.PhotometricInterpretation = "MONOCHROME2"
        frame.InstanceNumber = index + 1
        frame.ImagePositionPatient = [0.0, 0.0, float(index) * 3.0]
        frame.PixelSpacing = [0.5, 0.5]
        frame.PixelData = pixels.tobytes()
        frame.save_as(directory / f"{index:04d}.dcm", enforce_file_format=True)


@pytest.fixture(scope="session")
def study_on_disk(tmp_path_factory) -> tuple:
    """One study with two readable series, and the records that describe it."""
    pytest.importorskip("pydicom")
    root = tmp_path_factory.mktemp("dataset") / "data"
    records: dict = {"study-a": []}
    for position, plane in enumerate(("Sagittal", "Coronal")):
        series_uid = f"series-{position}"
        rows, columns = (56, 40) if position == 0 else (48, 64)
        write_dicom_series(
            root / "train_series" / "study-a" / series_uid,
            rows=rows, columns=columns, seed=position + 1,
        )
        records["study-a"].append(
            {"series_uid": series_uid, "plane": plane,
             "plane_id": position + 1, "fluid_id": 1, "fat_id": 1}
        )
    return root, records


@pytest.fixture
def dataset_config(study_on_disk):
    from rsna_knee.data.dataset import DatasetConfig

    root, _records = study_on_disk
    return DatasetConfig(
        data_root=str(root), split="train", n_slices=16, image_size=448, triplet_gap=1
    )


@pytest.fixture
def study_dataset(study_on_disk, dataset_config):
    from rsna_knee.data.dataset import StudyDataset

    _root, records = study_on_disk
    return StudyDataset(
        ["study-a"], records, dataset_config, crop_focus_policy=CROP_POLICY,
        center_offsets=(0,),
        targets=np.zeros((1, TARGET_COUNT), np.float32),
        weights=np.ones((1, TARGET_COUNT), np.float32),
    )
