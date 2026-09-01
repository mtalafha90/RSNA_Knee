"""The datasets the loader yields: one study, its series, its labels."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
import numpy as np
import torch

from ..constants import DUAL_STREAMS
from .slice_selection import DENSE_SLICES
from ..imaging.crop_policy import CROP_FRACTION, validate_crop_focus_policy
from ..imaging.dicom_io import find_series_dir, preprocess_triplets, read_dicom_series
from ..imaging.physical_scale import resample_volume_inplane, validate_physical_scale_policy
from ..imaging.triplets import REFERENCE_SIDE, SQUARE_IMAGE_SIZE, prepare_series_triplets, prepare_square_triplets


def collate_studies(items):
    """Keep study items ragged instead of padding heterogeneous rectangles."""
    return list(items)


@dataclass
class DatasetConfig:
    data_root: str
    split: str = "train"
    n_slices: int = 16
    image_size: int = 224
    noise_std: float = 0.02
    slice_dropout: float = 0.08
    triplet_gap: int = 1
    strict_dicom: bool = False
    train_gap_choices: tuple[int, ...] = (1, 2)
    center_jitter: int = 2
    center_offset: int = 0
    tta_center_offsets: tuple[int, ...] = ()
    rotation_deg: float = 5.0
    translate_frac: float = 0.03
    scale_jitter: float = 0.05
    gamma_jitter: float = 0.12
    bias_field_strength: float = 0.08
    series_cache_mb: int = 256
    physical_scale_policy: dict | None = None

    def __post_init__(self) -> None:
        if self.n_slices < 1 or self.image_size < 1 or self.triplet_gap < 1:
            raise ValueError("n_slices/image_size/triplet_gap must be positive")
        if any(g < 1 for g in self.train_gap_choices):
            raise ValueError("train_gap_choices must be >=1")
        if self.center_jitter < 0 or self.noise_std < 0 or self.series_cache_mb < 0:
            raise ValueError("jitter/noise/cache size must be non-negative")
        if not 0 <= self.slice_dropout < 1:
            raise ValueError("slice_dropout must be in [0,1)")
        if self.physical_scale_policy is not None:
            validate_physical_scale_policy(self.physical_scale_policy)


class _VolumeLRU:
    """Bounded per-process cache of decoded/preprocessed DICOM volumes."""

    def __init__(self, max_mb: int) -> None:
        self.max_bytes = int(max_mb) * 1024 * 1024
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()
        self.bytes = 0

    def get(self, key: str) -> np.ndarray | None:
        value = self.items.pop(key, None)
        if value is not None:
            self.items[key] = value
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_bytes <= 0 or value.nbytes > self.max_bytes:
            return
        old = self.items.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        self.items[key] = value
        self.bytes += value.nbytes
        while self.items and self.bytes > self.max_bytes:
            _, evicted = self.items.popitem(last=False)
            self.bytes -= evicted.nbytes


class SliceStreamDataset(Dataset):
    def __init__(
        self,
        study_uids,
        series_index,
        config: DatasetConfig,
        targets=None,
        weights=None,
        train: bool = False,
    ):
        self.study_uids = [str(x) for x in study_uids]
        self.series_index = series_index
        self.config = config
        self.targets = targets
        self.weights = weights
        self.train = bool(train)
        self.stream_names = list(DUAL_STREAMS)
        self._cache = _VolumeLRU(config.series_cache_mb)
        n = len(self.study_uids)
        if targets is not None and len(targets) != n:
            raise ValueError("targets length mismatch")
        if weights is not None and len(weights) != n:
            raise ValueError("weights length mismatch")
        if self.train and self.config.tta_center_offsets:
            raise ValueError("TTA offsets are inference-only")

    @property
    def in_channels(self):
        return 3

    @property
    def n_views(self) -> int:
        return len(self.config.tta_center_offsets) if self.config.tta_center_offsets else 1

    def __len__(self):
        return len(self.study_uids)

    def _zero(self):
        base = torch.zeros(
            self.config.n_slices,
            3,
            self.config.image_size,
            self.config.image_size,
            dtype=torch.float32,
        )
        if self.config.tta_center_offsets:
            return base.unsqueeze(0).repeat(self.n_views, 1, 1, 1, 1)
        return base

    def _augment_mri(self, volume: torch.Tensor) -> torch.Tensor:
        """Mild acquisition-like augmentation shared across one series."""
        angle = float(
            torch.empty(1).uniform_(-self.config.rotation_deg, self.config.rotation_deg)
        )
        max_shift = int(round(self.config.translate_frac * self.config.image_size))
        translate = [
            int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            if max_shift
            else 0,
            int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            if max_shift
            else 0,
        ]
        scale = float(
            torch.empty(1).uniform_(
                1 - self.config.scale_jitter, 1 + self.config.scale_jitter
            )
        )
        volume = TVF.affine(
            volume,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        if self.config.gamma_jitter > 0:
            gamma = float(
                torch.empty(1).uniform_(
                    1 - self.config.gamma_jitter, 1 + self.config.gamma_jitter
                )
            )
            volume = volume.clamp(0, 1).pow(gamma)
        if self.config.bias_field_strength > 0:
            h, w = volume.shape[-2:]
            yy = torch.linspace(-1, 1, h, device=volume.device).view(1, 1, h, 1)
            xx = torch.linspace(-1, 1, w, device=volume.device).view(1, 1, 1, w)
            ax = float(
                torch.empty(1).uniform_(
                    -self.config.bias_field_strength, self.config.bias_field_strength
                )
            )
            ay = float(
                torch.empty(1).uniform_(
                    -self.config.bias_field_strength, self.config.bias_field_strength
                )
            )
            field = (1 + ax * xx + ay * yy).clamp(0.8, 1.2)
            volume = (volume * field).clamp(0, 1)
        if self.config.noise_std > 0:
            volume = (volume + torch.randn_like(volume) * self.config.noise_std).clamp(
                0, 1
            )
        if self.config.slice_dropout > 0:
            drop = torch.rand(volume.shape[0]) < self.config.slice_dropout
            volume[drop] = 0
        return volume

    def _read_volume(self, path, stream_name: str) -> np.ndarray:
        policy = self.config.physical_scale_policy
        policy_tag = policy.get("policy_sha256", "") if policy is not None else "legacy"
        key = f"{path}|{stream_name}|{policy_tag}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if policy is None:
            raw = read_dicom_series(path)
        else:
            raw, stats = read_dicom_series(path, return_stats=True)
            plane = {
                "sagittal": "Sagittal",
                "coronal": "Coronal",
                "axial": "Axial",
            }[stream_name.split("_", 1)[0]]
            raw, _ = resample_volume_inplane(
                raw,
                source_spacing_mm=stats.get("pixel_spacing_mm"),
                plane=plane,
                policy=policy,
            )
        self._cache.put(key, raw)
        return raw

    def _training_view(self, raw: np.ndarray) -> torch.Tensor:
        choices = self.config.train_gap_choices
        gap = int(choices[int(torch.randint(len(choices), (1,)).item())])
        jitter_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(jitter_seed)
        volume = preprocess_triplets(
            raw,
            n_slices=self.config.n_slices,
            image_size=self.config.image_size,
            gap=gap,
            center_offset=0,
            jitter=self.config.center_jitter,
            rng=rng,
        )
        return self._augment_mri(volume)

    def _evaluation_view(self, raw: np.ndarray) -> torch.Tensor:
        if self.config.tta_center_offsets:
            return torch.stack(
                [
                    preprocess_triplets(
                        raw,
                        n_slices=self.config.n_slices,
                        image_size=self.config.image_size,
                        gap=self.config.triplet_gap,
                        center_offset=int(offset),
                        jitter=0,
                    )
                    for offset in self.config.tta_center_offsets
                ],
                dim=0,
            )
        return preprocess_triplets(
            raw,
            n_slices=self.config.n_slices,
            image_size=self.config.image_size,
            gap=self.config.triplet_gap,
            center_offset=self.config.center_offset,
            jitter=0,
        )

    def _load(self, uid, series_uid, stream_name: str):
        if not series_uid:
            return self._zero(), 0.0
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            return self._zero(), 0.0
        try:
            raw = self._read_volume(path, stream_name)
            volume = self._training_view(raw) if self.train else self._evaluation_view(raw)
        except Exception:
            if self.config.strict_dicom:
                raise
            return self._zero(), 0.0
        return volume, 1.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        mapping = self.series_index.get(uid, {})
        volumes, present = [], []
        for name in self.stream_names:
            volume, flag = self._load(uid, mapping.get(name), name)
            volumes.append(volume)
            present.append(flag)
        stacked = torch.stack(volumes)
        if self.config.tta_center_offsets:
            stacked = stacked.permute(1, 0, 2, 3, 4, 5).contiguous()
        item = {
            "study_uid": uid,
            "volumes": stacked,
            "present": torch.tensor(present, dtype=torch.float32),
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(
                np.asarray(self.targets[idx], dtype=np.float32)
            )
        if self.weights is not None:
            item["weight"] = torch.from_numpy(
                np.asarray(self.weights[idx], dtype=np.float32)
            )
        return item


class VariableSeriesDataset(SliceStreamDataset):
    """Knee dataset returning the actual eligible series count for each study."""

    def __init__(self, study_uids, series_records, config, targets=None, weights=None, train=False):
        super().__init__(study_uids, {}, config, targets=targets, weights=weights, train=train)
        if config.physical_scale_policy is not None:
            raise ValueError("B12-v1 freezes legacy resize; physical-scale normalization is disabled")
        self.series_records = series_records
        missing = [uid for uid in self.study_uids if not self.series_records.get(uid)]
        if missing:
            raise ValueError(f"B12 has {len(missing)} study/studies with zero eligible series")

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        volumes, present, meta = [], [], []
        for record in records:
            volume, flag = self._load(uid, record["series_uid"], "series")
            volumes.append(volume)
            present.append(flag)
            meta.append([record["plane_id"], record["fluid_id"], record["fat_id"]])
        stacked = torch.stack(volumes)
        if self.config.tta_center_offsets:
            # [K,V,S,C,H,W] -> [V,K,S,C,H,W]
            stacked = stacked.permute(1, 0, 2, 3, 4, 5).contiguous()
        item = {
            "study_uid": uid,
            "volumes": stacked,
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(meta, dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[idx], dtype=np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[idx], dtype=np.float32))
        return item


class DenseTripletDataset(VariableSeriesDataset):
    """All-series 32-centre B37 dataset with native crop and one 448 resize."""

    def __init__(
        self,
        study_uids,
        series_records,
        config,
        *,
        crop_focus_policy: dict,
        center_offsets: tuple[int, ...] = (0,),
        targets=None,
        weights=None,
    ) -> None:
        super().__init__(
            study_uids,
            series_records,
            config,
            targets=targets,
            weights=weights,
            train=False,
        )
        self.crop_focus_policy = validate_crop_focus_policy(crop_focus_policy)
        self.center_offsets = tuple(int(x) for x in center_offsets)
        if int(config.image_size) != SQUARE_IMAGE_SIZE:
            raise ValueError("B37 dataset requires 448x448 output")
        if not self.center_offsets:
            raise ValueError("B37 requires at least one center offset")
        if not np.isclose(
            float(self.crop_focus_policy["crop_fraction"]),
            CROP_FRACTION,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("B37 dataset requires the fixed 90% crop")

    def _zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        views = len(self.center_offsets)
        images = torch.zeros(
            views,
            DENSE_SLICES,
            3,
            SQUARE_IMAGE_SIZE,
            SQUARE_IMAGE_SIZE,
            dtype=torch.float32,
        )
        positions = torch.zeros(
            views,
            DENSE_SLICES,
            dtype=torch.float32,
        )
        return images, positions

    def _load_b37(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero()
            return image, position, 0.0
        try:
            raw = self._read_volume(path, plane.lower())
            images, positions = [], []
            for offset in self.center_offsets:
                image, pos = prepare_square_triplets(
                    raw,
                    image_size=SQUARE_IMAGE_SIZE,
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                    crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
                )
                images.append(image)
                positions.append(torch.from_numpy(pos))
            return torch.stack(images), torch.stack(positions), 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero()
            return image, position, 0.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        volumes, positions, present, meta = [], [], [], []
        for record in records:
            image, position, flag = self._load_b37(
                uid,
                record["series_uid"],
                str(record["plane"]),
            )
            volumes.append(image)
            positions.append(position)
            present.append(flag)
            meta.append(
                [record["plane_id"], record["fluid_id"], record["fat_id"]]
            )
        volume = torch.stack(volumes).permute(1, 0, 2, 3, 4, 5).contiguous()
        position = torch.stack(positions).permute(1, 0, 2).contiguous()
        item = {
            "study_uid": uid,
            "volumes": volume,
            "slice_position": position,
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(meta, dtype=torch.long),
        }
        if len(self.center_offsets) == 1:
            item["volumes"] = item["volumes"][0]
            item["slice_position"] = item["slice_position"][0]
        if self.targets is not None:
            item["target"] = torch.from_numpy(
                np.asarray(self.targets[idx], dtype=np.float32)
            )
        if self.weights is not None:
            item["weight"] = torch.from_numpy(
                np.asarray(self.weights[idx], dtype=np.float32)
            )
        return item


class StudyDataset(DenseTripletDataset):
    """B37 all-series dataset that returns a list of rectangular series tensors."""

    def __init__(
        self,
        study_uids,
        series_records,
        config,
        *,
        crop_focus_policy: dict,
        center_offsets: tuple[int, ...] = (0,),
        targets=None,
        weights=None,
    ) -> None:
        super().__init__(
            study_uids,
            series_records,
            config,
            crop_focus_policy=crop_focus_policy,
            center_offsets=center_offsets,
            targets=targets,
            weights=weights,
        )
        policy = validate_crop_focus_policy(crop_focus_policy)
        if not np.isclose(float(policy["crop_fraction"]), CROP_FRACTION, atol=1e-12, rtol=0):
            raise ValueError("B42 dataset requires B37's fixed 90% native crop")

    def _zero_b42(self) -> tuple[torch.Tensor, torch.Tensor]:
        views = len(self.center_offsets)
        image = torch.zeros(
            views,
            DENSE_SLICES,
            3,
            REFERENCE_SIDE,
            REFERENCE_SIDE,
            dtype=torch.float32,
        )
        position = torch.zeros(views, DENSE_SLICES, dtype=torch.float32)
        return image, position

    def _load_b42(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root, self.config.split, uid, str(series_uid)
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero_b42()
            return image, position, 0.0
        try:
            raw = self._read_volume(path, plane.lower())
            images, positions = [], []
            for offset in self.center_offsets:
                image, position = prepare_series_triplets(
                    raw,
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                    crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
                )
                images.append(image)
                positions.append(torch.from_numpy(position))
            return torch.stack(images), torch.stack(positions), 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero_b42()
            return image, position, 0.0

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        volumes, positions, present, meta, geometry = [], [], [], [], []
        for record in records:
            image, position, flag = self._load_b42(
                uid, record["series_uid"], str(record["plane"])
            )
            volumes.append(image)
            positions.append(position)
            present.append(flag)
            meta.append([record["plane_id"], record["fluid_id"], record["fat_id"]])
            geometry.append(
                {
                    "series_uid": str(record["series_uid"]),
                    "height": int(image.shape[-2]),
                    "width": int(image.shape[-1]),
                    "present": bool(flag > 0),
                }
            )
        position_tensor = torch.stack(positions)
        if len(self.center_offsets) == 1:
            volumes = [volume[0] for volume in volumes]
            position_tensor = position_tensor[:, 0]
        item = {
            "study_uid": uid,
            "volumes": volumes,
            "slice_position": position_tensor,
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(meta, dtype=torch.long),
            "geometry": geometry,
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[idx], dtype=np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[idx], dtype=np.float32))
        return item

