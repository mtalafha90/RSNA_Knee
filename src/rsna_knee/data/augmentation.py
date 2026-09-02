"""Distorting a training study so the model learns the knee rather than the picture."""
from __future__ import annotations

from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
import torch

from ..constants import DEFAULT_SEED
from .dataset import StudyDataset


DEFAULT_SLICE_JITTER = 0


# Seed offsets are B53's own, so its augmentation draws and loader order do not
# accidentally coincide with anything B52 did.
AUGMENT_SEED_OFFSET = 53_000_003


class AugmentationPolicy:
    """How hard to distort a training study.

    The defaults are read from the config rather than invented here: these nine
    keys have sat in `b42_constant_area_aspect_sparse.yaml` since B7 and were
    only ever applied to a dataset class this pipeline stopped using.
    """

    __slots__ = (
        "rotation_deg",
        "translate_frac",
        "scale_jitter",
        "gamma_jitter",
        "bias_field_strength",
        "noise_std",
        "slice_dropout",
    )

    def __init__(
        self,
        *,
        rotation_deg: float = 5.0,
        translate_frac: float = 0.03,
        scale_jitter: float = 0.05,
        gamma_jitter: float = 0.12,
        bias_field_strength: float = 0.08,
        noise_std: float = 0.02,
        slice_dropout: float = 0.08,
    ) -> None:
        for name, value in (
            ("rotation_deg", rotation_deg),
            ("translate_frac", translate_frac),
            ("scale_jitter", scale_jitter),
            ("gamma_jitter", gamma_jitter),
            ("bias_field_strength", bias_field_strength),
            ("noise_std", noise_std),
            ("slice_dropout", slice_dropout),
        ):
            if float(value) < 0:
                raise ValueError(f"{name} cannot be negative")
            setattr(self, name, float(value))
        if not 0 <= self.slice_dropout < 1:
            raise ValueError("slice_dropout must be in [0, 1)")

    @classmethod
    def from_config(cls, config: dict) -> "AugmentationPolicy":
        """Take every value from the frozen config, so none is invented here."""
        return cls(
            rotation_deg=float(config.get("b7_rotation_deg", 5.0)),
            translate_frac=float(config.get("b7_translate_frac", 0.03)),
            scale_jitter=float(config.get("b7_scale_jitter", 0.05)),
            gamma_jitter=float(config.get("b7_gamma_jitter", 0.12)),
            bias_field_strength=float(config.get("b7_bias_field_strength", 0.08)),
            noise_std=float(config.get("b7_noise_std", 0.02)),
            slice_dropout=float(config.get("b7_slice_dropout", 0.08)),
        )

    @classmethod
    def disabled(cls) -> "AugmentationPolicy":
        """Everything zero: what B52 actually ran, whatever its flag said."""
        return cls(
            rotation_deg=0.0,
            translate_frac=0.0,
            scale_jitter=0.0,
            gamma_jitter=0.0,
            bias_field_strength=0.0,
            noise_std=0.0,
            slice_dropout=0.0,
        )

    def to_dict(self) -> dict:
        return {name: float(getattr(self, name)) for name in self.__slots__}

    def active(self) -> dict:
        return {name: value for name, value in self.to_dict().items() if value > 0}

    def is_disabled(self) -> bool:
        return not self.active()


def augment_series(
    series: torch.Tensor, policy: AugmentationPolicy, generator: torch.Generator
) -> torch.Tensor:
    """Distort one prepared B42 series of shape [slices, 3, height, width].

    The operations and their order are `dataset._augment_mri`'s, unchanged. What
    differs is where they run -- on the tensor B42 actually produces -- and that
    every draw comes from the generator passed in rather than from the global
    random state, so a run is reproducible and a DataLoader worker cannot repeat
    another worker's numbers.

    B42 pixels are percentile-normalised into [0, 1] by `_normalise_volume`, so
    the clamps here are the same clamps the original used and mean the same
    thing.
    """
    if series.ndim != 4 or int(series.shape[1]) != 3:
        raise ValueError(f"expected [slices,3,H,W], got {tuple(series.shape)}")

    def uniform(low: float, high: float) -> float:
        if high <= low:
            return low
        drawn = torch.rand((), generator=generator, dtype=torch.float32)
        return float(low + (high - low) * drawn)

    volume = series.float()
    slices, _channels, height, width = volume.shape

    # --- rotation, translation and scale, as one warp ----------------------
    # One interpolation rather than three, so the image is blurred once.
    if policy.rotation_deg > 0 or policy.translate_frac > 0 or policy.scale_jitter > 0:
        angle = uniform(-policy.rotation_deg, policy.rotation_deg)
        # The original scaled the shift by a square image_size. B42 series are
        # rectangles of roughly constant area, so each axis uses its own side --
        # otherwise a tall series would shift far further sideways than up.
        max_x = int(round(policy.translate_frac * width))
        max_y = int(round(policy.translate_frac * height))
        translate = [
            int(round(uniform(-max_x, max_x))) if max_x else 0,
            int(round(uniform(-max_y, max_y))) if max_y else 0,
        ]
        scale = 1.0 + uniform(-policy.scale_jitter, policy.scale_jitter)
        volume = TVF.affine(
            volume,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )

    # --- gamma -------------------------------------------------------------
    if policy.gamma_jitter > 0:
        gamma = 1.0 + uniform(-policy.gamma_jitter, policy.gamma_jitter)
        volume = volume.clamp(0, 1).pow(gamma)

    # --- smooth bias field -------------------------------------------------
    # A gentle tilt across the image, which is what an imperfect receive coil
    # produces. Clamped to [0.8, 1.2] exactly as the original clamped it.
    if policy.bias_field_strength > 0:
        yy = torch.linspace(-1, 1, height, dtype=volume.dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1, 1, width, dtype=volume.dtype).view(1, 1, 1, width)
        ax = uniform(-policy.bias_field_strength, policy.bias_field_strength)
        ay = uniform(-policy.bias_field_strength, policy.bias_field_strength)
        field = (1 + ax * xx + ay * yy).clamp(0.8, 1.2)
        volume = (volume * field).clamp(0, 1)

    # --- noise -------------------------------------------------------------
    if policy.noise_std > 0:
        noise = torch.randn(volume.shape, generator=generator, dtype=volume.dtype)
        volume = (volume + policy.noise_std * noise).clamp(0, 1)

    # --- slice dropout -----------------------------------------------------
    if policy.slice_dropout > 0 and slices > 1:
        draw = torch.rand(slices, generator=generator, dtype=torch.float32)
        drop = draw < policy.slice_dropout
        # The original had no such guard, and at p=0.08 over 32 slices it would
        # essentially never fire. It costs nothing and the case it prevents --
        # a blank study still carrying a real label -- teaches the model
        # something false rather than nothing.
        if bool(drop.all()):
            drop[int(torch.argmax(draw))] = False
        volume = volume * (~drop).to(volume.dtype).view(slices, 1, 1, 1)

    return volume


class AugmentedStudyDataset(StudyDataset):
    """The B42 dataset, with the configured augmentation actually applied.

    Subclassed rather than edited. B42's geometry contract -- the 90% native
    crop, the constant-area resize, the 32 slice centres -- runs first and
    untouched; augmentation happens afterwards, on the tensor it produced. The
    validation dataset is a plain `StudyDataset`, so the two
    surfaces stay exactly as comparable as they were.
    """

    def __init__(
        self,
        *args,
        policy: AugmentationPolicy | None = None,
        seed: int = DEFAULT_SEED,
        slice_jitter: int = DEFAULT_SLICE_JITTER,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.policy = policy or AugmentationPolicy.disabled()
        self.augment_seed = int(seed) + AUGMENT_SEED_OFFSET
        self.slice_jitter = int(slice_jitter)
        if self.slice_jitter < 0:
            raise ValueError("slice_jitter cannot be negative")
        self.epoch = 0
        self._draw: torch.Generator | None = None

    def set_epoch(self, epoch: int) -> None:
        """Give the next pass a different draw. The training loop calls this."""
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        """A generator fixed by run seed, epoch and study position.

        Not shared state: a DataLoader worker holds its own copy of the dataset,
        so one generator on the instance would give every worker the same
        numbers and silently reduce the augmentation to one repeated draw.
        """
        generator = torch.Generator()
        generator.manual_seed(
            (self.augment_seed * 1_000_003 + self.epoch * 9_176 + index) % (2**31 - 1)
        )
        return generator

    def _load_b42(self, uid: str, series_uid: str, plane: str):
        """Optionally shift the slice centres before the frozen loader runs.

        Off by default. When on, it shifts every centre in a series together,
        which is what `center_offset` already means here, rather than reaching
        into the frozen `slice_centres`.
        """
        if self.slice_jitter <= 0 or self._draw is None:
            return super()._load_b42(uid, series_uid, plane)

        span = 2 * self.slice_jitter + 1
        shift = int(torch.randint(0, span, (1,), generator=self._draw).item()) - self.slice_jitter
        original = self.center_offsets
        self.center_offsets = tuple(offset + shift for offset in original)
        try:
            return super()._load_b42(uid, series_uid, plane)
        finally:
            self.center_offsets = original

    def __getitem__(self, index: int) -> dict:
        self._draw = self._generator(index)
        try:
            item = super().__getitem__(index)
        finally:
            draw = self._draw
            self._draw = None

        if self.policy.is_disabled():
            return item

        present = item["present"]
        volumes = item["volumes"]
        augmented = []
        for position, volume in enumerate(volumes):
            # A masked series is a zero placeholder the model already ignores.
            # Adding noise to it would turn a placeholder into something that is
            # no longer zero.
            if float(present[position]) <= 0:
                augmented.append(volume)
            else:
                augmented.append(augment_series(volume, self.policy, draw))
        item["volumes"] = augmented
        return item


def verify_augmentation_reaches_pixels(dataset: AugmentedStudyDataset) -> dict:
    """Draw the same study twice and confirm the pixels actually differ.

    This is the check B52 did not have. Its augmentation flag set fields on a
    config object that the dataset never read, and nothing in a 27-hour run
    would have told anyone. Reading two draws is the only way to know.
    """
    if not len(dataset):
        raise RuntimeError("cannot verify augmentation on an empty dataset")
    if dataset.policy.is_disabled():
        raise RuntimeError(
            "verify_augmentation_reaches_pixels was called with augmentation off"
        )

    dataset.set_epoch(1)
    first = dataset[0]
    dataset.set_epoch(2)
    second = dataset[0]

    live = [
        position
        for position in range(len(first["present"]))
        if float(first["present"][position]) > 0
    ]
    if not live:
        raise RuntimeError("the first study has no readable series to compare")

    differences = [
        float((first["volumes"][position] - second["volumes"][position]).abs().max())
        for position in live
    ]
    report = {
        "series_compared": len(live),
        "series_that_changed": int(sum(1 for value in differences if value > 0)),
        "max_absolute_difference": max(differences),
        "policy": dataset.policy.active(),
    }
    if report["series_that_changed"] == 0:
        raise RuntimeError(
            "augmentation did not reach the pixels: two draws of the same "
            "study are identical. This is exactly the B52 failure, and the run "
            "would be B52 under another name."
        )

    # Reset, so the verification cannot change what epoch 1 trains on.
    dataset.set_epoch(0)
    return report


def verify_augmentation_is_off(dataset) -> dict:
    """Draw the same study twice and confirm the pixels are identical.

    The mirror of `verify_augmentation_reaches_pixels`, and the control arm's
    equivalent. `--no-augment` exists to reproduce the run this experiment is
    measured against, and a control that augmented a little would compare
    nothing to nothing. Skipping the check for the control would leave the arm
    that most needs a guarantee as the only one without one.
    """
    if not len(dataset):
        raise RuntimeError("cannot verify augmentation on an empty dataset")
    if not dataset.policy.is_disabled():
        raise RuntimeError(
            "verify_augmentation_is_off was called with augmentation on; "
            "use verify_augmentation_reaches_pixels"
        )

    dataset.set_epoch(1)
    first = dataset[0]
    dataset.set_epoch(2)
    second = dataset[0]

    live = [
        position
        for position in range(len(first["present"]))
        if float(first["present"][position]) > 0
    ]
    if not live:
        raise RuntimeError("the first study has no readable series to compare")

    differences = [
        float((first["volumes"][position] - second["volumes"][position]).abs().max())
        for position in live
    ]
    report = {
        "series_compared": len(live),
        "series_that_changed": int(sum(1 for value in differences if value > 0)),
        "max_absolute_difference": max(differences),
        "policy": dataset.policy.active(),
        "augmentation_enabled": False,
    }
    if report["series_that_changed"]:
        raise RuntimeError(
            "augmentation is switched off, but two draws of the same study "
            f"differ by {report['max_absolute_difference']:.6f} across "
            f"{report['series_that_changed']} series. Something is still "
            "distorting the pixels, so this is not the control arm it claims "
            "to be."
        )

    dataset.set_epoch(0)
    return report

