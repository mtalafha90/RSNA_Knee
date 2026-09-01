"""The slice encoder that reads pixels, and which of its stages are allowed to learn."""
from __future__ import annotations

from pathlib import Path
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
import argparse
import hashlib
import json
import math
import numpy as np
import time
import torch
import torch.nn.functional as F


def encoder_state_sha256(encoder: nn.Module) -> str:
    """Deterministic fingerprint of encoder parameter/buffer values."""
    digest = hashlib.sha256()
    state = encoder.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        flat = tensor.reshape(-1)
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_encoder(model: nn.Module) -> None:
    """Freeze encoder gradients and training-time stochastic behaviour."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("B17 failed to freeze every encoder parameter")


# The encoder's parts, listed from the output backwards. Unfreezing `n` stages
# takes the first `n` of these. `pre_classifier` is the final normalisation and
# is tiny; it belongs with the last stage rather than on its own.
ENCODER_BLOCKS_FROM_OUTPUT = (
    ("pre_classifier", "features.7"),   # last stage plus its output norm
    ("features.6",),                    # the downsample feeding it
    ("features.5",),                    # third stage
    ("features.4",),
    ("features.3",),
)


DINOV3_ENCODER_VERSION = "dinov3_convnext_slice_encoder_v1"


# ConvNeXt tiny and small share the same channel widths and differ only in
# depth, so both emit 768-d features and both drop in unchanged.  Base (1024)
# and large (1536) do not.
DROP_IN_VARIANTS = ("tiny", "small")


# The frozen head is built around this width.  A wider encoder would change the
# study representation as well as the features, so it is refused rather than
# silently reshaped.
REQUIRED_OUTPUT_WIDTH = 768


DINOV3_CONVNEXT_MODELS = {
    "tiny": "convnext_tiny.dinov3_lvd1689m",
    "small": "convnext_small.dinov3_lvd1689m",
    "base": "convnext_base.dinov3_lvd1689m",
    "large": "convnext_large.dinov3_lvd1689m",
}


class ConvNeXtSliceEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, *, pretrained_weights: bool = True, normalize_input: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_weights else None
        net = convnext_tiny(weights=weights)
        first = net.features[0][0]
        if in_channels != 3:
            replacement = nn.Conv2d(
                in_channels,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=first.bias is not None,
            )
            if pretrained_weights:
                with torch.no_grad():
                    replacement.weight.copy_(first.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
                    if first.bias is not None:
                        replacement.bias.copy_(first.bias)
            net.features[0][0] = replacement
        self.features = net.features
        self.avgpool = net.avgpool
        self.pre_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
        self.out_dim = int(net.classifier[-1].in_features)
        self.normalize_input = bool(normalize_input)
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        mean = rgb_mean if in_channels == 3 else rgb_mean.mean().repeat(in_channels)
        std = rgb_std if in_channels == 3 else rgb_std.mean().repeat(in_channels)
        self.register_buffer("input_mean", mean.view(1, in_channels, 1, 1), persistent=False)
        self.register_buffer("input_std", std.view(1, in_channels, 1, 1), persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return (x - self.input_mean.to(dtype=x.dtype)) / self.input_std.to(dtype=x.dtype)
        return x

    def forward(self, x):
        x = self._normalize(x)
        return self.pre_classifier(self.avgpool(self.features(x)))

    def forward_spatial(self, x: torch.Tensor, grid_size: int = 2) -> torch.Tensor:
        """Return coarse ConvNeXt spatial tokens shaped [N, grid_size**2, D].

        B7 collapses every 2.5D slice to one global vector. B8 reuses the same
        ConvNeXt weights but retains a small spatial grid from the final feature
        map. The classifier normalization is reused before the grid is flattened,
        so B7/B5 encoder initialization remains meaningful.
        """
        grid_size = int(grid_size)
        if grid_size < 1:
            raise ValueError("grid_size must be >=1")
        x = self._normalize(x)
        feature_map = self.features(x)
        pooled = F.adaptive_avg_pool2d(feature_map, (grid_size, grid_size))
        # ConvNeXt's first classifier module is the learned channel normalization
        # used by the ordinary globally pooled path and accepts NCHW tensors.
        normalized = self.pre_classifier[0](pooled)
        return normalized.permute(0, 2, 3, 1).reshape(x.shape[0], grid_size * grid_size, self.out_dim)


MAX_TRAINABLE_STAGES = len(ENCODER_BLOCKS_FROM_OUTPUT)


class DinoV3SliceEncoder(nn.Module):
    """DINOv3 ConvNeXt encoder exposing the frozen encoder's interface.

    Mirrors :class:`rsna_knee.model.ConvNeXtSliceEncoder`: same constructor
    keywords, same `out_dim` attribute, same `forward` contract mapping a batch
    of slice images to one vector per slice.
    """

    def __init__(
        self,
        in_channels: int = 3,
        *,
        variant: str = "tiny",
        pretrained_weights: bool = True,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "the DINOv3 encoder needs `timm`; install it with `pip install timm`"
            ) from exc

        if variant not in DINOV3_CONVNEXT_MODELS:
            known = ", ".join(sorted(DINOV3_CONVNEXT_MODELS))
            raise ValueError(f"unknown DINOv3 variant {variant!r}; known: {known}")

        self.variant = variant
        self.model_name = DINOV3_CONVNEXT_MODELS[variant]
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=bool(pretrained_weights),
            num_classes=0,
            in_chans=int(in_channels),
        )
        self.out_dim = int(self.backbone.num_features)
        if self.out_dim != REQUIRED_OUTPUT_WIDTH:
            drop_in = ", ".join(sorted(DROP_IN_VARIANTS))
            raise ValueError(
                f"{self.model_name} emits {self.out_dim}-d features but the frozen "
                f"head requires {REQUIRED_OUTPUT_WIDTH}; drop-in variants: {drop_in}"
            )

        self.normalize_input = bool(normalize_input)
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        mean = rgb_mean if in_channels == 3 else rgb_mean.mean().repeat(in_channels)
        std = rgb_std if in_channels == 3 else rgb_std.mean().repeat(in_channels)
        self.register_buffer("input_mean", mean.view(1, in_channels, 1, 1), persistent=False)
        self.register_buffer("input_std", std.view(1, in_channels, 1, 1), persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return (x - self.input_mean.to(dtype=x.dtype)) / self.input_std.to(dtype=x.dtype)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._normalize(x))

    def describe(self) -> dict:
        return {
            "version": DINOV3_ENCODER_VERSION,
            "model_name": self.model_name,
            "variant": self.variant,
            "out_dim": self.out_dim,
            "normalize_input": self.normalize_input,
            "pretraining": "DINOv3 self-supervised, LVD-1689M",
            "licence": "Meta DINOv3 licence",
        }


def _prefixes_for(stages: int) -> tuple[str, ...]:
    if not 0 <= stages <= MAX_TRAINABLE_STAGES:
        raise ValueError(
            f"encoder_trainable_stages must be between 0 and {MAX_TRAINABLE_STAGES}"
        )
    names: list[str] = []
    for block in ENCODER_BLOCKS_FROM_OUTPUT[:stages]:
        names.extend(block)
    return tuple(names)


def attach_dinov3_encoder(
    model: nn.Module,
    *,
    variant: str = "tiny",
    pretrained_weights: bool = True,
) -> DinoV3SliceEncoder:
    """Replace a built model's encoder with the DINOv3 one, in place.

    The head is constructed first and its encoder swapped afterwards, which is
    the same shape of operation as loading report-aligned weights into it.  The
    replacement is refused unless the widths match, because a mismatch would
    change the study representation rather than only the features.
    """
    existing = getattr(model, "encoder", None)
    if existing is None:
        raise ValueError("model has no encoder attribute to replace")

    in_channels = int(getattr(existing, "input_mean").shape[1])
    normalize_input = bool(getattr(existing, "normalize_input", True))
    replacement = DinoV3SliceEncoder(
        in_channels,
        variant=variant,
        pretrained_weights=pretrained_weights,
        normalize_input=normalize_input,
    )
    if replacement.out_dim != int(existing.out_dim):
        raise ValueError(
            f"encoder width mismatch: existing {existing.out_dim}, "
            f"replacement {replacement.out_dim}"
        )

    model.encoder = replacement
    return replacement


def unfreeze_encoder_tail(model: nn.Module, stages: int) -> dict:
    """Free the last `stages` encoder blocks; everything else stays frozen.

    Call after the usual freeze, so the default remains a fully frozen encoder
    and this only ever relaxes it.
    """
    prefixes = _prefixes_for(stages)
    if not prefixes:
        return {
            "encoder_trainable_stages": 0,
            "encoder_trainable_parameters": 0,
            "encoder_trainable_prefixes": [],
        }

    freed = 0
    for name, parameter in model.encoder.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            parameter.requires_grad_(True)
            freed += parameter.numel()

    if freed == 0:
        raise RuntimeError(
            f"no encoder parameters matched {prefixes}; the encoder layout changed"
        )

    # The encoder stays in eval mode on purpose: the forward pass is unchanged,
    # only gradients now flow.
    model.encoder.eval()
    return {
        "encoder_trainable_stages": int(stages),
        "encoder_trainable_parameters": int(freed),
        "encoder_trainable_prefixes": list(prefixes),
    }

