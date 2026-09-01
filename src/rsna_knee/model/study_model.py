"""The full model: encoder, study hierarchy, sparse local branch, and the gate that mixes them."""
from __future__ import annotations

from dataclasses import dataclass
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
import torch
import torch.nn.functional as F

from ..data.slice_selection import BASE_SLICES, DENSE_SLICES
from ..imaging.crop_policy import CROP_FRACTION, crop_focus_policy
from ..imaging.triplets import PADDING_MODE, REFERENCE_AREA, RESIZE_POLICY, SQUARE_IMAGE_SIZE, STRIDE_ALIGNMENT
from .encoder import freeze_encoder, unfreeze_encoder_tail
from .sparse_head import SparseEvidenceHead


ENCODER_CHUNK_SIZE = 4


EFFECTIVE_BATCH = 2


ADAPTED_HIERARCHY_VERSION = "b50_adapted_hierarchy_mil_v1"


ADAPTED_HIERARCHY_EXPERIMENT = "B50_ADAPTED_STUDY_HIERARCHY"


GRID_SIZE = 6


ENCODER_LR_SCALE = 0.05


TOP_K = 8


ENCODER_TRAINABLE_STAGES = 1


LOCAL_AUX_WEIGHT = 1.0


TEMPERATURE = 1.0


# Bypassed under eval, so it would collect no gradient. Excluded explicitly so
# the B34 inference contract stays exactly reconstructible.
ALWAYS_FROZEN_PREFIXES = ("encoder.", "local_context.")


RAGGED_SERIES_VERSION = "b42_constant_area_native_aspect_rectangular_sparse_mil_v1"


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    local_logits: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor


SPARSE_MIL_VERSION = "b37_highres448_b36_sparse_mil_encoder_tail_v1"


def require_sparse_mil_contract(config: dict) -> dict:
    """Require every prospectively frozen B37 high-resolution mechanism choice."""
    image_size = int(config.get("b7_image_size", SQUARE_IMAGE_SIZE))
    grid_size = int(config.get("b37_grid_size", GRID_SIZE))
    top_k = int(config.get("b37_top_k", TOP_K))
    temperature = float(config.get("b37_temperature", TEMPERATURE))
    aux = float(config.get("b37_local_aux_weight", LOCAL_AUX_WEIGHT))
    stages = int(
        config.get("b37_encoder_trainable_stages", ENCODER_TRAINABLE_STAGES)
    )
    scale = float(config.get("b37_encoder_lr_scale", ENCODER_LR_SCALE))
    chunk = int(config.get("b37_encoder_chunk_size", ENCODER_CHUNK_SIZE))

    expected = {
        "b7_image_size": (image_size, SQUARE_IMAGE_SIZE),
        "b37_grid_size": (grid_size, GRID_SIZE),
        "b37_top_k": (top_k, TOP_K),
        "b37_encoder_trainable_stages": (
            stages,
            ENCODER_TRAINABLE_STAGES,
        ),
        "b37_encoder_chunk_size": (chunk, ENCODER_CHUNK_SIZE),
    }
    for key, (value, frozen) in expected.items():
        if value != frozen:
            raise ValueError(f"B37 freezes {key}={frozen}; got {value}")
    for key, value, frozen in (
        ("b37_temperature", temperature, TEMPERATURE),
        ("b37_local_aux_weight", aux, LOCAL_AUX_WEIGHT),
        ("b37_encoder_lr_scale", scale, ENCODER_LR_SCALE),
    ):
        if not np.isclose(value, frozen, atol=1e-12, rtol=0):
            raise ValueError(f"B37 freezes {key}={frozen}; got {value}")

    crop = crop_focus_policy({**config, "b7_image_size": 224})
    if not np.isclose(
        float(crop["crop_fraction"]), CROP_FRACTION, atol=1e-12, rtol=0
    ):
        raise ValueError("B37 requires the historical fixed 90% center crop")
    return crop


def hierarchy_parameter_names(base: nn.Module) -> list[str]:
    """Every base parameter B50 is allowed to unfreeze, in a stable order."""
    return [
        name
        for name, _ in base.named_parameters()
        if not name.startswith(ALWAYS_FROZEN_PREFIXES)
    ]


class SparseMILResidual(nn.Module):
    """Frozen B34 aggregation + trainable encoder tail + B36 sparse local head."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        grid_size: int = GRID_SIZE,
        top_k: int = TOP_K,
        temperature: float = TEMPERATURE,
        encoder_trainable_stages: int = ENCODER_TRAINABLE_STAGES,
        encoder_chunk_size: int = ENCODER_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        self.base = base_model
        freeze_encoder(self.base)
        for name, parameter in self.base.named_parameters():
            if not name.startswith("encoder."):
                parameter.requires_grad_(False)
        self.finetune = unfreeze_encoder_tail(
            self.base,
            int(encoder_trainable_stages),
        )
        self.encoder_trainable_stages = int(encoder_trainable_stages)
        self.encoder_chunk_size = int(encoder_chunk_size)
        if self.encoder_chunk_size < 1:
            raise ValueError("B37 encoder chunk size must be positive")
        self.base.encoder_batch_size = self.encoder_chunk_size
        self.base.eval()
        if int(self.base.n_slices) != BASE_SLICES:
            raise ValueError("B37 requires a 16-slice B34 base")

        dim = int(self.base.encoder.out_dim)
        self.head = SparseEvidenceHead(
            dim,
            grid_size=int(grid_size),
            top_k=int(top_k),
            temperature=float(temperature),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The deployed B34 hierarchy remains in its exact evaluation path; only
        # encoder gradients and the new sparse head are enabled.
        self.base.eval()
        self.base.encoder.eval()
        self.head.train(mode)
        return self

    def _encode_chunk(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.base.encoder
        normalized = encoder._normalize(chunk)
        fmap = encoder.features(normalized)
        global_feature = encoder.pre_classifier(encoder.avgpool(fmap)).reshape(
            chunk.shape[0], int(encoder.out_dim)
        )
        pooled = F.adaptive_avg_pool2d(
            fmap,
            (int(self.head.grid_size), int(self.head.grid_size)),
        )
        normalized_grid = encoder.pre_classifier[0](pooled)
        spatial = normalized_grid.permute(0, 2, 3, 1).reshape(
            chunk.shape[0], self.head.n_regions, int(encoder.out_dim)
        )
        return global_feature, spatial

    def _encode_active_group(
        self,
        active_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if active_group.ndim != 5:
            raise ValueError("B37 active group must be [N,G,3,H,W]")
        n, g, c, h, w = active_group.shape
        if c != 3 or h != SQUARE_IMAGE_SIZE or w != SQUARE_IMAGE_SIZE:
            raise ValueError("B37 active group requires 3x448x448 triplets")
        flat = active_group.reshape(n * g, c, h, w)
        global_blocks, spatial_blocks = [], []
        use_checkpoint = bool(self.training and self.encoder_trainable_stages > 0)
        for chunk in flat.split(self.encoder_chunk_size, dim=0):
            if use_checkpoint:
                global_feature, spatial = checkpoint(
                    self._encode_chunk,
                    chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                global_feature, spatial = self._encode_chunk(chunk)
            global_blocks.append(global_feature)
            spatial_blocks.append(spatial)
        dim = int(self.base.encoder.out_dim)
        regions = int(self.head.n_regions)
        return (
            torch.cat(global_blocks, dim=0).reshape(n, g, dim),
            torch.cat(spatial_blocks, dim=0).reshape(n, g, regions, dim),
        )

    def _encode_combined(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if volumes.ndim != 6:
            raise ValueError("B37 expects [B,K,32,3,448,448]")
        b, k, s, c, h, w = volumes.shape
        if (
            s != DENSE_SLICES
            or c != 3
            or h != SQUARE_IMAGE_SIZE
            or w != SQUARE_IMAGE_SIZE
        ):
            raise ValueError("B37 input shape does not match the frozen protocol")
        active_indices = torch.nonzero(
            present.reshape(-1) > 0,
            as_tuple=False,
        ).flatten()
        if active_indices.numel() == 0:
            raise RuntimeError("B37 batch has no readable MRI series")
        flat_series = volumes.reshape(b * k, s, c, h, w)
        active = flat_series.index_select(0, active_indices)

        # Preserve the B35/B36 first-16 / extra-16 grouping.  The first group has
        # the exact ordering used by a 16-centre B34 pass at the same resolution.
        base_global, base_spatial = self._encode_active_group(
            active[:, :BASE_SLICES]
        )
        extra_global, extra_spatial = self._encode_active_group(
            active[:, BASE_SLICES:]
        )
        global_active = torch.cat((base_global, extra_global), dim=1)
        spatial_active = torch.cat((base_spatial, extra_spatial), dim=1)

        dim = int(self.base.encoder.out_dim)
        regions = int(self.head.n_regions)
        all_global = global_active.new_zeros((b * k, s, dim)).index_copy(
            0,
            active_indices,
            global_active,
        )
        all_spatial = spatial_active.new_zeros(
            (b * k, s, regions, dim)
        ).index_copy(0, active_indices, spatial_active)
        return (
            all_global.reshape(b, k, s, dim),
            all_spatial.reshape(b, k, s, regions, dim),
        )

    def _base_logits_from_global(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base
        x = global_feature[:, :, :BASE_SLICES]
        plane = base.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = base.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = base.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        mask = present[:, :, None, None].to(x.dtype)
        x = (
            x
            + base.slice_position[None, None, :, :]
            + metadata[:, :, None, :]
        ) * mask
        tokens = base._pool_real_series_b31(x, present)
        padding = present <= 0
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        memory = base.context(tokens, src_key_padding_mask=safe_padding)
        memory = memory.masked_fill(padding[:, :, None], 0.0)
        queries = base.pathology_tokens[None, :, :].expand(memory.shape[0], -1, -1)
        queries = base.pathology_context(queries)
        attended, _ = base.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=safe_padding,
            need_weights=False,
        )
        queries = base.dropout(base.query_norm(queries + attended))
        logits = (
            queries * base.target_weight[None, :, :]
        ).sum(dim=-1) + base.target_bias
        return torch.where(empty[:, None], base.target_bias[None, :], logits)

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> ModelOutput:
        global_feature, spatial = self._encode_combined(volumes, present)
        base_logits = self._base_logits_from_global(
            global_feature,
            present,
            series_meta,
        )
        local_logits, top_indices, top_values = self.head(
            spatial,
            present,
            series_meta,
            slice_position,
        )
        gate = self.head.effective_gate().to(dtype=local_logits.dtype)
        logits = base_logits.float() + gate[None, :] * local_logits.float()
        return ModelOutput(
            logits=logits,
            base_logits=base_logits,
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values,
        )

    @torch.no_grad()
    def base_equivalence_error_448(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> float:
        """Guard only the reconstructed B34 function at the new 448 input size."""
        was_training = self.training
        self.eval()
        reference = self.base(
            volumes[:, :, :BASE_SLICES],
            present,
            series_meta,
        )
        global_feature, _ = self._encode_combined(volumes, present)
        reconstructed = self._base_logits_from_global(
            global_feature,
            present,
            series_meta,
        )
        if was_training:
            self.train(True)
        return float(
            (reference.float() - reconstructed.float()).abs().max().item()
        )

    def state(self) -> dict:
        trainable_encoder = sum(
            p.numel() for p in self.base.encoder.parameters() if p.requires_grad
        )
        frozen_base = sum(
            p.numel() for name, p in self.base.named_parameters()
            if not name.startswith("encoder.") and not p.requires_grad
        )
        return {
            "version": SPARSE_MIL_VERSION,
            "image_size": SQUARE_IMAGE_SIZE,
            "crop_fraction": CROP_FRACTION,
            "grid_size": int(self.head.grid_size),
            "regions_per_slice": int(self.head.n_regions),
            "dense_slices": DENSE_SLICES,
            "base_slices": BASE_SLICES,
            "top_k": int(self.head.top_k),
            "temperature": float(self.head.temperature),
            "encoder_chunk_size": self.encoder_chunk_size,
            "encoder_trainable_stages": self.encoder_trainable_stages,
            "encoder_trainable_parameters": int(trainable_encoder),
            "frozen_nonencoder_base_parameters": int(frozen_base),
            "head": self.head.state(),
        }


def require_geometry_contract(config: dict) -> dict:
    """Require the fixed B42 geometry while inheriting every B37 model control."""
    crop = require_sparse_mil_contract(config)
    expected_int = {
        "b42_reference_area": REFERENCE_AREA,
        "b42_stride_alignment": STRIDE_ALIGNMENT,
        "b42_effective_batch": EFFECTIVE_BATCH,
    }
    for key, expected in expected_int.items():
        value = int(config.get(key, expected))
        if value != expected:
            raise ValueError(f"B42 freezes {key}={expected}; got {value}")
    policy = str(config.get("b42_resize_policy", RESIZE_POLICY))
    if policy != RESIZE_POLICY:
        raise ValueError(
            f"B42 freezes b42_resize_policy={RESIZE_POLICY!r}; got {policy!r}"
        )
    padding = str(config.get("b42_padding_mode", PADDING_MODE))
    if padding != PADDING_MODE:
        raise ValueError(
            f"B42 freezes b42_padding_mode={PADDING_MODE!r}; got {padding!r}"
        )
    if int(config.get("b37_encoder_chunk_size", ENCODER_CHUNK_SIZE)) != ENCODER_CHUNK_SIZE:
        raise ValueError("B42 retains B37 encoder chunk size")
    return crop


class RaggedSeriesSparseMIL(SparseMILResidual):
    """B37 hierarchy/head with per-series rectangular ConvNeXt encoding."""

    def _encode_rect_group(self, group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if group.ndim != 4:
            raise ValueError("B42 group must be [G,3,H,W]")
        g, c, h, w = group.shape
        if c != 3 or h < STRIDE_ALIGNMENT or w < STRIDE_ALIGNMENT:
            raise ValueError("B42 group requires 3-channel stride-valid rectangles")
        if h % STRIDE_ALIGNMENT or w % STRIDE_ALIGNMENT:
            raise ValueError("B42 rectangular inputs must be stride aligned")
        global_blocks, spatial_blocks = [], []
        # `gradient_checkpointing` defaults to True when absent, so every model
        # built before this attribute existed behaves exactly as it did. B52 turns
        # it off: it trades speed for memory, and with five encoder stages the run
        # peaks near 1.4 GiB of a 16 GiB card, so the trade is the wrong way round.
        use_checkpoint = bool(
            self.training
            and self.encoder_trainable_stages > 0
            and getattr(self, "gradient_checkpointing", True)
        )
        for chunk in group.split(self.encoder_chunk_size, dim=0):
            if use_checkpoint:
                global_feature, spatial = checkpoint(
                    self._encode_chunk,
                    chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                global_feature, spatial = self._encode_chunk(chunk)
            global_blocks.append(global_feature)
            spatial_blocks.append(spatial)
        return torch.cat(global_blocks, dim=0), torch.cat(spatial_blocks, dim=0)

    def _encode_ragged_study(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(volumes, list) or not volumes:
            raise ValueError("B42 expects a non-empty list of series tensors")
        if present.ndim == 2:
            if int(present.shape[0]) != 1:
                raise ValueError("B42 forward processes exactly one study at a time")
            present_flat = present[0]
        elif present.ndim == 1:
            present_flat = present
        else:
            raise ValueError("B42 present mask must be [K] or [1,K]")
        if len(volumes) != int(present_flat.numel()):
            raise ValueError("B42 volumes/present series count mismatch")

        global_rows: list[torch.Tensor | None] = []
        spatial_rows: list[torch.Tensor | None] = []
        template_global = template_spatial = None
        for series_tensor, flag in zip(volumes, present_flat):
            if series_tensor.ndim != 4 or int(series_tensor.shape[0]) != DENSE_SLICES:
                raise ValueError("B42 series must be [32,3,H,W]")
            if float(flag.detach().item()) <= 0:
                global_rows.append(None)
                spatial_rows.append(None)
                continue
            base_global, base_spatial = self._encode_rect_group(
                series_tensor[:BASE_SLICES]
            )
            extra_global, extra_spatial = self._encode_rect_group(
                series_tensor[BASE_SLICES:]
            )
            global_series = torch.cat((base_global, extra_global), dim=0)
            spatial_series = torch.cat((base_spatial, extra_spatial), dim=0)
            global_rows.append(global_series)
            spatial_rows.append(spatial_series)
            if template_global is None:
                template_global, template_spatial = global_series, spatial_series

        if template_global is None or template_spatial is None:
            raise RuntimeError("B42 study has no readable MRI series")
        for index in range(len(global_rows)):
            if global_rows[index] is None:
                global_rows[index] = torch.zeros_like(template_global)
                spatial_rows[index] = torch.zeros_like(template_spatial)
        global_feature = torch.stack([x for x in global_rows if x is not None], dim=0).unsqueeze(0)
        spatial = torch.stack([x for x in spatial_rows if x is not None], dim=0).unsqueeze(0)
        return global_feature, spatial

    def forward(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> ModelOutput:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        if slice_position.ndim == 2:
            slice_position = slice_position.unsqueeze(0)
        global_feature, spatial = self._encode_ragged_study(volumes, present)
        base_logits = self._base_logits_from_global(global_feature, present, series_meta)
        local_logits, top_indices, top_values = self.head(
            spatial, present, series_meta, slice_position
        )
        gate = self.head.effective_gate().to(dtype=local_logits.dtype)
        logits = base_logits.float() + gate[None, :] * local_logits.float()
        return ModelOutput(
            logits=logits,
            base_logits=base_logits,
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values,
        )

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": RAGGED_SERIES_VERSION,
                "input_geometry": RESIZE_POLICY,
                "reference_pixel_area": REFERENCE_AREA,
                "stride_alignment": STRIDE_ALIGNMENT,
                "padding_mode": PADDING_MODE,
                "ragged_series_encoding": True,
            }
        )
        return state


class KneeStudyModel(RaggedSeriesSparseMIL):
    """B42 exactly, except that the study aggregation may receive gradients."""

    def __init__(self, *args, adapt_hierarchy: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adapt_hierarchy = bool(adapt_hierarchy)
        self.hierarchy_names = hierarchy_parameter_names(self.base)
        if self.adapt_hierarchy:
            lookup = dict(self.base.named_parameters())
            for name in self.hierarchy_names:
                lookup[name].requires_grad_(True)
        # The base stays in eval mode either way. See the module docstring: this
        # keeps B34's training-only local-context scaffold bypassed, which is
        # part of the frozen inference contract and not a detail of B50's.
        self.base.eval()

    def train(self, mode: bool = True):
        """Never put the base in training mode, whatever B50 unfroze."""
        super().train(mode)
        self.base.eval()
        self.base.encoder.eval()
        self.head.train(mode)
        return self

    def hierarchy_parameters(self) -> list[nn.Parameter]:
        lookup = dict(self.base.named_parameters())
        return [lookup[name] for name in self.hierarchy_names]

    def trainable_parameter_summary(self) -> dict:
        """What is actually learning, so the control can be checked against B42."""
        encoder = sum(
            p.numel() for p in self.base.encoder.parameters() if p.requires_grad
        )
        hierarchy = sum(p.numel() for p in self.hierarchy_parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "adapt_hierarchy": self.adapt_hierarchy,
            "encoder_trainable_parameters": int(encoder),
            "hierarchy_trainable_parameters": int(hierarchy),
            "head_trainable_parameters": int(head),
            "hierarchy_parameters_available": int(
                sum(p.numel() for p in self.hierarchy_parameters())
            ),
        }

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": ADAPTED_HIERARCHY_VERSION,
                "experiment": ADAPTED_HIERARCHY_EXPERIMENT,
                "trainable": self.trainable_parameter_summary(),
                "base_module_mode": "eval throughout; B34 local-context scaffold bypassed",
            }
        )
        return state

