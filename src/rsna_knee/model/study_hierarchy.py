"""Pooling encoded slices into series, and series into one study representation."""
from __future__ import annotations

from torch import nn
from torch.utils.checkpoint import checkpoint
import math
import torch
import torch.nn.functional as F

from ..constants import N_TARGETS
from .encoder import ConvNeXtSliceEncoder


STUDY_HIERARCHY_VERSION = "train_only_zero_init_depthwise_context_eval_exact_bypass_v1"


STUDY_HIERARCHY_ARCHITECTURE = "b31_training_only_local_context_scaffold_eval_bypass_v1"


LOCAL_CONTEXT_VERSION = "parameter_free_ln_plus_zero_init_depthwise_conv1d_k3_v1"


STUDY_HIERARCHY_AGGREGATION = "b29_query_with_b31_context_scaffold_during_training_and_exact_b29_scoring_at_eval_v1"


LOCAL_CONTEXT_PARAMETERS = 768 * 3


COMPLEMENTARY_GATE_PARAMETERS = 768


COMPLEMENTARY_QUERY_PARAMETERS = 768


COMPLEMENTARY_POOL_VERSION = "zero_init_tanh_feature_gate_complementary_query_v1"


class LearnedSeriesPool(nn.Module):
    """Compress one real MRI series from S slice tokens to one learned token."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % int(heads) != 0:
            raise ValueError("series pooling heads must divide feature dimension")
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            dim,
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        if slice_tokens.ndim != 3:
            raise ValueError("series pool expects [N,S,D]")
        query = self.query.expand(slice_tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            query,
            slice_tokens,
            slice_tokens,
            need_weights=False,
        )
        return self.dropout(self.norm(query + attended)).squeeze(1)


STUDY_HIERARCHY_CONTEXT_PARAMETERS = LOCAL_CONTEXT_PARAMETERS


LOCAL_CONTEXT_GATE_PARAMETERS = COMPLEMENTARY_GATE_PARAMETERS


LOCAL_CONTEXT_QUERY_PARAMETERS = COMPLEMENTARY_QUERY_PARAMETERS


LOCAL_CONTEXT_NEW_PARAMETERS = (
    LOCAL_CONTEXT_QUERY_PARAMETERS
    + LOCAL_CONTEXT_GATE_PARAMETERS
    + LOCAL_CONTEXT_PARAMETERS
)


class SeriesHierarchyNet(nn.Module):
    """B12.1: ConvNeXt slices -> learned series tokens -> study Transformer."""

    def __init__(
        self,
        n_slices: int,
        *,
        in_channels: int = 3,
        pretrained_weights: bool = False,
        normalize_input: bool = False,
        dropout: float = 0.25,
        encoder_batch_size: int = 24,
        gradient_checkpointing: bool = True,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_mult: float = 2.0,
        pathology_layers: int = 1,
        series_pool_heads: int | None = None,
    ) -> None:
        super().__init__()
        if n_slices < 1 or encoder_batch_size < 1:
            raise ValueError("n_slices and encoder_batch_size must be positive")
        self.n_slices = int(n_slices)
        self.in_channels = int(in_channels)
        self.encoder_batch_size = int(encoder_batch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder = ConvNeXtSliceEncoder(
            in_channels,
            pretrained_weights=pretrained_weights,
            normalize_input=normalize_input,
        )
        d = self.encoder.out_dim
        heads = int(transformer_heads)
        if d % heads != 0:
            raise ValueError("transformer_heads must divide encoder feature dimension")
        pool_heads = heads if series_pool_heads is None else int(series_pool_heads)
        if d % pool_heads != 0:
            raise ValueError("series_pool_heads must divide encoder feature dimension")

        # Construct every parameter shared with B12 in the same order as B12 so
        # the identical seed yields identical shared random initialization.
        self.slice_position = nn.Parameter(torch.randn(self.n_slices, d) * 0.02)
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)

        study_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=int(d * float(transformer_ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            study_layer,
            num_layers=int(transformer_layers),
            norm=nn.LayerNorm(d),
            # norm_first=True already forces this off inside PyTorch; saying so
            # explicitly keeps the behaviour identical and drops the warning.
            enable_nested_tensor=False,
        )

        self.pathology_tokens = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
        pathology_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=int(d * float(transformer_ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pathology_context = nn.TransformerEncoder(
            pathology_layer,
            num_layers=int(pathology_layers),
            norm=nn.LayerNorm(d),
            enable_nested_tensor=False,
        )
        self.cross_attention = nn.MultiheadAttention(
            d,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(float(dropout))
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, d))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.xavier_uniform_(self.target_weight)

        # New B12.1 parameters are deliberately created only after all B12-shared
        # parameters, preventing the added module from shifting their RNG draws.
        self.series_pool = LearnedSeriesPool(d, pool_heads, float(dropout))

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, chunk, use_reentrant=False)
        return self.encoder(chunk)

    def _encode_slices(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        if volumes.ndim != 6:
            raise ValueError("B12.1 model expects [B,K,S,3,H,W]")
        b, k, s, c, h, w = volumes.shape
        if s != self.n_slices or c != self.in_channels:
            raise ValueError("B12.1 slice/channel contract mismatch")
        if present.shape != (b, k):
            raise ValueError("B12.1 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B12.1 series_meta must have shape [B,K,3]")

        flat_series = volumes.reshape(b * k, s, c, h, w)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        d = self.encoder.out_dim
        if active_indices.numel() == 0:
            return volumes.new_zeros((b, k, s, d))

        active = flat_series.index_select(0, active_indices)
        flat_slices = active.reshape(-1, c, h, w)
        encoded = torch.cat(
            [self._encode_chunk(chunk) for chunk in flat_slices.split(self.encoder_batch_size, dim=0)],
            dim=0,
        ).reshape(active.shape[0], s, d)
        all_features = encoded.new_zeros((b * k, s, d)).index_copy(
            0, active_indices, encoded
        )
        features = all_features.reshape(b, k, s, d)

        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        mask = present[:, :, None, None].to(features.dtype)
        return (
            features
            + self.slice_position[None, None, :, :]
            + metadata[:, :, None, :]
        ) * mask

    def _pool_real_series(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B12.1 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))
        active_tokens = self.series_pool(flat.index_select(0, active_indices))
        all_tokens = active_tokens.new_zeros((b * k, d)).index_copy(
            0, active_indices, active_tokens
        )
        return all_tokens.reshape(b, k, d)

    def _study_memory(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slice_features = self._encode_slices(volumes, present, series_meta)
        tokens = self._pool_real_series(slice_features, present)
        padding = present <= 0
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        contextual = self.context(tokens, src_key_padding_mask=safe_padding)
        contextual = contextual.masked_fill(padding[:, :, None], 0.0)
        return contextual, safe_padding, empty

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        memory, padding, empty = self._study_memory(volumes, present, series_meta)
        b = memory.shape[0]
        queries = self.pathology_tokens[None, :, :].expand(b, -1, -1)
        queries = self.pathology_context(queries)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=padding,
            need_weights=False,
        )
        queries = self.dropout(self.query_norm(queries + attended))
        logits = (queries * self.target_weight[None, :, :]).sum(dim=-1) + self.target_bias
        return torch.where(empty[:, None], self.target_bias[None, :], logits)


STUDY_HIERARCHY_GATE_PARAMETERS = LOCAL_CONTEXT_GATE_PARAMETERS


STUDY_HIERARCHY_QUERY_PARAMETERS = LOCAL_CONTEXT_QUERY_PARAMETERS


STUDY_HIERARCHY_NEW_PARAMETERS = LOCAL_CONTEXT_NEW_PARAMETERS


class ComplementarySeriesPoolNet(SeriesHierarchyNet):
    """B20 hierarchy plus one zero-gated complementary learned slice summary."""

    def __init__(self, *args, **kwargs) -> None:
        # Construct the complete historical B20/B12.1 model first so the same
        # construction seed preserves every shared random draw.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.complementary_query = nn.Parameter(torch.randn(d) * 0.02)
        self.complementary_gate = nn.Parameter(torch.zeros(d))

    def effective_complementary_gate(self) -> torch.Tensor:
        return torch.tanh(self.complementary_gate)

    def _complementary_weights(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        """Return deterministic learned softmax weights [N,S] for real series."""
        if active_slice_features.ndim != 3:
            raise ValueError("B29 complementary pool expects [N,S,D]")
        if active_slice_features.shape[-1] != self.encoder.out_dim:
            raise ValueError("B29 complementary pool feature dimension mismatch")
        d = int(active_slice_features.shape[-1])
        query = self.complementary_query.to(dtype=active_slice_features.dtype)
        scores = torch.matmul(active_slice_features, query) / math.sqrt(float(d))
        # Float32 softmax is deterministic and numerically stable under bf16;
        # no dropout is used in this branch, preserving the B20 RNG path.
        return torch.softmax(scores.float(), dim=1).to(dtype=active_slice_features.dtype)

    def _complementary_summary(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        """Build the second learned summary without adding trainable projection layers."""
        weights = self._complementary_weights(active_slice_features)
        summary = torch.sum(weights[:, :, None] * active_slice_features, dim=1)
        d = int(summary.shape[-1])
        # Parameter-free normalisation keeps C on a scale comparable with the
        # LayerNorm-normalised historical B20 token while adding no parameters.
        return F.layer_norm(summary.float(), (d,)).to(dtype=active_slice_features.dtype)

    def _pool_real_series_b29(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B29 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        active_slices = flat.index_select(0, active_indices)

        # Historical B20 series token. This call and its stochastic operations
        # are unchanged and occur before the new deterministic branch.
        active_primary = self.series_pool(active_slices)

        active_complement = self._complementary_summary(active_slices)
        gate = self.effective_complementary_gate().to(dtype=active_primary.dtype)
        active_tokens = active_primary + gate[None, :] * (active_complement - active_primary)

        all_tokens = active_tokens.new_zeros((b * k, d)).index_copy(
            0, active_indices, active_tokens
        )
        return all_tokens.reshape(b, k, d)

    def _study_memory(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slice_features = self._encode_slices(volumes, present, series_meta)
        tokens = self._pool_real_series_b29(slice_features, present)
        padding = present <= 0
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        contextual = self.context(tokens, src_key_padding_mask=safe_padding)
        contextual = contextual.masked_fill(padding[:, :, None], 0.0)
        return contextual, safe_padding, empty

    def complementary_state(self) -> dict:
        gate_raw = self.complementary_gate.detach().float().cpu()
        gate_effective = torch.tanh(gate_raw)
        query = self.complementary_query.detach().float().cpu()
        primary_query = self.series_pool.query.detach().float().cpu().reshape(-1)
        query_cosine = float(
            F.cosine_similarity(query.reshape(1, -1), primary_query.reshape(1, -1), dim=1).item()
        )
        return {
            "version": COMPLEMENTARY_POOL_VERSION,
            "new_parameter_count": int(gate_raw.numel() + query.numel()),
            "query_parameter_count": int(query.numel()),
            "gate_parameter_count": int(gate_raw.numel()),
            "query_max_abs": float(query.abs().max().item()),
            "query_mean_abs": float(query.abs().mean().item()),
            "query_l2": float(torch.linalg.vector_norm(query).item()),
            "query_cosine_to_primary": query_cosine,
            "gate_raw_max_abs": float(gate_raw.abs().max().item()),
            "gate_raw_mean_abs": float(gate_raw.abs().mean().item()),
            "gate_raw_l2": float(torch.linalg.vector_norm(gate_raw).item()),
            "gate_effective_max_abs": float(gate_effective.abs().max().item()),
            "gate_effective_mean_abs": float(gate_effective.abs().mean().item()),
            "gate_effective_l2": float(torch.linalg.vector_norm(gate_effective).item()),
        }


class LocalContextSeriesPoolNet(ComplementarySeriesPoolNet):
    """B29 plus zero-init local through-plane context for attention scoring only."""

    def __init__(self, *args, **kwargs) -> None:
        # B29 (and therefore complete historical B20) is constructed first.
        super().__init__(*args, **kwargs)
        d = int(self.encoder.out_dim)
        self.local_context = nn.Conv1d(
            d,
            d,
            kernel_size=3,
            padding=1,
            groups=d,
            bias=False,
        )
        nn.init.zeros_(self.local_context.weight)
        self._attention_audit_enabled = False
        self._attention_audit_accum = None
        self._require_b31_contract()

    def _require_b31_contract(self) -> None:
        d = int(self.encoder.out_dim)
        if self.local_context.groups != d:
            raise ValueError("B31 local context must remain depthwise")
        if tuple(self.local_context.kernel_size) != (3,):
            raise ValueError("B31 local context kernel must remain 3")
        if self.local_context.bias is not None:
            raise ValueError("B31 local context must remain bias-free")
        if int(self.local_context.weight.numel()) != LOCAL_CONTEXT_PARAMETERS:
            raise ValueError("B31 local context parameter count changed")

    def _contextualized_slice_features(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        if active_slice_features.ndim != 3:
            raise ValueError("B31 local context expects [N,S,D]")
        d = int(active_slice_features.shape[-1])
        if d != int(self.encoder.out_dim):
            raise ValueError("B31 local context feature dimension mismatch")
        normalized = F.layer_norm(active_slice_features.float(), (d,)).to(
            dtype=active_slice_features.dtype
        )
        delta = self.local_context(normalized.transpose(1, 2)).transpose(1, 2)
        return active_slice_features + delta

    def _contextual_weights(self, active_slice_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contextual = self._contextualized_slice_features(active_slice_features)
        weights = self._complementary_weights(contextual)
        return weights, contextual

    def _contextual_complementary_summary(
        self, active_slice_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, contextual = self._contextual_weights(active_slice_features)
        # Values remain the ORIGINAL B20 slice tokens; context only changes scores.
        summary = torch.sum(weights[:, :, None] * active_slice_features, dim=1)
        d = int(summary.shape[-1])
        summary = F.layer_norm(summary.float(), (d,)).to(dtype=active_slice_features.dtype)
        return summary, weights, contextual

    def enable_attention_audit(self, enabled: bool = True, *, reset: bool = False) -> None:
        self._attention_audit_enabled = bool(enabled)
        if reset:
            self._attention_audit_accum = None

    @torch.no_grad()
    def _update_attention_audit(
        self,
        active_slice_features: torch.Tensor,
        context_weights: torch.Tensor,
        contextual_features: torch.Tensor,
        primary_tokens: torch.Tensor,
        complementary_tokens: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        raw_weights = self._complementary_weights(active_slice_features).float().clamp_min(1e-12)
        context_weights = context_weights.float().clamp_min(1e-12)
        s = int(raw_weights.shape[-1])
        log_s = math.log(float(max(s, 2)))

        raw_entropy = -(raw_weights * raw_weights.log()).sum(dim=-1) / log_s
        context_entropy = -(context_weights * context_weights.log()).sum(dim=-1) / log_s
        m = 0.5 * (raw_weights + context_weights)
        js = 0.5 * (
            (raw_weights * (raw_weights / m).log()).sum(dim=-1)
            + (context_weights * (context_weights / m).log()).sum(dim=-1)
        ) / math.log(2.0)

        top1 = (raw_weights.argmax(dim=-1) == context_weights.argmax(dim=-1)).float()
        topk = min(3, s)
        raw_top = torch.topk(raw_weights, k=topk, dim=-1).indices
        ctx_top = torch.topk(context_weights, k=topk, dim=-1).indices
        overlap = (raw_top[:, :, None] == ctx_top[:, None, :]).any(dim=-1).float().sum(dim=-1)
        overlap = overlap / float(topk)

        if s > 1:
            raw_adj = torch.abs(raw_weights[:, 1:] - raw_weights[:, :-1]).mean(dim=-1)
            ctx_adj = torch.abs(context_weights[:, 1:] - context_weights[:, :-1]).mean(dim=-1)
        else:
            raw_adj = torch.zeros_like(raw_entropy)
            ctx_adj = torch.zeros_like(context_entropy)

        context_delta = contextual_features - active_slice_features
        context_delta_ratio = torch.linalg.vector_norm(context_delta.float(), dim=(-2, -1)) / (
            torch.linalg.vector_norm(active_slice_features.float(), dim=(-2, -1)) + 1e-12
        )
        residual = gate[None, :] * (complementary_tokens - primary_tokens)
        residual_ratio = torch.linalg.vector_norm(residual.float(), dim=-1) / (
            torch.linalg.vector_norm(primary_tokens.float(), dim=-1) + 1e-12
        )

        count = torch.tensor(float(raw_weights.shape[0]), device=raw_weights.device)
        sums = torch.stack(
            [
                count,
                raw_entropy.sum(),
                context_entropy.sum(),
                js.sum(),
                top1.sum(),
                overlap.sum(),
                raw_adj.sum(),
                ctx_adj.sum(),
                context_delta_ratio.sum(),
                context_delta_ratio.max(),
                residual_ratio.sum(),
                residual_ratio.max(),
            ]
        ).detach()
        if self._attention_audit_accum is None:
            self._attention_audit_accum = sums.clone()
        else:
            self._attention_audit_accum[:9] = self._attention_audit_accum[:9] + sums[:9]
            self._attention_audit_accum[9] = torch.maximum(self._attention_audit_accum[9], sums[9])
            self._attention_audit_accum[10] = self._attention_audit_accum[10] + sums[10]
            self._attention_audit_accum[11] = torch.maximum(self._attention_audit_accum[11], sums[11])

    def attention_audit_state(self, *, reset: bool = False) -> dict:
        acc = self._attention_audit_accum
        if acc is None:
            state = {
                "series_count": 0,
                "raw_b29_attention_entropy_normalized_mean": None,
                "context_attention_entropy_normalized_mean": None,
                "raw_vs_context_js_divergence_normalized_mean": None,
                "raw_vs_context_top1_agreement": None,
                "raw_vs_context_top3_overlap_fraction_mean": None,
                "raw_attention_adjacent_absdiff_mean": None,
                "context_attention_adjacent_absdiff_mean": None,
                "context_delta_norm_ratio_mean": None,
                "context_delta_norm_ratio_max": None,
                "effective_residual_norm_ratio_mean": None,
                "effective_residual_norm_ratio_max": None,
            }
        else:
            x = acc.detach().float().cpu()
            count = max(float(x[0].item()), 1.0)
            state = {
                "series_count": int(round(float(x[0].item()))),
                "raw_b29_attention_entropy_normalized_mean": float(x[1].item() / count),
                "context_attention_entropy_normalized_mean": float(x[2].item() / count),
                "raw_vs_context_js_divergence_normalized_mean": float(x[3].item() / count),
                "raw_vs_context_top1_agreement": float(x[4].item() / count),
                "raw_vs_context_top3_overlap_fraction_mean": float(x[5].item() / count),
                "raw_attention_adjacent_absdiff_mean": float(x[6].item() / count),
                "context_attention_adjacent_absdiff_mean": float(x[7].item() / count),
                "context_delta_norm_ratio_mean": float(x[8].item() / count),
                "context_delta_norm_ratio_max": float(x[9].item()),
                "effective_residual_norm_ratio_mean": float(x[10].item() / count),
                "effective_residual_norm_ratio_max": float(x[11].item()),
            }
        if reset:
            self._attention_audit_accum = None
        return state

    def local_context_state(self) -> dict:
        w = self.local_context.weight.detach().float().cpu()
        return {
            "version": LOCAL_CONTEXT_VERSION,
            "parameter_count": int(w.numel()),
            "kernel_size": 3,
            "depthwise_groups": int(self.local_context.groups),
            "bias": False,
            "weight_max_abs": float(w.abs().max().item()),
            "weight_mean_abs": float(w.abs().mean().item()),
            "weight_l2": float(torch.linalg.vector_norm(w).item()),
        }

    def _pool_real_series_b31(
        self,
        slice_features: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if slice_features.ndim != 4:
            raise ValueError("B31 slice features must be [B,K,S,D]")
        b, k, s, d = slice_features.shape
        flat = slice_features.reshape(b * k, s, d)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return slice_features.new_zeros((b, k, d))

        active_slices = flat.index_select(0, active_indices)
        active_primary = self.series_pool(active_slices)
        active_complement, context_weights, contextual = self._contextual_complementary_summary(
            active_slices
        )
        gate = self.effective_complementary_gate().to(dtype=active_primary.dtype)
        active_tokens = active_primary + gate[None, :] * (active_complement - active_primary)

        if self._attention_audit_enabled:
            self._update_attention_audit(
                active_slices,
                context_weights,
                contextual,
                active_primary,
                active_complement,
                gate,
            )

        all_tokens = active_tokens.new_zeros((b * k, d)).index_copy(0, active_indices, active_tokens)
        return all_tokens.reshape(b, k, d)

    def _study_memory(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slice_features = self._encode_slices(volumes, present, series_meta)
        tokens = self._pool_real_series_b31(slice_features, present)
        padding = present <= 0
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        contextual = self.context(tokens, src_key_padding_mask=safe_padding)
        contextual = contextual.masked_fill(padding[:, :, None], 0.0)
        return contextual, safe_padding, empty

    def b31_state(self) -> dict:
        return {
            "complementary_pool": self.complementary_state(),
            "local_context": self.local_context_state(),
        }


class StudyHierarchyNet(LocalContextSeriesPoolNet):
    """B31 training path with exact local-context bypass under ``eval()``."""

    def _contextualized_slice_features(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Exact inference contract: trained local-context weights are not read.
            return active_slice_features
        return super()._contextualized_slice_features(active_slice_features)

    def b34_state(self) -> dict:
        return {
            "scaffold_version": STUDY_HIERARCHY_VERSION,
            "training_context_active": bool(self.training),
            "eval_context_exact_bypass": True,
            "inference_context_parameters_used": 0,
            "complementary_pool": self.complementary_state(),
            "local_context_scaffold": self.local_context_state(),
        }


def build_study_hierarchy(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> StudyHierarchyNet:
    if spec.get("architecture") != STUDY_HIERARCHY_ARCHITECTURE:
        raise ValueError("not a B34 training-only context-scaffold model spec")
    if spec.get("aggregation") != STUDY_HIERARCHY_AGGREGATION:
        raise ValueError("B34 aggregation policy mismatch")
    if spec.get("b34_scaffold_version") != STUDY_HIERARCHY_VERSION:
        raise ValueError("B34 scaffold version mismatch")
    if int(spec.get("b34_new_parameter_count", -1)) != STUDY_HIERARCHY_NEW_PARAMETERS:
        raise ValueError("B34 new-parameter count mismatch")
    if int(spec.get("b31_context_parameter_count", -1)) != STUDY_HIERARCHY_CONTEXT_PARAMETERS:
        raise ValueError("B34 inherited context parameter count mismatch")
    if spec.get("b31_context_version") != LOCAL_CONTEXT_VERSION:
        raise ValueError("B34 inherited B31 context version mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = StudyHierarchyNet(
        int(spec["n_slices"]),
        in_channels=int(spec.get("in_channels", 3)),
        pretrained_weights=bool(pretrained_weights),
        normalize_input=bool(spec["normalize_input"]),
        dropout=float(spec["dropout"]),
        encoder_batch_size=int(spec["encoder_batch_size"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
        transformer_ff_mult=float(spec["transformer_ff_mult"]),
        pathology_layers=int(spec["pathology_layers"]),
        series_pool_heads=int(spec["series_pool_heads"]),
    )
    if int(model.complementary_query.numel()) != STUDY_HIERARCHY_QUERY_PARAMETERS:
        raise ValueError("B34 query dimension changed")
    if int(model.complementary_gate.numel()) != STUDY_HIERARCHY_GATE_PARAMETERS:
        raise ValueError("B34 gate dimension changed")
    if int(model.local_context.weight.numel()) != STUDY_HIERARCHY_CONTEXT_PARAMETERS:
        raise ValueError("B34 scaffold dimension changed")
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model

