"""Target-specific top-k pooling over a grid of local evidence."""
from __future__ import annotations

from torch import nn
import math
import torch
import torch.nn.functional as F

from ..constants import N_TARGETS
from ..data.slice_selection import DEFAULT_GRID_SIZE, POSITION_BASIS, TOKEN_DROPOUT, _position_basis


SPARSE_HEAD_VERSION = "b36_pathology_sparse_topk_mil_residual_v1"


DEFAULT_TOP_K = 8


DEFAULT_TEMPERATURE = 1.0


class SparseEvidenceHead(nn.Module):
    """Pathology-specific local evidence scorer with sparse top-k MIL pooling."""

    def __init__(
        self,
        dim: int = 768,
        *,
        grid_size: int = DEFAULT_GRID_SIZE,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = DEFAULT_TEMPERATURE,
        token_dropout: float = TOKEN_DROPOUT,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.n_regions = self.grid_size * self.grid_size
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.top_k < 1:
            raise ValueError("B36 top_k must be >=1")
        if self.temperature <= 0:
            raise ValueError("B36 temperature must be positive")

        self.token_dropout = nn.Dropout(float(token_dropout))
        self.position_projection = nn.Linear(POSITION_BASIS, self.dim, bias=False)
        self.region_embedding = nn.Parameter(torch.zeros(self.n_regions, self.dim))
        self.plane_embedding = nn.Embedding(4, self.dim, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, self.dim, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, self.dim, padding_idx=0)

        # These are evidence classifiers, not attention queries.  Their dot
        # products are token-level logits and are intentionally not divided by
        # sqrt(D), which would make the initial evidence surface nearly flat.
        self.evidence_weight = nn.Parameter(torch.empty(N_TARGETS, self.dim))
        self.evidence_bias = nn.Parameter(torch.zeros(N_TARGETS))
        self.gate = nn.Parameter(torch.zeros(N_TARGETS))

        nn.init.zeros_(self.position_projection.weight)
        nn.init.zeros_(self.plane_embedding.weight)
        nn.init.zeros_(self.fluid_embedding.weight)
        nn.init.zeros_(self.fat_embedding.weight)
        nn.init.normal_(self.evidence_weight, mean=0.0, std=0.02)

    def effective_gate(self) -> torch.Tensor:
        return torch.tanh(self.gate)

    def _tokens(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial.ndim != 5:
            raise ValueError("B36 spatial features must be [B,K,S,R,D]")
        b, k, s, r, d = spatial.shape
        if d != self.dim or r != self.n_regions:
            raise ValueError("B36 spatial feature shape does not match the head")
        if present.shape != (b, k):
            raise ValueError("B36 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B36 series metadata shape mismatch")
        if slice_position.shape != (b, k, s):
            raise ValueError("B36 slice-position shape mismatch")

        # Parameter-free LN keeps the pretrained local representation on a
        # stable per-token scale before pathology-specific evidence scoring.
        x = F.layer_norm(spatial.float(), (d,)).to(dtype=spatial.dtype)
        pos = self.position_projection(_position_basis(slice_position).to(x.device)).to(
            dtype=x.dtype
        )
        region = self.region_embedding.to(dtype=x.dtype)[None, None, None, :, :]
        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = (plane + fluid + fat).to(dtype=x.dtype)
        x = x + pos[:, :, :, None, :] + region + metadata[:, :, None, None, :]
        x = self.token_dropout(x)
        tokens = x.reshape(b, k * s * r, d)
        invalid = (
            (present <= 0)[:, :, None, None]
            .expand(b, k, s, r)
            .reshape(b, k * s * r)
        )
        if invalid.all(dim=1).any():
            raise RuntimeError("B36 received a study with no readable MRI series")
        if int((~invalid).sum(dim=1).min().item()) < self.top_k:
            raise RuntimeError("B36 has fewer valid local tokens than top_k")
        return tokens, invalid

    def forward(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, invalid = self._tokens(
            spatial,
            present,
            series_meta,
            slice_position,
        )
        # [B,T,N] token-level pathology evidence logits.
        score = torch.einsum(
            "bnd,td->btn",
            tokens,
            self.evidence_weight.to(dtype=tokens.dtype),
        ) + self.evidence_bias.to(dtype=tokens.dtype)[None, :, None]
        score = score.masked_fill(invalid[:, None, :], float("-inf"))

        top_values, top_indices = torch.topk(
            score,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        # Smooth max over only the explicitly selected evidence instances.
        # Subtracting log(k) keeps the pooled logit on the token-logit scale.
        tau = float(self.temperature)
        local_logits = tau * (
            torch.logsumexp(top_values.float() / tau, dim=-1)
            - math.log(float(self.top_k))
        )
        return local_logits, top_indices, top_values.float()

    def state(self) -> dict:
        raw = self.gate.detach().float().cpu()
        effective = torch.tanh(raw)
        return {
            "version": SPARSE_HEAD_VERSION,
            "grid_size": self.grid_size,
            "regions_per_slice": self.n_regions,
            "feature_dim": self.dim,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "gate_raw": [float(x) for x in raw.tolist()],
            "gate_effective": [float(x) for x in effective.tolist()],
            "gate_effective_abs_mean": float(effective.abs().mean().item()),
            "gate_effective_abs_max": float(effective.abs().max().item()),
        }

