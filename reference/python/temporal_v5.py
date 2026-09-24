"""Temporal-V5 residual expert.

This is the public inference-side portion of the Temporal-V5 branch. It keeps
the causal multi-scale residual path and observability gate visible. The
private repository supplies the frozen world-model token producer, checkpoint
loading, teacher losses, and deployment-specific event head.

Expected shapes:
    current, future: [batch, steps, dim]
    pose:            [batch, frames, keypoints, 3] (x, y, visibility)
    boxes:           [batch, frames, >=4] (x, y, w, h, ...optional quality)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TemporalV5Config:
    dim: int = 256
    attention_heads: int = 4
    residual_limit: float = 0.20
    dilations: tuple[int, ...] = (1, 2, 4)


class CausalDepthwiseBlock(nn.Module):
    """Left-padded depthwise temporal convolution; no future leakage."""

    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv1d(dim, dim, 3, groups=dim, dilation=dilation, bias=False)
        self.pointwise = nn.Conv1d(dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(16, dim)

    def forward(self, sequence: Tensor) -> Tensor:
        channels_first = sequence.transpose(1, 2)
        padded = F.pad(channels_first, (2 * self.dilation, 0))
        output = self.pointwise(self.depthwise(padded))
        return F.gelu(self.norm(output)).transpose(1, 2)


class TemporalV5ResidualExpert(nn.Module):
    """Bounded, observability-aware residual for current/future tokens.

    The residual starts as an identity mapping. This mirrors the V5 safety
    property: a newly attached expert cannot silently invalidate the frozen
    baseline before it has been validated.
    """

    def __init__(self, config: TemporalV5Config | None = None) -> None:
        super().__init__()
        self.config = config or TemporalV5Config()
        dim = self.config.dim
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(dim, dilation) for dilation in self.config.dilations
        )
        self.mix = nn.Sequential(
            nn.Linear(dim * len(self.config.dilations), dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.future_attention = nn.MultiheadAttention(
            dim, self.config.attention_heads, dropout=0.05, batch_first=True
        )
        self.future_norm = nn.LayerNorm(dim)
        self.current_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.future_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.observability = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        self._identity_initialization()

    def _identity_initialization(self) -> None:
        # The candidate initially reproduces V5 exactly; validation earns
        # every deviation through the private distillation/event objectives.
        nn.init.zeros_(self.current_head[-1].weight)
        nn.init.zeros_(self.current_head[-1].bias)
        nn.init.zeros_(self.future_head[-1].weight)
        nn.init.zeros_(self.future_head[-1].bias)
        nn.init.zeros_(self.observability[-1].weight)
        nn.init.constant_(self.observability[-1].bias, 2.0)

    @staticmethod
    def observability_features(pose: Tensor, boxes: Tensor, steps: int) -> Tensor:
        indices = torch.linspace(0, pose.size(1) - 1, steps, device=pose.device).round().long()
        selected_pose = pose[:, indices].float()
        selected_boxes = boxes[:, indices].float()
        visible = selected_pose[..., 2].clamp(0.0, 1.0).mean((1, 2))
        confidence = (
            selected_boxes[..., 4].clamp(0.0, 1.0).mean(1)
            if selected_boxes.size(-1) > 4
            else visible
        )
        valid = (
            selected_boxes[..., 5].clamp(0.0, 1.0).mean(1)
            if selected_boxes.size(-1) > 5
            else visible
        )
        center = selected_boxes[..., :2]
        jitter = torch.diff(center, dim=1).norm(dim=-1).median(dim=1).values
        stability = torch.exp(-8.0 * jitter).clamp(0.0, 1.0)
        return torch.stack((visible, confidence, valid, stability), dim=1)

    def forward(self, current: Tensor, future: Tensor, pose: Tensor, boxes: Tensor):
        normalized = self.input_norm(current)
        multi_scale = [block(normalized) for block in self.blocks]
        memory = self.mix(torch.cat(multi_scale, dim=-1))

        observability = self.observability_features(pose, boxes, current.size(1))
        reliability = torch.sigmoid(self.observability(observability)).unsqueeze(1)
        current_delta = (
            self.config.residual_limit
            * torch.tanh(self.current_head(memory))
            * reliability
        )

        attended, _ = self.future_attention(future, memory, memory, need_weights=False)
        future_context = self.future_norm(future + attended)
        future_delta = (
            self.config.residual_limit
            * torch.tanh(self.future_head(torch.cat((future_context, future), dim=-1)))
            * reliability
        )
        return current + current_delta, future + future_delta, {
            "residual_reliability": reliability.squeeze(1),
            "observability": observability,
            "current_residual": current_delta,
            "future_residual": future_delta,
        }


def apply_temporal_v5(
    expert: TemporalV5ResidualExpert,
    current: Tensor,
    future: Tensor,
    pose: Tensor,
    boxes: Tensor,
) -> dict[str, Tensor]:
    """Public adapter for a frozen-token producer and a downstream event head."""

    refined_current, refined_future, diagnostics = expert(current, future, pose, boxes)
    return {
        "current_tokens": refined_current,
        "future_tokens": refined_future,
        **diagnostics,
    }
