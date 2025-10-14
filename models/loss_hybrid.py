"""Hybrid fusion loss for HBANet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridLossWeights:
    structure: float = 1.0
    intensity: float = 10.0
    variation: float = 0.5


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    grad_x = x[..., 1:] - x[..., :-1]
    grad_y = x[..., 1:, :] - x[..., :-1, :]
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_x = torch.abs(x[..., 1:] - x[..., :-1]).mean()
    tv_y = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
    return tv_x + tv_y


class HybridFusionLoss(nn.Module):
    """Hybrid loss combining structure, intensity, and variation terms."""

    def __init__(self, weights: HybridLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or HybridLossWeights()

    @staticmethod
    def _to_grayscale(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        if x.shape[1] >= 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        return x.mean(dim=1, keepdim=True)

    def forward(self, ir: torch.Tensor, vis: torch.Tensor, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ir_gray = self._to_grayscale(ir)
        vis_gray = self._to_grayscale(vis)
        fused_gray = self._to_grayscale(fused)

        grad_ir = _gradient_magnitude(ir_gray)
        grad_vis = _gradient_magnitude(vis_gray)
        grad_fused = _gradient_magnitude(fused_gray)

        target_grad = torch.maximum(grad_ir, grad_vis)
        structure_loss = F.l1_loss(grad_fused, target_grad)

        target_intensity = torch.maximum(ir_gray, vis_gray)
        intensity_loss = F.l1_loss(fused_gray, target_intensity)

        variation_loss = _total_variation(fused_gray)

        total_loss = (
            self.weights.structure * structure_loss
            + self.weights.intensity * intensity_loss
            + self.weights.variation * variation_loss
        )
        return total_loss, structure_loss, intensity_loss, variation_loss
