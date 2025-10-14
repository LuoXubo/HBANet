"""
Hybrid Boundary-Aware Attention Network (HBANet)
------------------------------------------------
Aligned with the architecture defined in `details.md`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """Helper block: Conv2d + BatchNorm2d + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Simple residual block with two Conv-BN layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.act(out)


class SharedEncoder(nn.Module):
    """Shared CNN encoder for IR and VIS branches."""

    def __init__(self, in_channels: int, base_channels: int, num_res_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvBNReLU(in_channels, base_channels)]
        layers.append(ConvBNReLU(base_channels, base_channels))
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(base_channels))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SobelBoundary(nn.Module):
    """Compute normalized boundary prior using Sobel gradients."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("weight_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("weight_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            luminance = x
        elif x.shape[1] >= 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            luminance = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            luminance = x.mean(dim=1, keepdim=True)

        grad_x = F.conv2d(luminance, self.weight_x, padding=1)
        grad_y = F.conv2d(luminance, self.weight_y, padding=1)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        normalized = magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return normalized


class BoundaryAwareAttentionUnit(nn.Module):
    """Spatial attention guided by boundary prior."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        boundary = F.interpolate(boundary, size=features.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.refine(torch.cat([features, boundary], dim=1))
        return features * attention


class CrossDomainAttentionUnit(nn.Module):
    """Cross attention between IR and VIS feature tokens."""

    def __init__(self, channels: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_ir = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_vis = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)

        self.norm_ir1 = nn.LayerNorm(channels)
        self.norm_ir2 = nn.LayerNorm(channels)
        self.norm_vis1 = nn.LayerNorm(channels)
        self.norm_vis2 = nn.LayerNorm(channels)

        hidden = channels * 2
        self.ffn_ir = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.ffn_vis = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    @staticmethod
    def _flatten(features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, c, h, w = features.shape
        seq = features.flatten(2).transpose(1, 2)
        return seq, (h, w)

    @staticmethod
    def _reshape(sequence: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        b, hw, c = sequence.shape
        h, w = size
        return sequence.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_ir, spatial_size = self._flatten(feat_ir)
        seq_vis, _ = self._flatten(feat_vis)

        cross_ir, _ = self.attn_ir(seq_ir, seq_vis, seq_vis)
        seq_ir = self.norm_ir1(seq_ir + cross_ir)
        seq_ir = self.norm_ir2(seq_ir + self.ffn_ir(seq_ir))

        cross_vis, _ = self.attn_vis(seq_vis, seq_ir, seq_ir)
        seq_vis = self.norm_vis1(seq_vis + cross_vis)
        seq_vis = self.norm_vis2(seq_vis + self.ffn_vis(seq_vis))

        fused_ir = self._reshape(seq_ir, spatial_size)
        fused_vis = self._reshape(seq_vis, spatial_size)
        return fused_ir, fused_vis


class HybridBoundaryAwareAttention(nn.Module):
    """Combine BAAU and CDAU into a hybrid fusion module."""

    def __init__(self, channels: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.boundary_extractor = SobelBoundary()
        self.baau = BoundaryAwareAttentionUnit(channels)
        self.cadau = CrossDomainAttentionUnit(channels, num_heads, dropout)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor, vis_image: torch.Tensor) -> torch.Tensor:
        boundary_prior = self.boundary_extractor(vis_image)
        boundary_enhanced = self.baau(feat_vis, boundary_prior)

        cross_ir, cross_vis = self.cadau(feat_ir, feat_vis)
        cross_fused = 0.5 * (cross_ir + cross_vis)

        fused = torch.cat([boundary_enhanced, cross_fused], dim=1)
        return self.fusion(fused)


class Decoder(nn.Module):
    """Lightweight reconstruction decoder."""

    def __init__(self, channels: int, out_channels: int, num_res_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [ResidualBlock(channels) for _ in range(num_res_blocks)]
        layers.extend(
            [
                ConvBNReLU(channels, channels // 2),
                nn.Conv2d(channels // 2, out_channels, kernel_size=1, bias=True),
            ]
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class HBANet(nn.Module):
    """Hybrid Boundary-Aware Attention Network."""

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int | None = None,
        base_channels: int = 64,
        encoder_res_blocks: int = 4,
        decoder_res_blocks: int = 2,
        num_heads: int = 4,
        img_range: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if out_chans is None:
            out_chans = in_chans if in_chans == 1 else 1

        base_channels = kwargs.get("base_channels", kwargs.get("embed_dim", base_channels))
        num_heads_value = kwargs.get("num_heads", num_heads)
        if isinstance(num_heads_value, (list, tuple)):
            num_heads = num_heads_value[0]
        else:
            num_heads = num_heads_value

        encoder_depths = kwargs.get("depths")
        if isinstance(encoder_depths, (list, tuple)) and encoder_depths:
            encoder_res_blocks = encoder_depths[0]
        encoder_res_blocks = kwargs.get("encoder_res_blocks", encoder_res_blocks)
        decoder_res_blocks = kwargs.get("decoder_res_blocks", decoder_res_blocks)

        self.encoder = SharedEncoder(in_chans, base_channels, encoder_res_blocks)
        self.hba_module = HybridBoundaryAwareAttention(base_channels, num_heads)
        self.decoder = Decoder(base_channels, out_chans, decoder_res_blocks)

        self.img_range = img_range
        self.register_buffer("_range_tensor", torch.tensor(img_range, dtype=torch.float32))

    def forward(self, ir_image: torch.Tensor, vis_image: torch.Tensor) -> torch.Tensor:
        feat_ir = self.encoder(ir_image)
        feat_vis = self.encoder(vis_image)
        fused_features = self.hba_module(feat_ir, feat_vis, vis_image)
        fused_image = self.decoder(fused_features)
        return torch.clamp(fused_image, 0.0, self._range_tensor.item())


if __name__ == "__main__":
    model = HBANet(in_chans=1, base_channels=64, num_heads=4)
    ir = torch.rand(1, 1, 256, 256)
    vis = torch.rand(1, 1, 256, 256)
    out = model(ir, vis)
    print("Output shape:", out.shape)
