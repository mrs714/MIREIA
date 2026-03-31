from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpFuseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class MireiaRoadSegmentationModel(nn.Module):
    """Lightweight multitask segmentation with lane and road output heads."""

    def __init__(
        self,
        dropout: float = 0.1,
        input_size: tuple[int, int] = (256, 256),
        backbone_weights: MobileNet_V3_Small_Weights | None = MobileNet_V3_Small_Weights.DEFAULT,
    ):
        super().__init__()

        self.num_classes = 1
        self.output_heads = ("lane", "road")
        self.input_size = input_size

        backbone = mobilenet_v3_small(weights=backbone_weights)
        features = backbone.features

        # Encoder taps at strides x2, x4, x8 and x16.
        self.stage1 = nn.Sequential(*features[:2])
        self.stage2 = nn.Sequential(*features[2:4])
        self.stage3 = nn.Sequential(*features[4:9])
        self.stage4 = nn.Sequential(*features[9:13])

        self.up3 = UpFuseBlock(576, 48, 160, dropout=dropout)
        self.up2 = UpFuseBlock(160, 24, 96, dropout=dropout)
        self.up1 = UpFuseBlock(96, 16, 64, dropout=dropout)

        self.shared_head = ConvBlock(64, 32, dropout=dropout)
        self.lane_head = nn.Conv2d(32, 1, kernel_size=1)
        self.road_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("Expected input tensor shape (B, C, H, W)")

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        d3 = self.up3(s4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        shared = self.shared_head(d1)
        lane_logits = self.lane_head(shared)
        road_logits = self.road_head(shared)

        lane_logits = F.interpolate(
            lane_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        road_logits = F.interpolate(
            road_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return lane_logits, road_logits


__all__ = ["MireiaRoadSegmentationModel"]
