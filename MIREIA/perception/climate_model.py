from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MireiaEnvironmentClassifier(nn.Module):
    """Lightweight multi-task classifier for day/night and weather climate."""

    def __init__(
        self,
        num_weather_classes: int = 5,
        dropout: float = 0.2,
        input_size: tuple[int, int] = (512, 512),
        backbone_weights: MobileNet_V3_Small_Weights | None = MobileNet_V3_Small_Weights.DEFAULT,
    ):
        super().__init__()

        if num_weather_classes <= 0:
            raise ValueError("num_weather_classes must be > 0")

        self.num_weather_classes = int(num_weather_classes)
        self.input_size = input_size

        backbone = mobilenet_v3_small(weights=backbone_weights)
        self.feature_extractor = backbone.features
        self.avgpool = backbone.avgpool

        # MobileNetV3-Small global pooled feature width.
        input_dim = 576

        self.day_night_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        self.weather_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_weather_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        day_night_logits = self.day_night_head(features)
        weather_logits = self.weather_head(features)
        return day_night_logits, weather_logits