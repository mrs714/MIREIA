from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class E2EModelConfig:
    input_size: Tuple[int, int] = (512, 512)
    spatial_channels: int = 3
    gru_hidden_dim: int = 256
    gru_num_layers: int = 2
    gru_dropout: float = 0.1
    mlp_dropout: float = 0.3
    use_sigmoid: bool = False


class SpatialBackbone(nn.Module):
    def __init__(self, input_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class TemporalBDUGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.residual_proj = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        gated = x * gate
        out, h_n = self.gru(gated)
        residual = self.residual_proj(x)
        out = out + residual
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        return torch.cat([forward_last, backward_last], dim=1)


class TemporalBDUGRUSequence(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.residual_proj = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        gated = x * gate
        out, _ = self.gru(gated)
        residual = self.residual_proj(x)
        return out + residual


class RiskMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3, use_sigmoid: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


class TimeDistributedMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3, use_sigmoid: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


class E2ERiskPredictor(nn.Module):
    def __init__(self, config: E2EModelConfig | None = None):
        super().__init__()
        if config is None:
            config = E2EModelConfig()
        self.config = config
        self.spatial_backbone = SpatialBackbone(input_size=config.input_size)
        self.temporal_bdugru = TemporalBDUGRU(
            input_dim=self.spatial_backbone.feature_dim,
            hidden_dim=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            dropout=config.gru_dropout,
        )
        self.regression_head = RiskMLP(
            input_dim=config.gru_hidden_dim * 2,
            dropout=config.mlp_dropout,
            use_sigmoid=config.use_sigmoid,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected input shape (B, N, C, H, W)")
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.spatial_backbone(x)
        features = features.view(batch_size, num_frames, -1)
        temporal = self.temporal_bdugru(features)
        return self.regression_head(temporal)


class Seq2SeqRiskPredictor(nn.Module):
    def __init__(self, config: E2EModelConfig | None = None):
        super().__init__()
        if config is None:
            config = E2EModelConfig()
        self.config = config
        self.spatial_backbone = SpatialBackbone(input_size=config.input_size)
        self.temporal_bdugru = TemporalBDUGRUSequence(
            input_dim=self.spatial_backbone.feature_dim,
            hidden_dim=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            dropout=config.gru_dropout,
        )
        self.regression_head = TimeDistributedMLP(
            input_dim=config.gru_hidden_dim * 2,
            dropout=config.mlp_dropout,
            use_sigmoid=config.use_sigmoid,
        )

    def forward(self, x: torch.Tensor, m_eval_frames: int = 5) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected input shape (B, N, C, H, W)")
        batch_size, num_frames, channels, height, width = x.shape
        if m_eval_frames <= 0 or m_eval_frames > num_frames:
            raise ValueError("m_eval_frames must be between 1 and N")

        x = x.view(batch_size * num_frames, channels, height, width)
        spatial_feats = self.spatial_backbone(x)
        spatial_seq = spatial_feats.view(batch_size, num_frames, -1)
        rnn_out = self.temporal_bdugru(spatial_seq)
        eval_seq = rnn_out[:, -m_eval_frames:, :]
        return self.regression_head(eval_seq)
