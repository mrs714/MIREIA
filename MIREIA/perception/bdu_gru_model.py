from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from MIREIA.perception.e2e_model import (
    TemporalBDUGRU,
    TemporalBDUGRUSequence,
    RiskMLP,
    TimeDistributedMLP,
)


@dataclass(frozen=True)
class BDUGRUModelConfig:
    feature_dim: int = 32
    gru_hidden_dim: int = 256
    gru_num_layers: int = 2
    gru_dropout: float = 0.1
    mlp_dropout: float = 0.3
    use_sigmoid: bool = False


class BDUGRURiskPredictor(nn.Module):
    """Many-to-one BDU-GRU risk predictor over feature vectors."""

    def __init__(self, config: BDUGRUModelConfig | None = None):
        super().__init__()
        if config is None:
            config = BDUGRUModelConfig()
        self.config = config
        self.temporal_bdugru = TemporalBDUGRU(
            input_dim=config.feature_dim,
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
        if x.ndim != 3:
            raise ValueError("Expected input shape (B, N, D)")
        if x.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.config.feature_dim}, got {x.shape[-1]}"
            )
        temporal = self.temporal_bdugru(x)
        return self.regression_head(temporal)


class Seq2SeqBDUGRURiskPredictor(nn.Module):
    """Many-to-many BDU-GRU risk predictor over feature vectors."""

    def __init__(self, config: BDUGRUModelConfig | None = None):
        super().__init__()
        if config is None:
            config = BDUGRUModelConfig()
        self.config = config
        self.temporal_bdugru = TemporalBDUGRUSequence(
            input_dim=config.feature_dim,
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
        if x.ndim != 3:
            raise ValueError("Expected input shape (B, N, D)")
        if x.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.config.feature_dim}, got {x.shape[-1]}"
            )

        num_frames = x.shape[1]
        if m_eval_frames <= 0 or m_eval_frames > num_frames:
            raise ValueError("m_eval_frames must be between 1 and N")

        temporal = self.temporal_bdugru(x)
        eval_seq = temporal[:, -m_eval_frames:, :]
        return self.regression_head(eval_seq)


__all__ = [
    "BDUGRUModelConfig",
    "BDUGRURiskPredictor",
    "Seq2SeqBDUGRURiskPredictor",
]
