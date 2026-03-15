from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainMetrics:
    loss: float


class E2ETrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        grad_clip: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion or nn.MSELoss()
        self.grad_clip = grad_clip

    def train_one_epoch(self, dataloader: DataLoader) -> TrainMetrics:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y)
            loss.backward()

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        mean_loss = total_loss / max(1, total_samples)
        return TrainMetrics(loss=mean_loss)

    def evaluate(self, dataloader: DataLoader) -> TrainMetrics:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

        mean_loss = total_loss / max(1, total_samples)
        return TrainMetrics(loss=mean_loss)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> Dict[str, Iterable[float]]:
        history: Dict[str, list[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        for _ in range(epochs):
            train_metrics = self.train_one_epoch(train_loader)
            history["train_loss"].append(train_metrics.loss)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics.loss)

        return history
