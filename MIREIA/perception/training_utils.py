from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from MIREIA.config import Config
from MIREIA.data_collection.database import create_scenario_dataloaders


def build_scenario_dataloaders(
    seq_len: int,
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
    transform=None,
    scenarios_root: Optional[str] = None,
    model_type: str = "single",
    m_eval_frames: int = 5,
    target_mode: Optional[str] = None,
    subset_ratio: Optional[float] = None,
    subset_seed: int = Config.RANDOM_SEED,
    subset_mode: str = "first",
    max_scenarios: Optional[int] = None,
    window_subset_ratio: Optional[float] = None,
    window_subset_seed: int = Config.RANDOM_SEED,
    window_subset_mode: str = "random",
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, str]:
    if target_mode is None:
        target_mode = "sequence" if model_type == "seq2seq" else "last"

    train_loader, val_loader = create_scenario_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        transform=transform,
        scenarios_root=scenarios_root,
        target_mode=target_mode,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
        subset_mode=subset_mode,
        max_scenarios=max_scenarios,
        window_subset_ratio=window_subset_ratio,
        window_subset_seed=window_subset_seed,
        window_subset_mode=window_subset_mode,
        **dataset_kwargs,
    )
    return train_loader, val_loader, target_mode


def select_targets(
    batch_y: torch.Tensor,
    preds: torch.Tensor,
    model_type: str,
    m_eval_frames: int,
) -> torch.Tensor:
    if model_type == "seq2seq":
        if batch_y.ndim == 3:
            return batch_y[:, -m_eval_frames:, :]
        return batch_y.unsqueeze(1).expand_as(preds)

    if batch_y.ndim == 3:
        return batch_y[:, -1, :]
    return batch_y


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    epochs: int = 1,
    start_epoch: int = 1,
    history: Optional[Dict[str, list[float]]] = None,
    log_every: int = 25,
    max_batches_per_epoch: Optional[int] = None,
    model_type: str = "single",
    m_eval_frames: int = 5,
    grad_clip: Optional[float] = None,
) -> Dict[str, Iterable[float]]:
    criterion = criterion or nn.MSELoss()
    if history is None:
        history = {"train_loss": [], "val_loss": []}

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()

        model.train()
        running_loss = 0.0
        total_samples = 0
        batch_times = []
        total_batches = len(train_loader)

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader, start=1):
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            batch_start.record()

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()

            if model_type == "seq2seq":
                preds = model(batch_x, m_eval_frames=m_eval_frames)
            else:
                preds = model(batch_x)

            target = select_targets(batch_y, preds, model_type, m_eval_frames)
            loss = criterion(preds, target)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            batch_end.record()
            torch.cuda.synchronize() if device.type == "cuda" else None
            batch_times.append(batch_start.elapsed_time(batch_end) / 1000.0)

            if batch_idx == 1 or batch_idx % log_every == 0:
                avg_loss = running_loss / max(1, total_samples)
                avg_batch_time = sum(batch_times) / max(1, len(batch_times))
                remaining_batches = total_batches - batch_idx
                eta_seconds = remaining_batches * avg_batch_time
                print(
                    f"Batch {batch_idx}/{total_batches} | "
                    f"avg loss: {avg_loss:.6f} | "
                    f"batch shape: {tuple(batch_x.shape)} | "
                    f"ETA: {eta_seconds:.1f}s ({eta_seconds/60:.1f}m)"
                )

            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

        train_loss = running_loss / max(1, total_samples)
        history["train_loss"].append(train_loss)
        print(f"Train loss: {train_loss:.6f}")

        if val_loader is not None:
            val_loss = evaluate_model(
                model,
                val_loader,
                device,
                criterion=criterion,
                model_type=model_type,
                m_eval_frames=m_eval_frames,
            )
            history["val_loss"].append(val_loss)
            print(f"Val loss:   {val_loss:.6f}")

        epoch_end.record()
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = epoch_start.elapsed_time(epoch_end) / 1000.0
        print(f"Epoch time: {elapsed:.1f}s")

    return history


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    model_type: str = "single",
    m_eval_frames: int = 5,
) -> float:
    criterion = criterion or nn.MSELoss()
    model.eval()
    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if model_type == "seq2seq":
                preds = model(batch_x, m_eval_frames=m_eval_frames)
            else:
                preds = model(batch_x)
            target = select_targets(batch_y, preds, model_type, m_eval_frames)
            loss = criterion(preds, target)
            val_loss += loss.item() * batch_x.size(0)
            val_samples += batch_x.size(0)

    return val_loss / max(1, val_samples)


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Dict[str, Iterable[float]],
    epoch: int,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
