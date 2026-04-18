from __future__ import annotations

import os
import time
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from MIREIA.config import Config
from MIREIA.data_collection.database import create_scenario_dataloaders
from MIREIA.data_collection.scenario_multitask_dataset import (
    create_environment_dataloaders,
)


def _is_sequence_model_type(model_type: str) -> bool:
    return str(model_type).strip().lower() in {"seq2seq", "e2e"}


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
    partition_mode: str = "scenario",
    val_scenario_tokens: str | Iterable[str] | None = None,
    frame_train_ratio: float = 0.7,
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
        target_mode = "sequence" if _is_sequence_model_type(model_type) else "last"

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
        partition_mode=partition_mode,
        val_scenario_tokens=val_scenario_tokens,
        frame_train_ratio=frame_train_ratio,
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
    if _is_sequence_model_type(model_type):
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
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> Dict[str, Iterable[float]]:
    criterion = criterion or nn.MSELoss()
    if history is None:
        history = {"train_loss": [], "val_loss": []}

    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    use_cuda_timing = device.type == "cuda"

    for epoch in range(start_epoch, start_epoch + epochs):
        if use_cuda_timing:
            epoch_start = torch.cuda.Event(enable_timing=True)
            epoch_end = torch.cuda.Event(enable_timing=True)
            epoch_start.record()
        else:
            epoch_start_time = time.perf_counter()

        model.train()
        running_loss = 0.0
        total_samples = 0
        batch_times = []
        total_batches = len(train_loader)
        effective_total_batches = (
            min(total_batches, max_batches_per_epoch)
            if max_batches_per_epoch is not None
            else total_batches
        )
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader, start=1):
            if use_cuda_timing:
                batch_start = torch.cuda.Event(enable_timing=True)
                batch_end = torch.cuda.Event(enable_timing=True)
                batch_start.record()
            else:
                batch_start_time = time.perf_counter()

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if _is_sequence_model_type(model_type):
                    preds = model(batch_x, m_eval_frames=m_eval_frames)
                else:
                    preds = model(batch_x)

                target = select_targets(batch_y, preds, model_type, m_eval_frames)
                loss = criterion(preds, target)

            step_loss = loss / grad_accum_steps
            if amp_enabled:
                scaler.scale(step_loss).backward()
            else:
                step_loss.backward()

            should_step = (
                (batch_idx % grad_accum_steps == 0)
                or (batch_idx == effective_total_batches)
            )
            if should_step:
                if grad_clip is not None:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            if use_cuda_timing:
                batch_end.record()
                torch.cuda.synchronize()
                batch_times.append(batch_start.elapsed_time(batch_end) / 1000.0)
            else:
                batch_times.append(time.perf_counter() - batch_start_time)

            if batch_idx == 1 or batch_idx % log_every == 0:
                avg_loss = running_loss / max(1, total_samples)
                avg_batch_time = sum(batch_times) / max(1, len(batch_times))
                remaining_batches = effective_total_batches - batch_idx
                eta_seconds = remaining_batches * avg_batch_time
                print(
                    f"Batch {batch_idx}/{effective_total_batches} | "
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
                use_amp=use_amp,
            )
            history["val_loss"].append(val_loss)
            print(f"Val loss:   {val_loss:.6f}")

        if use_cuda_timing:
            epoch_end.record()
            torch.cuda.synchronize()
            elapsed = epoch_start.elapsed_time(epoch_end) / 1000.0
        else:
            elapsed = time.perf_counter() - epoch_start_time
        print(f"Epoch time: {elapsed:.1f}s")

    return history


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    model_type: str = "single",
    m_eval_frames: int = 5,
    use_amp: bool = False,
) -> float:
    criterion = criterion or nn.MSELoss()
    model.eval()
    val_loss = 0.0
    val_samples = 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if _is_sequence_model_type(model_type):
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


def default_environment_checkpoint_path(
    checkpoint_name: str = "environment_multitask_checkpoint.pt",
) -> str:
    return os.path.join(Config.PATH_TO_MODELS, checkpoint_name)


def build_environment_dataloaders(
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
    transform=None,
    scenarios_root: Optional[str] = None,
    partition_mode: str = "scenario",
    val_scenario_tokens: str | Iterable[str] | None = None,
    frame_train_ratio: float = 0.7,
    subset_ratio: Optional[float] = None,
    subset_seed: int = Config.RANDOM_SEED,
    subset_mode: str = "first",
    max_scenarios: Optional[int] = None,
    frame_subset_ratio: Optional[float] = None,
    frame_subset_seed: int = Config.RANDOM_SEED,
    frame_subset_mode: str = "random",
    climate_to_idx: Optional[Dict[str, int]] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_loader, val_loader, discovered_climate_to_idx = create_environment_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        transform=transform,
        scenarios_root=scenarios_root,
        partition_mode=partition_mode,
        val_scenario_tokens=val_scenario_tokens,
        frame_train_ratio=frame_train_ratio,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
        subset_mode=subset_mode,
        max_scenarios=max_scenarios,
        frame_subset_ratio=frame_subset_ratio,
        frame_subset_seed=frame_subset_seed,
        frame_subset_mode=frame_subset_mode,
        climate_to_idx=climate_to_idx,
        **dataset_kwargs,
    )
    return train_loader, val_loader, discovered_climate_to_idx


def evaluate_environment_classifier(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    day_criterion: Optional[nn.Module] = None,
    weather_criterion: Optional[nn.Module] = None,
    day_loss_weight: float = 1.0,
    weather_loss_weight: float = 1.0,
    use_amp: bool = False,
) -> Dict[str, float]:
    day_criterion = day_criterion or nn.CrossEntropyLoss()
    weather_criterion = weather_criterion or nn.CrossEntropyLoss()

    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    total_loss = 0.0
    total_day_loss = 0.0
    total_weather_loss = 0.0
    total_samples = 0
    day_correct = 0
    weather_correct = 0

    with torch.no_grad():
        for batch_x, day_targets, weather_targets in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            day_targets = day_targets.to(device, non_blocking=True)
            weather_targets = weather_targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                day_logits, weather_logits = model(batch_x)
                day_loss = day_criterion(day_logits, day_targets)
                weather_loss = weather_criterion(weather_logits, weather_targets)
                loss = day_loss_weight * day_loss + weather_loss_weight * weather_loss

            batch_size = batch_x.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_day_loss += day_loss.item() * batch_size
            total_weather_loss += weather_loss.item() * batch_size

            day_preds = torch.argmax(day_logits, dim=1)
            weather_preds = torch.argmax(weather_logits, dim=1)
            day_correct += int((day_preds == day_targets).sum().item())
            weather_correct += int((weather_preds == weather_targets).sum().item())

    denom = max(1, total_samples)
    return {
        "loss": total_loss / denom,
        "day_loss": total_day_loss / denom,
        "weather_loss": total_weather_loss / denom,
        "day_acc": day_correct / denom,
        "weather_acc": weather_correct / denom,
    }


def train_environment_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    day_criterion: Optional[nn.Module] = None,
    weather_criterion: Optional[nn.Module] = None,
    epochs: int = 1,
    start_epoch: int = 1,
    history: Optional[Dict[str, list[float]]] = None,
    day_loss_weight: float = 1.0,
    weather_loss_weight: float = 1.0,
    log_every: int = 25,
    max_batches_per_epoch: Optional[int] = None,
    grad_clip: Optional[float] = None,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> Dict[str, list[float]]:
    day_criterion = day_criterion or nn.CrossEntropyLoss()
    weather_criterion = weather_criterion or nn.CrossEntropyLoss()

    default_history = {
        "train_loss": [],
        "train_day_loss": [],
        "train_weather_loss": [],
        "train_day_acc": [],
        "train_weather_acc": [],
        "val_loss": [],
        "val_day_loss": [],
        "val_weather_loss": [],
        "val_day_acc": [],
        "val_weather_acc": [],
    }
    if history is None:
        history = default_history
    else:
        for key, default_value in default_history.items():
            history.setdefault(key, list(default_value))

    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.perf_counter()
        model.train()

        running_loss = 0.0
        running_day_loss = 0.0
        running_weather_loss = 0.0
        total_samples = 0
        day_correct = 0
        weather_correct = 0

        total_batches = len(train_loader)
        effective_total_batches = (
            min(total_batches, max_batches_per_epoch)
            if max_batches_per_epoch is not None
            else total_batches
        )

        batch_times: list[float] = []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_x, day_targets, weather_targets) in enumerate(train_loader, start=1):
            batch_start = time.perf_counter()

            batch_x = batch_x.to(device, non_blocking=True)
            day_targets = day_targets.to(device, non_blocking=True)
            weather_targets = weather_targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                day_logits, weather_logits = model(batch_x)
                day_loss = day_criterion(day_logits, day_targets)
                weather_loss = weather_criterion(weather_logits, weather_targets)
                loss = day_loss_weight * day_loss + weather_loss_weight * weather_loss

            step_loss = loss / grad_accum_steps
            if amp_enabled:
                scaler.scale(step_loss).backward()
            else:
                step_loss.backward()

            should_step = (
                (batch_idx % grad_accum_steps == 0)
                or (batch_idx == effective_total_batches)
            )
            if should_step:
                if grad_clip is not None:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = batch_x.size(0)
            total_samples += batch_size
            running_loss += loss.item() * batch_size
            running_day_loss += day_loss.item() * batch_size
            running_weather_loss += weather_loss.item() * batch_size

            day_preds = torch.argmax(day_logits, dim=1)
            weather_preds = torch.argmax(weather_logits, dim=1)
            day_correct += int((day_preds == day_targets).sum().item())
            weather_correct += int((weather_preds == weather_targets).sum().item())

            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)

            if batch_idx == 1 or batch_idx % log_every == 0:
                avg_loss = running_loss / max(1, total_samples)
                avg_day_acc = day_correct / max(1, total_samples)
                avg_weather_acc = weather_correct / max(1, total_samples)
                avg_batch_time = sum(batch_times) / max(1, len(batch_times))
                remaining_batches = effective_total_batches - batch_idx
                eta_seconds = remaining_batches * avg_batch_time
                print(
                    f"Batch {batch_idx}/{effective_total_batches} | "
                    f"avg loss: {avg_loss:.6f} | "
                    f"day acc: {avg_day_acc:.3f} | "
                    f"weather acc: {avg_weather_acc:.3f} | "
                    f"batch shape: {tuple(batch_x.shape)} | "
                    f"ETA: {eta_seconds:.1f}s ({eta_seconds/60:.1f}m)"
                )

            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

        denom = max(1, total_samples)
        train_loss = running_loss / denom
        train_day_loss = running_day_loss / denom
        train_weather_loss = running_weather_loss / denom
        train_day_acc = day_correct / denom
        train_weather_acc = weather_correct / denom

        history["train_loss"].append(train_loss)
        history["train_day_loss"].append(train_day_loss)
        history["train_weather_loss"].append(train_weather_loss)
        history["train_day_acc"].append(train_day_acc)
        history["train_weather_acc"].append(train_weather_acc)

        print(
            f"Train loss: {train_loss:.6f} "
            f"(day: {train_day_loss:.6f}, weather: {train_weather_loss:.6f})"
        )
        print(
            f"Train acc:  day={train_day_acc:.3f} | weather={train_weather_acc:.3f}"
        )

        if val_loader is not None:
            val_metrics = evaluate_environment_classifier(
                model,
                val_loader,
                device,
                day_criterion=day_criterion,
                weather_criterion=weather_criterion,
                day_loss_weight=day_loss_weight,
                weather_loss_weight=weather_loss_weight,
                use_amp=use_amp,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_day_loss"].append(val_metrics["day_loss"])
            history["val_weather_loss"].append(val_metrics["weather_loss"])
            history["val_day_acc"].append(val_metrics["day_acc"])
            history["val_weather_acc"].append(val_metrics["weather_acc"])
            print(
                f"Val loss:   {val_metrics['loss']:.6f} "
                f"(day: {val_metrics['day_loss']:.6f}, weather: {val_metrics['weather_loss']:.6f})"
            )
            print(
                f"Val acc:    day={val_metrics['day_acc']:.3f} | "
                f"weather={val_metrics['weather_acc']:.3f}"
            )

        elapsed = time.perf_counter() - epoch_start
        print(f"Epoch time: {elapsed:.1f}s")

    return history
