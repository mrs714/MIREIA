from __future__ import annotations

import os
import random
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from MIREIA.config import Config
from MIREIA.perception.road_segmentation import MireiaRoadSegmentationModel
from MIREIA.perception.training_utils import load_checkpoint, save_checkpoint

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _safe_torch_load(checkpoint_path: str, map_location: torch.device | str | None = None) -> Any:
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def default_road_segmentation_checkpoint_path(checkpoint_name: str = "") -> str:
    if checkpoint_name:
        return os.path.join(Config.PATH_TO_MODELS, checkpoint_name)
    return os.path.join(Config.PATH_TO_MODELS, "road_segmentation_multitask_checkpoint.pt")


def default_road_segmentation_best_checkpoint_path(best_checkpoint_name: str = "") -> str:
    if best_checkpoint_name:
        return os.path.join(Config.PATH_TO_MODELS, best_checkpoint_name)
    return os.path.join(Config.PATH_TO_MODELS, "road_segmentation_multitask_best.pt")


def _resolve_checkpoint_path(checkpoint_path: str, checkpoint_name: str) -> str:
    if checkpoint_path:
        return checkpoint_path
    return default_road_segmentation_checkpoint_path(checkpoint_name=checkpoint_name)


def _resolve_best_checkpoint_path(best_checkpoint_path: str, best_checkpoint_name: str) -> str:
    if best_checkpoint_path:
        return best_checkpoint_path
    return default_road_segmentation_best_checkpoint_path(best_checkpoint_name=best_checkpoint_name)


def _resolve_device(device: str) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_history_template() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "train_lane_loss": [],
        "train_road_loss": [],
        "train_lane_iou": [],
        "train_road_iou": [],
        "train_mean_iou": [],
        "train_lane_dice": [],
        "train_road_dice": [],
        "train_mean_dice": [],
        "val_loss": [],
        "val_lane_loss": [],
        "val_road_loss": [],
        "val_lane_iou": [],
        "val_road_iou": [],
        "val_mean_iou": [],
        "val_lane_dice": [],
        "val_road_dice": [],
        "val_mean_dice": [],
        "learning_rate": [],
    }


def _normalize_history(history: dict[str, Any] | None) -> dict[str, list[float]]:
    template = _build_history_template()
    if history is None:
        return template

    normalized: dict[str, list[float]] = {}
    for key, default_values in template.items():
        values = history.get(key, default_values)
        if isinstance(values, list):
            normalized[key] = [float(v) for v in values]
        else:
            normalized[key] = list(default_values)
    return normalized


def _select_subset_indices(
    total_count: int,
    subset_ratio: float | None,
    subset_seed: int,
    subset_mode: str,
) -> list[int] | None:
    if subset_ratio is None:
        return None

    if not (0.0 < subset_ratio <= 1.0):
        raise ValueError("subset_ratio must be in (0, 1]")

    target_count = max(1, int(total_count * subset_ratio))
    if target_count >= total_count:
        return None

    mode = str(subset_mode).strip().lower()
    if mode == "first":
        return list(range(target_count))
    if mode == "random":
        rng = random.Random(subset_seed)
        return rng.sample(list(range(total_count)), target_count)

    raise ValueError("subset_mode must be 'first' or 'random'")


def _resolve_val_split_key(dataset_dict: Any) -> str:
    available_keys = set(dataset_dict.keys())
    for candidate in ("validation", "val", "test"):
        if candidate in available_keys:
            return candidate
    raise ValueError(
        "Could not find validation split in dataset. Expected one of: "
        "'validation', 'val', 'test'."
    )


def _dataloader_kwargs(
    num_workers: int | None,
    pin_memory: bool | None,
    prefetch_factor: int,
    persistent_workers: bool | None,
) -> dict[str, Any]:
    if num_workers is None:
        cpu_count = os.cpu_count() or 0
        num_workers = min(4, max(0, cpu_count - 1))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)
    return kwargs


def _resolve_mask_key(sample: dict[str, Any], candidates: tuple[str, ...], label: str) -> str:
    for key in candidates:
        if key in sample:
            return key
    raise ValueError(
        f"Could not resolve '{label}' mask key in dataset sample. "
        f"Looked for: {candidates}; sample keys: {list(sample.keys())}"
    )


class HFRoadSegmentationDataset(Dataset):
    """Hugging Face dataset wrapper yielding image + lane mask + road mask."""

    def __init__(
        self,
        hf_split: Any,
        image_size: tuple[int, int],
        augment: bool = False,
        normalize: bool = True,
        indices: Sequence[int] | None = None,
        random_seed: int = Config.RANDOM_SEED,
    ):
        self.data = hf_split
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.indices = list(indices) if indices is not None else list(range(len(hf_split)))
        self.rng = random.Random(random_seed)

        if len(self.indices) == 0:
            raise ValueError("Dataset split is empty after applying subset filters")

        sample = hf_split[self.indices[0]]
        self.lane_key = _resolve_mask_key(
            sample=sample,
            candidates=("lane", "lines", "line", "lanes"),
            label="lane",
        )
        self.road_key = _resolve_mask_key(
            sample=sample,
            candidates=("segment", "road", "roads", "drivable", "mask"),
            label="road",
        )

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _to_pil_image(image_like: Any) -> Image.Image:
        if isinstance(image_like, Image.Image):
            return image_like
        return Image.fromarray(image_like)

    @staticmethod
    def _to_binary_mask(mask: Image.Image) -> torch.Tensor:
        mask_tensor = TF.pil_to_tensor(mask).float()
        if mask_tensor.ndim != 3:
            raise ValueError("Mask tensor must have shape (C, H, W)")
        mask_tensor = mask_tensor[:1]
        return (mask_tensor > 0).float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[self.indices[index]]

        image = self._to_pil_image(item["image"]).convert("RGB")
        lane_mask = self._to_pil_image(item[self.lane_key]).convert("L")
        road_mask = self._to_pil_image(item[self.road_key]).convert("L")

        if self.augment and self.rng.random() < 0.5:
            image = TF.hflip(image)
            lane_mask = TF.hflip(lane_mask)
            road_mask = TF.hflip(road_mask)

        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        lane_mask = TF.resize(
            lane_mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )
        road_mask = TF.resize(
            road_mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        lane_mask_tensor = self._to_binary_mask(lane_mask)
        road_mask_tensor = self._to_binary_mask(road_mask)
        return image_tensor, lane_mask_tensor, road_mask_tensor


def _build_road_segmentation_dataloaders(
    dataset_name: str,
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int | None,
    pin_memory: bool | None,
    prefetch_factor: int,
    persistent_workers: bool | None,
    subset_ratio: float | None,
    val_subset_ratio: float | None,
    subset_seed: int,
    subset_mode: str,
) -> tuple[DataLoader, DataLoader]:
    if load_dataset is None:
        raise ImportError(
            "Missing optional dependency 'datasets'. "
            "Install it in the active environment with: pip install datasets"
        )

    dataset_dict = load_dataset(dataset_name)
    if "train" not in dataset_dict:
        raise ValueError("Dataset must provide a 'train' split")

    val_key = _resolve_val_split_key(dataset_dict)
    train_split = dataset_dict["train"]
    val_split = dataset_dict[val_key]

    train_indices = _select_subset_indices(
        total_count=len(train_split),
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
        subset_mode=subset_mode,
    )

    effective_val_ratio = val_subset_ratio if val_subset_ratio is not None else subset_ratio
    val_indices = _select_subset_indices(
        total_count=len(val_split),
        subset_ratio=effective_val_ratio,
        subset_seed=subset_seed + 1,
        subset_mode=subset_mode,
    )

    train_dataset = HFRoadSegmentationDataset(
        hf_split=train_split,
        image_size=image_size,
        augment=True,
        normalize=True,
        indices=train_indices,
        random_seed=subset_seed,
    )
    val_dataset = HFRoadSegmentationDataset(
        hf_split=val_split,
        image_size=image_size,
        augment=False,
        normalize=True,
        indices=val_indices,
        random_seed=subset_seed + 1,
    )

    loader_kwargs = _dataloader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def _batch_iou_dice(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)

    intersection = (preds * masks).sum(dim=dims)
    union = ((preds + masks) > 0).float().sum(dim=dims)
    iou = (intersection + eps) / (union + eps)

    denom = preds.sum(dim=dims) + masks.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return iou.mean(), dice.mean()


def _multitask_iou_dice(
    lane_logits: torch.Tensor,
    lane_masks: torch.Tensor,
    road_logits: torch.Tensor,
    road_masks: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    lane_iou, lane_dice = _batch_iou_dice(lane_logits, lane_masks, threshold=threshold)
    road_iou, road_dice = _batch_iou_dice(road_logits, road_masks, threshold=threshold)
    mean_iou = 0.5 * (lane_iou + road_iou)
    mean_dice = 0.5 * (lane_dice + road_dice)
    return {
        "lane_iou": lane_iou,
        "road_iou": road_iou,
        "mean_iou": mean_iou,
        "lane_dice": lane_dice,
        "road_dice": road_dice,
        "mean_dice": mean_dice,
    }


def evaluate_road_segmentation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool = False,
    pred_threshold: float = 0.5,
    lane_loss_weight: float = 1.0,
    road_loss_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    running_loss = 0.0
    running_lane_loss = 0.0
    running_road_loss = 0.0
    running_lane_iou = 0.0
    running_road_iou = 0.0
    running_mean_iou = 0.0
    running_lane_dice = 0.0
    running_road_dice = 0.0
    running_mean_dice = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, lane_masks, road_masks in val_loader:
            images = images.to(device, non_blocking=True)
            lane_masks = lane_masks.to(device, non_blocking=True)
            road_masks = road_masks.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                lane_logits, road_logits = model(images)
                lane_loss = criterion(lane_logits, lane_masks)
                road_loss = criterion(road_logits, road_masks)
                loss = lane_loss_weight * lane_loss + road_loss_weight * road_loss

            metrics = _multitask_iou_dice(
                lane_logits=lane_logits,
                lane_masks=lane_masks,
                road_logits=road_logits,
                road_masks=road_masks,
                threshold=pred_threshold,
            )

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            running_lane_loss += float(lane_loss.item()) * batch_size
            running_road_loss += float(road_loss.item()) * batch_size
            running_lane_iou += float(metrics["lane_iou"].item()) * batch_size
            running_road_iou += float(metrics["road_iou"].item()) * batch_size
            running_mean_iou += float(metrics["mean_iou"].item()) * batch_size
            running_lane_dice += float(metrics["lane_dice"].item()) * batch_size
            running_road_dice += float(metrics["road_dice"].item()) * batch_size
            running_mean_dice += float(metrics["mean_dice"].item()) * batch_size
            sample_count += batch_size

    denom = max(1, sample_count)
    return {
        "loss": running_loss / denom,
        "lane_loss": running_lane_loss / denom,
        "road_loss": running_road_loss / denom,
        "lane_iou": running_lane_iou / denom,
        "road_iou": running_road_iou / denom,
        "mean_iou": running_mean_iou / denom,
        "lane_dice": running_lane_dice / denom,
        "road_dice": running_road_dice / denom,
        "mean_dice": running_mean_dice / denom,
    }


def train_road_segmentation_model(
    task: str | None = None,
    dataset_name: str = "bnsapa/road-detection",
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: int = 256,
    dropout: float = 0.1,
    backbone_pretrained: bool = True,
    num_workers: int | None = 0,
    pin_memory: bool | None = None,
    prefetch_factor: int = 2,
    persistent_workers: bool | None = None,
    subset_ratio: float | None = None,
    val_subset_ratio: float | None = None,
    subset_seed: int = Config.RANDOM_SEED,
    subset_mode: str = "random",
    disable_amp: bool = False,
    pred_threshold: float = 0.5,
    lane_loss_weight: float = 1.0,
    road_loss_weight: float = 1.0,
    grad_accum_steps: int = 1,
    grad_clip: float | None = None,
    checkpoint_path: str = "",
    checkpoint_name: str = "",
    best_checkpoint_path: str = "",
    best_checkpoint_name: str = "",
    no_resume: bool = False,
    device: str = "",
    log_every: int = 25,
) -> dict[str, Any]:
    """Train (or resume) multitask road segmentation with lane and road heads."""
    if task is not None:
        normalized_task = str(task).strip().lower()
        if normalized_task not in {"lane", "segment", "both", "multitask"}:
            raise ValueError("task must be one of: lane, segment, both, multitask")
        print("task argument is ignored because multitask training always predicts lane and road.")

    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")
    if not (0.0 < pred_threshold < 1.0):
        raise ValueError("pred_threshold must be in (0, 1)")
    if lane_loss_weight <= 0.0 or road_loss_weight <= 0.0:
        raise ValueError("lane_loss_weight and road_loss_weight must be > 0")

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path, checkpoint_name)
    best_checkpoint_path = _resolve_best_checkpoint_path(best_checkpoint_path, best_checkpoint_name)
    resume_enabled = not no_resume

    size = (int(image_size), int(image_size))
    train_loader, val_loader = _build_road_segmentation_dataloaders(
        dataset_name=dataset_name,
        image_size=size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        subset_ratio=subset_ratio,
        val_subset_ratio=val_subset_ratio,
        subset_seed=subset_seed,
        subset_mode=subset_mode,
    )

    resolved_device = _resolve_device(device)
    if backbone_pretrained:
        try:
            model = MireiaRoadSegmentationModel(
                dropout=dropout,
                input_size=size,
            )
        except Exception as exc:
            print(f"Could not load pretrained backbone weights ({exc}). Falling back to random init.")
            model = MireiaRoadSegmentationModel(
                dropout=dropout,
                input_size=size,
                backbone_weights=None,
            )
    else:
        model = MireiaRoadSegmentationModel(
            dropout=dropout,
            input_size=size,
            backbone_weights=None,
        )

    model = model.to(resolved_device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    criterion = nn.BCEWithLogitsLoss()

    history = _build_history_template()
    start_epoch = 1
    best_mean_iou = 0.0

    if resume_enabled and os.path.isfile(checkpoint_path):
        try:
            checkpoint_state = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                device=resolved_device,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to resume checkpoint with the current multitask model architecture. "
                "If this is a legacy single-task checkpoint, set no_resume=True or provide a new checkpoint path."
            ) from exc

        if isinstance(checkpoint_state, dict):
            history = _normalize_history(checkpoint_state.get("history"))
            start_epoch = int(checkpoint_state.get("epoch", 0)) + 1
            best_mean_iou = float(
                checkpoint_state.get("best_mean_iou", checkpoint_state.get("best_iou", best_mean_iou))
            )

            scheduler_state = checkpoint_state.get("scheduler_state_dict")
            if isinstance(scheduler_state, dict):
                scheduler.load_state_dict(scheduler_state)

        print(f"Resuming from {checkpoint_path} at epoch {start_epoch}")
    else:
        print(f"No checkpoint resume. Starting fresh at {checkpoint_path}")

    amp_enabled = bool(torch.cuda.is_available() and not disable_amp and resolved_device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    total_epochs = start_epoch + epochs - 1
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()

        running_loss = 0.0
        running_lane_loss = 0.0
        running_road_loss = 0.0
        running_lane_iou = 0.0
        running_road_iou = 0.0
        running_mean_iou = 0.0
        running_lane_dice = 0.0
        running_road_dice = 0.0
        running_mean_dice = 0.0
        sample_count = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, lane_masks, road_masks) in enumerate(train_loader, start=1):
            images = images.to(resolved_device, non_blocking=True)
            lane_masks = lane_masks.to(resolved_device, non_blocking=True)
            road_masks = road_masks.to(resolved_device, non_blocking=True)

            with torch.autocast(device_type=resolved_device.type, enabled=amp_enabled):
                lane_logits, road_logits = model(images)
                lane_loss = criterion(lane_logits, lane_masks)
                road_loss = criterion(road_logits, road_masks)
                loss = lane_loss_weight * lane_loss + road_loss_weight * road_loss

            step_loss = loss / grad_accum_steps
            if amp_enabled:
                scaler.scale(step_loss).backward()
            else:
                step_loss.backward()

            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == len(train_loader))
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

            metrics = _multitask_iou_dice(
                lane_logits=lane_logits,
                lane_masks=lane_masks,
                road_logits=road_logits,
                road_masks=road_masks,
                threshold=pred_threshold,
            )

            batch_size_now = images.size(0)
            running_loss += float(loss.item()) * batch_size_now
            running_lane_loss += float(lane_loss.item()) * batch_size_now
            running_road_loss += float(road_loss.item()) * batch_size_now
            running_lane_iou += float(metrics["lane_iou"].item()) * batch_size_now
            running_road_iou += float(metrics["road_iou"].item()) * batch_size_now
            running_mean_iou += float(metrics["mean_iou"].item()) * batch_size_now
            running_lane_dice += float(metrics["lane_dice"].item()) * batch_size_now
            running_road_dice += float(metrics["road_dice"].item()) * batch_size_now
            running_mean_dice += float(metrics["mean_dice"].item()) * batch_size_now
            sample_count += batch_size_now

            if batch_idx == 1 or batch_idx % log_every == 0:
                avg_loss = running_loss / max(1, sample_count)
                avg_lane_iou = running_lane_iou / max(1, sample_count)
                avg_road_iou = running_road_iou / max(1, sample_count)
                avg_mean_iou = running_mean_iou / max(1, sample_count)
                print(
                    f"Epoch {epoch}/{total_epochs} | "
                    f"batch {batch_idx}/{len(train_loader)} | "
                    f"loss={avg_loss:.6f} | "
                    f"lane_iou={avg_lane_iou:.4f} | "
                    f"road_iou={avg_road_iou:.4f} | "
                    f"mean_iou={avg_mean_iou:.4f}"
                )

        denom = max(1, sample_count)
        train_loss = running_loss / denom
        train_lane_loss = running_lane_loss / denom
        train_road_loss = running_road_loss / denom
        train_lane_iou = running_lane_iou / denom
        train_road_iou = running_road_iou / denom
        train_mean_iou = running_mean_iou / denom
        train_lane_dice = running_lane_dice / denom
        train_road_dice = running_road_dice / denom
        train_mean_dice = running_mean_dice / denom

        val_metrics = evaluate_road_segmentation(
            model=model,
            val_loader=val_loader,
            device=resolved_device,
            criterion=criterion,
            use_amp=amp_enabled,
            pred_threshold=pred_threshold,
            lane_loss_weight=lane_loss_weight,
            road_loss_weight=road_loss_weight,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["train_lane_loss"].append(train_lane_loss)
        history["train_road_loss"].append(train_road_loss)
        history["train_lane_iou"].append(train_lane_iou)
        history["train_road_iou"].append(train_road_iou)
        history["train_mean_iou"].append(train_mean_iou)
        history["train_lane_dice"].append(train_lane_dice)
        history["train_road_dice"].append(train_road_dice)
        history["train_mean_dice"].append(train_mean_dice)
        history["val_loss"].append(val_metrics["loss"])
        history["val_lane_loss"].append(val_metrics["lane_loss"])
        history["val_road_loss"].append(val_metrics["road_loss"])
        history["val_lane_iou"].append(val_metrics["lane_iou"])
        history["val_road_iou"].append(val_metrics["road_iou"])
        history["val_mean_iou"].append(val_metrics["mean_iou"])
        history["val_lane_dice"].append(val_metrics["lane_dice"])
        history["val_road_dice"].append(val_metrics["road_dice"])
        history["val_mean_dice"].append(val_metrics["mean_dice"])
        history["learning_rate"].append(current_lr)

        scheduler.step(val_metrics["loss"])

        print(
            f"Epoch {epoch}/{total_epochs} summary | "
            f"train_loss={train_loss:.6f}, val_loss={val_metrics['loss']:.6f} | "
            f"train_mean_iou={train_mean_iou:.4f}, val_mean_iou={val_metrics['mean_iou']:.4f} | "
            f"train_lane_iou={train_lane_iou:.4f}, val_lane_iou={val_metrics['lane_iou']:.4f} | "
            f"train_road_iou={train_road_iou:.4f}, val_road_iou={val_metrics['road_iou']:.4f}"
        )

        improved = val_metrics["mean_iou"] > best_mean_iou
        if improved:
            best_mean_iou = float(val_metrics["mean_iou"])

        extra_payload = {
            "model_type": "road_segmentation_multitask",
            "task": "multitask",
            "tasks": ["lane", "road"],
            "dataset_mask_keys": {"lane": "lane", "road": "segment"},
            "dataset_name": dataset_name,
            "image_size": [size[0], size[1]],
            "pred_threshold": pred_threshold,
            "lane_loss_weight": lane_loss_weight,
            "road_loss_weight": road_loss_weight,
            "num_classes": 1,
            "backbone_pretrained": backbone_pretrained,
            "best_mean_iou": best_mean_iou,
            "best_iou": best_mean_iou,
            "scheduler_state_dict": scheduler.state_dict(),
        }

        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            history=history,
            epoch=epoch,
            extra=extra_payload,
        )

        if improved:
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                history=history,
                epoch=epoch,
                extra=extra_payload,
            )
            print(
                f"Saved new best checkpoint (mean IoU={best_mean_iou:.4f}) "
                f"to {best_checkpoint_path}"
            )

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "best_checkpoint_path": best_checkpoint_path,
        "best_mean_iou": best_mean_iou,
        "best_iou": best_mean_iou,
        "start_epoch": start_epoch,
        "final_epoch": total_epochs,
        "model_type": "road_segmentation_multitask",
        "task": "multitask",
        "tasks": ["lane", "road"],
        "dataset_name": dataset_name,
        "pred_threshold": pred_threshold,
        "lane_loss_weight": lane_loss_weight,
        "road_loss_weight": road_loss_weight,
        "image_size": size,
        "device": str(resolved_device),
    }


def load_road_segmentation_model(
    checkpoint_path: str = "",
    checkpoint_name: str = "road_segmentation_multitask_checkpoint.pt",
    device: str = "",
) -> dict[str, Any]:
    resolved_path = checkpoint_path or os.path.join(Config.PATH_TO_MODELS, checkpoint_name)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

    resolved_device = _resolve_device(device)
    payload = _safe_torch_load(resolved_path, map_location=resolved_device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")

    image_size_raw = payload.get("image_size", [256, 256])
    if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    else:
        image_size = (256, 256)

    model = MireiaRoadSegmentationModel(
        input_size=image_size,
        dropout=0.1,
        backbone_weights=None,
    ).to(resolved_device)

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain model_state_dict: {resolved_path}")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights are incompatible with the current two-head model. "
            "Use a multitask checkpoint or retrain with the updated trainer."
        ) from exc

    model.eval()

    history = payload.get("history", {})
    if not isinstance(history, dict):
        history = {}

    tasks_raw = payload.get("tasks", ["lane", "road"])
    if isinstance(tasks_raw, (list, tuple)) and tasks_raw:
        tasks = [str(task_name) for task_name in tasks_raw]
    else:
        tasks = ["lane", "road"]

    best_mean_iou = float(payload.get("best_mean_iou", payload.get("best_iou", 0.0)))

    return {
        "model": model,
        "history": history,
        "checkpoint_path": resolved_path,
        "task": "multitask",
        "tasks": tasks,
        "dataset_name": str(payload.get("dataset_name", "bnsapa/road-detection")),
        "pred_threshold": float(payload.get("pred_threshold", 0.5)),
        "image_size": image_size,
        "best_mean_iou": best_mean_iou,
        "best_iou": best_mean_iou,
        "final_epoch": int(payload.get("epoch", 0)),
        "device": str(resolved_device),
    }


if __name__ == "__main__":
    train_road_segmentation_model()
