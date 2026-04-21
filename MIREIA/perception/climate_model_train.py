from __future__ import annotations

import os

import torch
from torch import nn, optim
from torchvision import transforms

from MIREIA.config import Config
from MIREIA.perception.climate_model import MireiaEnvironmentClassifier
from MIREIA.perception.training_utils import (
    build_default_train_val_include_names,
    build_environment_dataloaders,
    default_val_scenario_tokens_csv,
    default_environment_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    train_environment_classifier,
)


def _resolve_checkpoint_path(checkpoint_path: str, checkpoint_name: str) -> str:
    if checkpoint_path:
        return checkpoint_path
    return default_environment_checkpoint_path(checkpoint_name)


def _load_checkpoint_metadata(checkpoint_path: str) -> dict[str, object] | None:
    if not os.path.isfile(checkpoint_path):
        return None
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return None


def train_environment_model(
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    disable_amp: bool = False,
    log_every: int = 25,
    grad_accum_steps: int = 1,
    grad_clip: float | None = None,
    image_size: int = 512,
    dropout: float = 0.2,
    day_loss_weight: float = 1.0,
    weather_loss_weight: float = 1.0,
    scenarios_root: str | None = None,
    partition_mode: str = "scenario",
    val_scenario_tokens: str | list[str] | None = default_val_scenario_tokens_csv(),
    frame_train_ratio: float = 0.7,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
    subset_ratio: float | None = None,
    subset_seed: int = Config.RANDOM_SEED,
    subset_mode: str = "first",
    max_scenarios: int | None = None,
    frame_subset_ratio: float | None = None,
    frame_subset_seed: int = Config.RANDOM_SEED,
    frame_subset_mode: str = "random",
    checkpoint_path: str = "",
    checkpoint_name: str = "environment_multitask_checkpoint.pt",
    no_resume: bool = False,
    device: str = "",
    resize_images: bool = False,
) -> dict[str, object]:
    """Train (or resume) the multitask day/night + climate model."""
    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path, checkpoint_name)
    resume_enabled = not no_resume
    checkpoint_exists = os.path.isfile(checkpoint_path)

    prior_metadata = None
    if resume_enabled and checkpoint_exists:
        prior_metadata = _load_checkpoint_metadata(checkpoint_path)

    prior_climate_to_idx = None
    if isinstance(prior_metadata, dict):
        maybe_mapping = prior_metadata.get("climate_to_idx")
        if isinstance(maybe_mapping, dict) and maybe_mapping:
            prior_climate_to_idx = {
                str(label): int(idx) for label, idx in maybe_mapping.items()
            }

    transform_steps = []
    if resize_images:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)

    if partition_mode == "scenario" and include_names is None:
        include_names = build_default_train_val_include_names(
            scenarios_root=scenarios_root or Config.PATH_TO_SCENARIOS
        )

    train_loader, val_loader, climate_to_idx = build_environment_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        transform=transform,
        scenarios_root=scenarios_root,
        partition_mode=partition_mode,
        val_scenario_tokens=val_scenario_tokens,
        frame_train_ratio=frame_train_ratio,
        include_names=include_names,
        exclude_names=exclude_names,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
        subset_mode=subset_mode,
        max_scenarios=max_scenarios,
        frame_subset_ratio=frame_subset_ratio,
        frame_subset_seed=frame_subset_seed,
        frame_subset_mode=frame_subset_mode,
        climate_to_idx=prior_climate_to_idx,
    )

    idx_to_climate = [
        label for label, _ in sorted(climate_to_idx.items(), key=lambda item: item[1])
    ]

    if device:
        resolved_device = torch.device(device)
    else:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MireiaEnvironmentClassifier(
        num_weather_classes=len(climate_to_idx),
        dropout=dropout,
        input_size=(image_size, image_size),
    ).to(resolved_device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    day_criterion = nn.CrossEntropyLoss()
    weather_criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    history = None

    if resume_enabled and checkpoint_exists:
        state = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=resolved_device,
        )
        loaded_history = state.get("history")
        if isinstance(loaded_history, dict):
            history = loaded_history
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resuming from {checkpoint_path} at epoch {start_epoch}")
    else:
        print(f"No checkpoint resume. Starting fresh at {checkpoint_path}")

    use_amp = bool(torch.cuda.is_available() and not disable_amp)
    history = train_environment_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=resolved_device,
        day_criterion=day_criterion,
        weather_criterion=weather_criterion,
        epochs=epochs,
        start_epoch=start_epoch,
        history=history,
        day_loss_weight=day_loss_weight,
        weather_loss_weight=weather_loss_weight,
        log_every=log_every,
        grad_clip=grad_clip,
        use_amp=use_amp,
        grad_accum_steps=grad_accum_steps,
    )

    final_epoch = start_epoch + epochs - 1
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        history=history,
        epoch=final_epoch,
        extra={
            "model_type": "environment_multitask",
            "day_night_labels": ["day", "night"],
            "climate_to_idx": climate_to_idx,
            "idx_to_climate": idx_to_climate,
            "image_size": [image_size, image_size],
            "scenarios_root": scenarios_root or Config.PATH_TO_SCENARIOS,
            "use_amp": use_amp,
        },
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "start_epoch": start_epoch,
        "final_epoch": final_epoch,
        "model_type": "environment_multitask",
        "climate_to_idx": climate_to_idx,
        "idx_to_climate": idx_to_climate,
        "device": str(resolved_device),
    }


if __name__ == "__main__":
    train_environment_model()
