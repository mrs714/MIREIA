from __future__ import annotations

import os

import torch
from torch import nn, optim

from MIREIA.config import Config
from MIREIA.data_collection.feature_sequence_dataset import (
    create_feature_sequence_dataloaders,
)
from MIREIA.perception.bdu_gru_model import (
    BDUGRUModelConfig,
    BDUGRURiskPredictor,
    Seq2SeqBDUGRURiskPredictor,
)
from MIREIA.perception.training_utils import (
    load_checkpoint,
    save_checkpoint,
    train_model,
)


def _normalize_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    if normalized in {"bdu_gru", "e2e", "seq2seq"}:
        return "seq2seq"
    if normalized in {"single", "bdu_gru_single"}:
        return "single"
    raise ValueError("model_type must be 'bdu_gru' (alias: 'seq2seq') or 'single'")


def _public_model_type(model_type: str) -> str:
    return "bdu_gru" if model_type == "seq2seq" else model_type


def train_bdu_gru_model(
    resume_epochs: int = 1,
    model_type: str = "bdu_gru",
    seq_len: int = Config.INFERENCE_SEQUENCE_LENGTH,
    burn_in_frames: int = Config.INFERENCE_BURN_IN_FRAMES,
    m_eval_frames: int = Config.INFERENCE_EVAL_FRAMES,
    feature_dim: int = 32,
    feature_key: str = "feature_vector_32",
    risk_key: str = "ground_truth_risk",
    batch_size: int = 16,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
    scenarios_root: str | None = None,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
    subset_ratio: float | None = None,
    subset_seed: int = Config.RANDOM_SEED,
    subset_mode: str = "first",
    max_scenarios: int | None = None,
    window_subset_ratio: float | None = None,
    window_subset_seed: int = Config.RANDOM_SEED,
    window_subset_mode: str = "random",
    prefer_labeled_jsonl: bool = True,
    fallback_to_dataset_jsonl: bool = False,
    fallback_to_zeros: bool = False,
    checkpoint_path: str = "",
    checkpoint_name: str = "bdu_gru_risk_checkpoint.pt",
    no_resume: bool = False,
    device: str = "",
    grad_clip: float | None = 1.0,
    loss_type: str = "smooth_l1",
    huber_beta: float = 1.0,
    grad_accum_steps: int = 1,
) -> dict[str, object]:
    """Train or resume BDU-GRU risk model over per-frame 32D feature vectors."""
    if resume_epochs <= 0:
        raise ValueError("resume_epochs must be > 0")
    if burn_in_frames + m_eval_frames != seq_len:
        raise ValueError(
            f"Invalid temporal setup: burn_in({burn_in_frames}) + "
            f"eval({m_eval_frames}) != seq_len({seq_len})"
        )

    if grad_clip is None or float(grad_clip) <= 0.0:
        grad_clip = 1.0
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be > 0")

    model_type_internal = _normalize_model_type(model_type)
    target_mode = "sequence" if model_type_internal == "seq2seq" else "last"

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader, val_loader = create_feature_sequence_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        scenarios_root=scenarios_root,
        include_names=include_names,
        exclude_names=exclude_names,
        target_mode=target_mode,
        risk_key=risk_key,
        feature_key=feature_key,
        feature_dim=feature_dim,
        prefer_labeled_jsonl=prefer_labeled_jsonl,
        fallback_to_dataset_jsonl=fallback_to_dataset_jsonl,
        fallback_to_zeros=fallback_to_zeros,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
        subset_mode=subset_mode,
        max_scenarios=max_scenarios,
        window_subset_ratio=window_subset_ratio,
        window_subset_seed=window_subset_seed,
        window_subset_mode=window_subset_mode,
    )

    print(f"Temporal config: seq_len={seq_len}, burn_in={burn_in_frames}, eval={m_eval_frames}")
    print(
        f"DataLoader workers: num_workers={num_workers}, "
        f"prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers}"
    )
    print(f"Gradient clipping max_norm: {grad_clip}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    batch_x, batch_y = next(iter(train_loader))
    print("Batch X shape:", batch_x.shape)
    print("Batch Y shape:", batch_y.shape)
    print("Batch X dtype:", batch_x.dtype)
    print("Batch Y dtype:", batch_y.dtype)

    if device:
        resolved_device = torch.device(device)
    else:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BDUGRUModelConfig(feature_dim=feature_dim)
    model = (
        Seq2SeqBDUGRURiskPredictor(config)
        if model_type_internal == "seq2seq"
        else BDUGRURiskPredictor(config)
    ).to(resolved_device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    normalized_loss = str(loss_type).strip().lower()
    if normalized_loss == "mse":
        criterion = nn.MSELoss()
    elif normalized_loss in {"smooth_l1", "huber"}:
        criterion = nn.SmoothL1Loss(beta=huber_beta)
    else:
        raise ValueError("loss_type must be 'smooth_l1' (recommended) or 'mse'")

    if not checkpoint_path:
        checkpoint_path = os.path.join(Config.PATH_TO_MODELS, checkpoint_name)

    start_epoch = 1
    history = {"train_loss": [], "val_loss": []}

    if os.path.exists(checkpoint_path) and not no_resume:
        ckpt = load_checkpoint(checkpoint_path, model, optimizer=optimizer, device=resolved_device)
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        loaded_model_type = ckpt.get("model_type_internal", ckpt.get("model_type", model_type_internal))
        model_type_internal = _normalize_model_type(str(loaded_model_type))
        m_eval_frames = int(ckpt.get("m_eval_frames", m_eval_frames))
        target_mode = ckpt.get("target_mode", target_mode)
        use_amp = bool(ckpt.get("use_amp", use_amp))
        print(f"Resuming from {checkpoint_path} at epoch {start_epoch}")
    elif os.path.exists(checkpoint_path) and no_resume:
        print(f"Checkpoint exists but no_resume=True, starting fresh: {checkpoint_path}")
    else:
        print("No checkpoint found. Starting fresh.")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=resolved_device,
        criterion=criterion,
        epochs=resume_epochs,
        start_epoch=start_epoch,
        history=history,
        model_type=model_type_internal,
        m_eval_frames=m_eval_frames,
        use_amp=use_amp,
        grad_clip=grad_clip,
        grad_accum_steps=grad_accum_steps,
    )

    final_epoch = start_epoch + resume_epochs - 1
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        history=history,
        epoch=final_epoch,
        extra={
            "model_type": _public_model_type(model_type_internal),
            "model_type_internal": model_type_internal,
            "model_family": "bdu_gru",
            "m_eval_frames": m_eval_frames,
            "seq_len": seq_len,
            "target_mode": target_mode,
            "feature_dim": feature_dim,
            "feature_key": feature_key,
            "risk_key": risk_key,
            "use_amp": use_amp,
            "grad_clip": grad_clip,
            "loss_type": normalized_loss,
            "huber_beta": huber_beta,
        },
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "final_epoch": final_epoch,
        "start_epoch": start_epoch,
        "model_type": _public_model_type(model_type_internal),
        "model_type_internal": model_type_internal,
        "m_eval_frames": m_eval_frames,
        "seq_len": seq_len,
        "burn_in_frames": burn_in_frames,
        "feature_dim": feature_dim,
        "feature_key": feature_key,
        "risk_key": risk_key,
        "target_mode": target_mode,
        "device": str(resolved_device),
        "grad_clip": grad_clip,
        "loss_type": normalized_loss,
        "huber_beta": huber_beta,
    }


__all__ = ["train_bdu_gru_model"]
