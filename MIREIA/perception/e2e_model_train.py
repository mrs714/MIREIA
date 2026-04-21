from __future__ import annotations

import os

import torch
from torch import nn, optim
from torchvision import transforms

from MIREIA.config import Config
from MIREIA.perception.e2e_model import E2EModelConfig, E2ERiskPredictor, Seq2SeqRiskPredictor
from MIREIA.perception.training_utils import (
	build_default_train_val_include_names,
	build_scenario_dataloaders,
	default_val_scenario_tokens_csv,
	load_checkpoint,
	save_checkpoint,
	train_model,
)


def _normalize_model_type(model_type: str) -> str:
	normalized = str(model_type).strip().lower()
	if normalized in {"e2e", "seq2seq"}:
		return "seq2seq"
	if normalized == "single":
		return "single"
	raise ValueError("model_type must be 'e2e' (alias: 'seq2seq') or 'single'")


def _public_model_type(model_type: str) -> str:
	return "e2e" if model_type == "seq2seq" else model_type


def train_e2e_model(
	resume_epochs: int = 1,
	model_type: str = "e2e",
	seq_len: int = Config.INFERENCE_SEQUENCE_LENGTH,
	burn_in_frames: int = Config.INFERENCE_BURN_IN_FRAMES,
	m_eval_frames: int = Config.INFERENCE_EVAL_FRAMES,
	batch_size: int = 8,
	num_workers: int = 8,
	prefetch_factor: int = 4,
	pin_memory: bool | None = None,
	persistent_workers: bool | None = None,
	learning_rate: float = 1e-4,
	use_amp: bool = True,
	partition_mode: str = "scenario",
	val_scenario_tokens: str | list[str] | None = default_val_scenario_tokens_csv(),
	frame_train_ratio: float = 0.7,
	include_names: list[str] | None = None,
	exclude_names: list[str] | None = None,
	window_subset_ratio: float | None = None,
	window_subset_mode: str = "random",
	window_subset_seed: int = Config.RANDOM_SEED,
	checkpoint_path: str = "",
	checkpoint_name: str = "e2e_risk_checkpoint.pt",
	device: str = "",
	transform=None,
	grad_clip: float | None = None,
	grad_accum_steps: int = 1,
	prefer_labeled_jsonl: bool = False,
	crop_bbox_key: str | None = None,
) -> dict[str, object]:
	"""Train or resume the e2e risk model with runtime-aligned defaults."""
	if resume_epochs <= 0:
		raise ValueError("resume_epochs must be > 0")
	if burn_in_frames + m_eval_frames != seq_len:
		raise ValueError(
			f"Invalid temporal setup: burn_in({burn_in_frames}) + "
			f"eval({m_eval_frames}) != seq_len({seq_len})"
		)

	model_type_internal = _normalize_model_type(model_type)

	if pin_memory is None:
		pin_memory = torch.cuda.is_available()
	if persistent_workers is None:
		persistent_workers = num_workers > 0

	if transform is None:
		transform = transforms.Compose([transforms.ToTensor()])

	if partition_mode == "scenario" and include_names is None:
		include_names = build_default_train_val_include_names(scenarios_root=Config.PATH_TO_SCENARIOS)

	train_loader, val_loader, target_mode = build_scenario_dataloaders(
		seq_len=seq_len,
		batch_size=batch_size,
		num_workers=num_workers,
		prefetch_factor=prefetch_factor,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		transform=transform,
		partition_mode=partition_mode,
		val_scenario_tokens=val_scenario_tokens,
		frame_train_ratio=frame_train_ratio,
		include_names=include_names,
		exclude_names=exclude_names,
		model_type=model_type_internal,
		m_eval_frames=m_eval_frames,
		window_subset_ratio=window_subset_ratio,
		window_subset_mode=window_subset_mode,
		window_subset_seed=window_subset_seed,
		prefer_labeled_jsonl=prefer_labeled_jsonl,
		crop_bbox_key=crop_bbox_key,
	)

	print(f"Temporal config: seq_len={seq_len}, burn_in={burn_in_frames}, eval={m_eval_frames}")
	print(
		f"DataLoader workers: num_workers={num_workers}, "
		f"prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers}"
	)
	print(
		f"Input image mode: {'full-frame' if crop_bbox_key is None else f'cropped via {crop_bbox_key}'}"
	)
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

	model = (
		Seq2SeqRiskPredictor(E2EModelConfig())
		if model_type_internal == "seq2seq"
		else E2ERiskPredictor(E2EModelConfig())
	).to(resolved_device)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.MSELoss()

	if not checkpoint_path:
		if checkpoint_name:
			checkpoint_path = os.path.join(Config.PATH_TO_MODELS, checkpoint_name)
		else:
			default_name = "e2e_risk_checkpoint.pt" if model_type_internal == "seq2seq" else "single_checkpoint.pt"
			checkpoint_path = os.path.join(Config.PATH_TO_MODELS, default_name)

	start_epoch = 1
	history = {"train_loss": [], "val_loss": []}

	if os.path.exists(checkpoint_path):
		ckpt = load_checkpoint(checkpoint_path, model, optimizer=optimizer, device=resolved_device)
		history = ckpt.get("history", history)
		start_epoch = int(ckpt.get("epoch", 0)) + 1
		loaded_model_type = ckpt.get("model_type_internal", ckpt.get("model_type", model_type_internal))
		model_type_internal = _normalize_model_type(str(loaded_model_type))
		m_eval_frames = int(ckpt.get("m_eval_frames", m_eval_frames))
		target_mode = ckpt.get("target_mode", target_mode)
		use_amp = bool(ckpt.get("use_amp", use_amp))
		print(f"Resuming from {checkpoint_path} at epoch {start_epoch}")
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
			"m_eval_frames": m_eval_frames,
			"seq_len": seq_len,
			"target_mode": target_mode,
			"use_amp": use_amp,
			"crop_bbox_key": crop_bbox_key,
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
		"target_mode": target_mode,
		"device": str(resolved_device),
	}


__all__ = ["train_e2e_model"]
