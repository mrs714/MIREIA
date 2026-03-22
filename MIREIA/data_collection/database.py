from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from MIREIA.config import Config


@dataclass(frozen=True)
class ScenarioSource:
	name: str
	jsonl_path: str
	image_root: str


class ScenarioSequenceDataset(Dataset):
	def __init__(
		self,
		seq_len: int,
		split: str,
		scenarios_root: Optional[str] = None,
		transform: Optional[Callable] = None,
		target_mode: str = "last",
		risk_key: str = "ground_truth_risk",
		expected_scenarios: int = 54,
		include_names: Optional[Iterable[str]] = None,
		exclude_names: Optional[Iterable[str]] = None,
		town10hd_token: str = "Town10HD",
		normalize_paths: bool = True,
	):
		if seq_len <= 0:
			raise ValueError("seq_len must be > 0")
		if split not in {"train", "val"}:
			raise ValueError("split must be 'train' or 'val'")

		self.seq_len = seq_len
		self.split = split
		self.risk_key = risk_key
		self.target_mode = target_mode
		self.normalize_paths = normalize_paths
		self.town10hd_token = town10hd_token

		self.scenarios_root = scenarios_root or Config.PATH_TO_SCENARIOS
		self.include_names = set(include_names or [])
		self.exclude_names = set(exclude_names or [])

		if transform is None:
			self.transform = transforms.Compose(
				[
					transforms.Resize((512, 512)),
					transforms.ToTensor(),
				]
			)
		else:
			self.transform = transform

		sources = self._discover_scenarios()
		if expected_scenarios and len(sources) != expected_scenarios:
			print(
				f"Warning: expected {expected_scenarios} scenarios but found {len(sources)} under {self.scenarios_root}"
			)

		self._sources: List[ScenarioSource] = sources
		self._records: List[List[dict]] = [self._load_records(s.jsonl_path) for s in sources]
		self._index: List[tuple[int, int]] = self._build_index(self._records, seq_len)

	def __len__(self) -> int:
		return len(self._index)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		scenario_idx, start = self._index[index]
		records = self._records[scenario_idx]
		window = records[start : start + self.seq_len]

		images = [self._load_image(self._sources[scenario_idx], rec) for rec in window]
		seq_tensor = torch.stack(images, dim=0)
		target = self._build_target(window)
		return seq_tensor, target

	def _discover_scenarios(self) -> List[ScenarioSource]:
		if not os.path.isdir(self.scenarios_root):
			raise FileNotFoundError(f"Scenarios root not found: {self.scenarios_root}")

		candidates: List[ScenarioSource] = []
		for entry in sorted(os.listdir(self.scenarios_root)):
			scenario_dir = os.path.join(self.scenarios_root, entry)
			if not os.path.isdir(scenario_dir):
				continue
			if entry in {"videos", "__pycache__"}:
				continue
			if self.include_names and entry not in self.include_names:
				continue
			if self.exclude_names and entry in self.exclude_names:
				continue

			jsonl_path = os.path.join(scenario_dir, "dataset.jsonl")
			if not os.path.isfile(jsonl_path):
				continue

			is_val = self.town10hd_token in entry
			if self.split == "val" and not is_val:
				continue
			if self.split == "train" and is_val:
				continue

			candidates.append(
				ScenarioSource(
					name=entry,
					jsonl_path=jsonl_path,
					image_root=os.path.dirname(jsonl_path),
				)
			)

		return candidates

	def _build_target(self, window: Sequence[dict]) -> torch.Tensor:
		if self.target_mode == "mean":
			value = sum(rec[self.risk_key] for rec in window) / len(window)
		else:
			value = window[-1][self.risk_key]
		return torch.tensor([value], dtype=torch.float32)

	def _load_image(self, source: ScenarioSource, record: dict) -> torch.Tensor:
		rel_path = record.get("rgb_image_path", "")
		if not rel_path:
			raise ValueError(
				f"Missing rgb_image_path for scenario {source.name} in {source.jsonl_path}"
			)

		full_path = self._resolve_path(source.image_root, rel_path)
		if not os.path.isfile(full_path):
			raise FileNotFoundError(f"Dashcam image not found: {full_path}")

		with Image.open(full_path) as img:
			img = img.convert("RGB")
			return self.transform(img)

	def _resolve_path(self, image_root: str, rel_path: str) -> str:
		if os.path.isabs(rel_path):
			path = rel_path
		else:
			path = os.path.join(image_root, rel_path)
		return os.path.normpath(path) if self.normalize_paths else path

	@staticmethod
	def _build_index(records: List[List[dict]], seq_len: int) -> List[tuple[int, int]]:
		index: List[tuple[int, int]] = []
		for scenario_idx, scenario_records in enumerate(records):
			max_start = len(scenario_records) - seq_len
			if max_start < 0:
				continue
			for start in range(max_start + 1):
				index.append((scenario_idx, start))
		return index

	@staticmethod
	def _load_records(jsonl_path: str) -> List[dict]:
		records: List[dict] = []
		with open(jsonl_path, "r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line:
					continue
				records.append(json.loads(line))
		return records


def create_e2e_dataloaders(
	seq_len: int,
	batch_size: int = 4,
	num_workers: int = 0,
	shuffle: bool = True,
	pin_memory: Optional[bool] = None,
	transform: Optional[Callable] = None,
	scenarios_root: Optional[str] = None,
	**dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
	if pin_memory is None:
		pin_memory = torch.cuda.is_available()

	train_dataset = ScenarioSequenceDataset(
		seq_len=seq_len,
		split="train",
		scenarios_root=scenarios_root,
		transform=transform,
		**dataset_kwargs,
	)
	val_dataset = ScenarioSequenceDataset(
		seq_len=seq_len,
		split="val",
		scenarios_root=scenarios_root,
		transform=transform,
		**dataset_kwargs,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	return train_loader, val_loader
