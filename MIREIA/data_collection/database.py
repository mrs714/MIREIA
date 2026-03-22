from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from MIREIA.config import Config
from MIREIA.data_collection.dataset_utils import (
	BaseSequenceDataset,
	DEFAULT_IMAGE_SIZE,
	load_jsonl_records,
	resolve_image_path,
)


@dataclass(frozen=True)
class ScenarioSource:
	name: str
	jsonl_path: str
	image_root: str


class ScenarioSequenceDataset(BaseSequenceDataset):
	def __init__(
		self,
		seq_len: int,
		split: str,
		scenarios_root: Optional[str] = None,
		image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
		transform: Optional[Callable] = None,
		target_mode: str = "last",
		risk_key: str = "ground_truth_risk",
		expected_scenarios: int = 54,
		include_names: Optional[Iterable[str]] = None,
		exclude_names: Optional[Iterable[str]] = None,
		town10hd_token: str = "Town10HD",
		normalize_paths: bool = True,
	):
		super().__init__(
			seq_len=seq_len,
			transform=transform,
			image_size=image_size,
			target_mode=target_mode,
			risk_key=risk_key,
		)
		if split not in {"train", "val"}:
			raise ValueError("split must be 'train' or 'val'")

		self.split = split
		self.normalize_paths = normalize_paths
		self.town10hd_token = town10hd_token

		self.scenarios_root = scenarios_root or Config.PATH_TO_SCENARIOS
		self.include_names = set(include_names or [])
		self.exclude_names = set(exclude_names or [])

		sources = self._discover_scenarios()
		if expected_scenarios and len(sources) != expected_scenarios:
			print(
				f"Warning: expected {expected_scenarios} scenarios but found {len(sources)} under {self.scenarios_root}"
			)

		self._sources: List[ScenarioSource] = sources
		self._records: List[List[dict]] = [load_jsonl_records(s.jsonl_path) for s in sources]
		self._index: List[tuple[int, int]] = self._build_index(self._records, seq_len)

	def __len__(self) -> int:
		return len(self._index)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		scenario_idx, start = self._index[index]
		records = self._records[scenario_idx]
		window = records[start : start + self.seq_len]

		images = [self._load_record_image(self._sources[scenario_idx], rec) for rec in window]
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

	def _load_record_image(self, source: ScenarioSource, record: dict) -> torch.Tensor:
		rel_path = record.get("rgb_image_path", "")
		if not rel_path:
			raise ValueError(
				f"Missing rgb_image_path for scenario {source.name} in {source.jsonl_path}"
			)

		full_path = resolve_image_path(source.image_root, rel_path, self.normalize_paths)
		if not os.path.isfile(full_path):
			raise FileNotFoundError(f"Dashcam image not found: {full_path}")

		return self._load_image_tensor(full_path)

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


def create_scenario_dataloaders(
	seq_len: int,
	batch_size: int = 4,
	num_workers: Optional[int] = None,
	shuffle: bool = True,
	pin_memory: Optional[bool] = None,
	prefetch_factor: int = 2,
	persistent_workers: Optional[bool] = None,
	transform: Optional[Callable] = None,
	scenarios_root: Optional[str] = None,
	**dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
	if num_workers is None:
		cpu_count = os.cpu_count() or 0
		num_workers = min(8, max(0, cpu_count - 1))
	if pin_memory is None:
		pin_memory = torch.cuda.is_available()
	if persistent_workers is None:
		persistent_workers = num_workers > 0

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

	loader_kwargs = {
		"batch_size": batch_size,
		"num_workers": num_workers,
		"pin_memory": pin_memory,
	}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = prefetch_factor
		loader_kwargs["persistent_workers"] = persistent_workers

	train_loader = DataLoader(
		train_dataset,
		shuffle=shuffle,
		**loader_kwargs,
	)
	val_loader = DataLoader(
		val_dataset,
		shuffle=False,
		**loader_kwargs,
	)
	return train_loader, val_loader
