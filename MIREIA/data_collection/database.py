from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from MIREIA.config import Config
from MIREIA.data_collection.dataset_utils import (
	BaseSequenceDataset,
	DEFAULT_IMAGE_SIZE,
	compute_frame_split_boundary,
	load_jsonl_records,
	normalize_frame_train_ratio,
	normalize_partition_mode,
	normalize_validation_tokens,
	resolve_image_path,
	scenario_is_validation_split,
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
		partition_mode: str = "scenario",
		val_scenario_tokens: str | Iterable[str] | None = None,
		town10hd_token: str = "Town10HD",
		frame_train_ratio: float = 0.7,
		normalize_paths: bool = True,
		subset_ratio: Optional[float] = None,
		subset_seed: int = Config.RANDOM_SEED,
		subset_mode: str = "first",
		max_scenarios: Optional[int] = None,
		window_subset_ratio: Optional[float] = None,
		window_subset_seed: int = Config.RANDOM_SEED,
		window_subset_mode: str = "random",
		prefer_labeled_jsonl: bool = True,
		labeled_jsonl_name: str = "dataset_labeled.jsonl",
		dataset_jsonl_name: str = "dataset.jsonl",
		crop_bbox_key: str | None = "crop_bbox_xyxy",
		manual_crop_bbox: Sequence[float] | None = None,
	):
		super().__init__(
			seq_len=seq_len,
			transform=transform,
			image_size=image_size,
			target_mode=target_mode,
			risk_key=risk_key,
			crop_bbox_key=crop_bbox_key,
			manual_crop_bbox=manual_crop_bbox,
		)
		if split not in {"train", "val"}:
			raise ValueError("split must be 'train' or 'val'")

		self.split = split
		self.normalize_paths = normalize_paths
		self.partition_mode = normalize_partition_mode(partition_mode)
		self.val_scenario_tokens = normalize_validation_tokens(
			val_scenario_tokens,
			fallback_token=town10hd_token,
		)
		self.frame_train_ratio = normalize_frame_train_ratio(frame_train_ratio)
		self.subset_ratio = subset_ratio
		self.subset_seed = subset_seed
		self.subset_mode = subset_mode
		self.max_scenarios = max_scenarios
		self.window_subset_ratio = window_subset_ratio
		self.window_subset_seed = window_subset_seed
		self.window_subset_mode = window_subset_mode
		self.prefer_labeled_jsonl = bool(prefer_labeled_jsonl)
		self.labeled_jsonl_name = str(labeled_jsonl_name)
		self.dataset_jsonl_name = str(dataset_jsonl_name)

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
		self._index: List[tuple[int, int]] = self._build_index_for_split(self._records, seq_len)
		self._index = self._apply_window_subset(self._index)

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

			labeled_jsonl_path = os.path.join(scenario_dir, self.labeled_jsonl_name)
			default_jsonl_path = os.path.join(scenario_dir, self.dataset_jsonl_name)

			if self.prefer_labeled_jsonl and os.path.isfile(labeled_jsonl_path):
				jsonl_path = labeled_jsonl_path
			elif os.path.isfile(default_jsonl_path):
				jsonl_path = default_jsonl_path
			else:
				continue

			if self.partition_mode == "scenario":
				is_val = scenario_is_validation_split(entry, self.val_scenario_tokens)
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

		candidates = self._apply_subset(candidates)
		return candidates

	def _build_index_for_split(
		self,
		records: List[List[dict]],
		seq_len: int,
	) -> List[tuple[int, int]]:
		if self.partition_mode != "frame":
			return self._build_index(records, seq_len)

		index: List[tuple[int, int]] = []
		for scenario_idx, scenario_records in enumerate(records):
			scenario_len = len(scenario_records)
			max_start = scenario_len - seq_len
			if max_start < 0:
				continue

			split_boundary = compute_frame_split_boundary(scenario_len, self.frame_train_ratio)
			if self.split == "train":
				start_min = 0
				start_max = split_boundary - seq_len
			else:
				start_min = split_boundary
				start_max = max_start

			if start_max < start_min:
				continue

			for start in range(start_min, start_max + 1):
				index.append((scenario_idx, start))
		return index

	def _apply_subset(self, candidates: List[ScenarioSource]) -> List[ScenarioSource]:
		if self.subset_ratio is not None:
			if not (0.0 < self.subset_ratio <= 1.0):
				raise ValueError("subset_ratio must be in (0, 1]")
			count = max(1, int(len(candidates) * self.subset_ratio))
			candidates = self._select_subset(candidates, count)

		if self.max_scenarios is not None:
			if self.max_scenarios <= 0:
				raise ValueError("max_scenarios must be > 0")
			candidates = candidates[: self.max_scenarios]

		return candidates

	def _select_subset(self, candidates: List[ScenarioSource], count: int) -> List[ScenarioSource]:
		if count >= len(candidates):
			return candidates
		if self.subset_mode == "random":
			import random

			rng = random.Random(self.subset_seed)
			return rng.sample(candidates, count)
		if self.subset_mode != "first":
			raise ValueError("subset_mode must be 'first' or 'random'")
		return candidates[:count]

	def _apply_window_subset(
		self, index: List[tuple[int, int]]
	) -> List[tuple[int, int]]:
		if self.window_subset_ratio is None:
			return index
		if not (0.0 < self.window_subset_ratio <= 1.0):
			raise ValueError("window_subset_ratio must be in (0, 1]")
		count = max(1, int(len(index) * self.window_subset_ratio))
		if count >= len(index):
			return index

		if self.window_subset_mode == "random":
			import random

			seed = self.window_subset_seed + (1 if self.split == "val" else 0)
			rng = random.Random(seed)
			return rng.sample(index, count)
		if self.window_subset_mode != "first":
			raise ValueError("window_subset_mode must be 'first' or 'random'")
		return index[:count]

	def _load_record_image(self, source: ScenarioSource, record: dict) -> torch.Tensor:
		rel_path = record.get("rgb_image_path", "")
		if not rel_path:
			raise ValueError(
				f"Missing rgb_image_path for scenario {source.name} in {source.jsonl_path}"
			)

		full_path = resolve_image_path(source.image_root, rel_path, self.normalize_paths)
		if not os.path.isfile(full_path):
			raise FileNotFoundError(f"Dashcam image not found: {full_path}")

		crop_bbox = self._resolve_record_crop_bbox(record)
		return self._load_image_tensor(full_path, crop_bbox_xyxy=crop_bbox)

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
