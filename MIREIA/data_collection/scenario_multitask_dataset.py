from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from MIREIA.config import Config
from MIREIA.data_collection.dataset_utils import (
    DEFAULT_IMAGE_SIZE,
    build_default_transform,
    load_jsonl_records,
    load_rgb_image,
    resolve_image_path,
)


@dataclass(frozen=True)
class ScenarioEnvironmentSource:
    name: str
    jsonl_path: str
    image_root: str
    day_night_label: int
    climate_label: str


def _camel_to_snake(token: str) -> str:
    token = token.strip().replace("-", "_")
    if not token:
        return "unknown"
    chunks = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", token)
    if not chunks:
        return token.lower()
    return "_".join(chunk.lower() for chunk in chunks)


def _normalize_weather_preset(preset_name: str) -> str:
    base = preset_name.strip()
    for suffix in ("Noon", "Sunset", "Night"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    # CARLA exposes both MidRain and MidRainy naming depending on preset.
    base = base.replace("Rainy", "Rain")
    return _camel_to_snake(base)


def _extract_weather_preset_from_scenario_name(scenario_name: str) -> str | None:
    parts = scenario_name.split("_")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return None


def infer_scenario_day_night_label(scenario_name: str, weather: str | dict) -> int:
    if isinstance(weather, dict):
        sun_altitude = weather.get("sun_altitude_angle")
        if sun_altitude is not None:
            try:
                return int(float(sun_altitude) < 0.0)
            except (TypeError, ValueError):
                pass

    lower_name = scenario_name.lower()
    if "_night" in lower_name or lower_name.endswith("night"):
        return 1
    if isinstance(weather, str) and "night" in weather.lower():
        return 1
    return 0


def infer_scenario_climate_label(scenario_name: str, weather: str | dict) -> str:
    preset_name: str | None = None
    if isinstance(weather, str) and weather.strip():
        preset_name = weather.strip()
    elif isinstance(weather, dict):
        preset_name = _extract_weather_preset_from_scenario_name(scenario_name)

    if not preset_name:
        preset_name = "CustomWeather"

    climate_label = _normalize_weather_preset(preset_name)

    if isinstance(weather, dict):
        fog_density = weather.get("fog_density", 0.0)
        try:
            fog_density_value = float(fog_density)
        except (TypeError, ValueError):
            fog_density_value = 0.0
        if fog_density_value >= 40.0:
            climate_label = f"{climate_label}_fog"

    return climate_label


class ScenarioEnvironmentDataset(Dataset):
    """Frame-level scenario dataset for multitask environment classification."""

    def __init__(
        self,
        split: str,
        scenarios_root: Optional[str] = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        transform: Optional[Callable] = None,
        expected_scenarios: Optional[int] = None,
        include_names: Optional[Iterable[str]] = None,
        exclude_names: Optional[Iterable[str]] = None,
        town10hd_token: str = "Town10HD",
        normalize_paths: bool = True,
        subset_ratio: Optional[float] = None,
        subset_seed: int = Config.RANDOM_SEED,
        subset_mode: str = "first",
        max_scenarios: Optional[int] = None,
        frame_subset_ratio: Optional[float] = None,
        frame_subset_seed: int = Config.RANDOM_SEED,
        frame_subset_mode: str = "random",
        climate_to_idx: Optional[Dict[str, int]] = None,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.scenarios_root = scenarios_root or Config.PATH_TO_SCENARIOS
        self.transform = transform or build_default_transform(image_size)
        self.normalize_paths = normalize_paths
        self.include_names = set(include_names or [])
        self.exclude_names = set(exclude_names or [])
        self.town10hd_token = town10hd_token

        self.subset_ratio = subset_ratio
        self.subset_seed = subset_seed
        self.subset_mode = subset_mode
        self.max_scenarios = max_scenarios

        self.frame_subset_ratio = frame_subset_ratio
        self.frame_subset_seed = frame_subset_seed
        self.frame_subset_mode = frame_subset_mode

        self._sources = self._discover_sources()
        if expected_scenarios and len(self._sources) != expected_scenarios:
            print(
                f"Warning: expected {expected_scenarios} scenarios but found {len(self._sources)} "
                f"under {self.scenarios_root}"
            )

        self._records: List[List[dict]] = [load_jsonl_records(src.jsonl_path) for src in self._sources]

        if climate_to_idx is None:
            climates = sorted({src.climate_label for src in self._sources})
            self._climate_to_idx = {label: idx for idx, label in enumerate(climates)}
        else:
            self._climate_to_idx = dict(climate_to_idx)
            missing = sorted(
                {src.climate_label for src in self._sources if src.climate_label not in self._climate_to_idx}
            )
            if missing:
                missing_repr = ", ".join(missing)
                raise ValueError(
                    "Scenario climates are missing from climate_to_idx mapping: "
                    f"{missing_repr}"
                )

        self._idx_to_climate = [
            label for label, _ in sorted(self._climate_to_idx.items(), key=lambda item: item[1])
        ]

        self._index: List[tuple[int, int]] = self._build_index(self._records)
        self._index = self._apply_frame_subset(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_idx, frame_idx = self._index[index]
        source = self._sources[source_idx]
        record = self._records[source_idx][frame_idx]

        rel_path = str(record.get("rgb_image_path", "")).strip()
        if not rel_path:
            raise ValueError(
                f"Missing rgb_image_path for scenario {source.name} in {source.jsonl_path}"
            )

        image_path = resolve_image_path(source.image_root, rel_path, self.normalize_paths)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Dashcam image not found: {image_path}")

        image = load_rgb_image(image_path, self.transform)
        day_night_label = torch.tensor(source.day_night_label, dtype=torch.long)
        climate_label = torch.tensor(self._climate_to_idx[source.climate_label], dtype=torch.long)
        return image, day_night_label, climate_label

    @property
    def climate_to_idx(self) -> Dict[str, int]:
        return dict(self._climate_to_idx)

    @property
    def idx_to_climate(self) -> List[str]:
        return list(self._idx_to_climate)

    def _discover_sources(self) -> List[ScenarioEnvironmentSource]:
        if not os.path.isdir(self.scenarios_root):
            raise FileNotFoundError(f"Scenarios root not found: {self.scenarios_root}")

        candidates: List[ScenarioEnvironmentSource] = []
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
            scenario_json_path = os.path.join(scenario_dir, "scenario.json")
            if not os.path.isfile(jsonl_path) or not os.path.isfile(scenario_json_path):
                continue

            is_val = self.town10hd_token in entry
            if self.split == "val" and not is_val:
                continue
            if self.split == "train" and is_val:
                continue

            with open(scenario_json_path, "r", encoding="utf-8") as handle:
                scenario_meta = json.load(handle)

            weather = scenario_meta.get("weather", "")
            day_night_label = infer_scenario_day_night_label(entry, weather)
            climate_label = infer_scenario_climate_label(entry, weather)

            candidates.append(
                ScenarioEnvironmentSource(
                    name=entry,
                    jsonl_path=jsonl_path,
                    image_root=os.path.dirname(jsonl_path),
                    day_night_label=day_night_label,
                    climate_label=climate_label,
                )
            )

        return self._apply_scenario_subset(candidates)

    def _apply_scenario_subset(
        self, candidates: List[ScenarioEnvironmentSource]
    ) -> List[ScenarioEnvironmentSource]:
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

    def _select_subset(
        self, candidates: List[ScenarioEnvironmentSource], count: int
    ) -> List[ScenarioEnvironmentSource]:
        if count >= len(candidates):
            return candidates
        if self.subset_mode == "random":
            rng = random.Random(self.subset_seed)
            return rng.sample(candidates, count)
        if self.subset_mode != "first":
            raise ValueError("subset_mode must be 'first' or 'random'")
        return candidates[:count]

    def _apply_frame_subset(self, index: List[tuple[int, int]]) -> List[tuple[int, int]]:
        if self.frame_subset_ratio is None:
            return index
        if not (0.0 < self.frame_subset_ratio <= 1.0):
            raise ValueError("frame_subset_ratio must be in (0, 1]")

        count = max(1, int(len(index) * self.frame_subset_ratio))
        if count >= len(index):
            return index

        if self.frame_subset_mode == "random":
            seed = self.frame_subset_seed + (1 if self.split == "val" else 0)
            rng = random.Random(seed)
            return rng.sample(index, count)
        if self.frame_subset_mode != "first":
            raise ValueError("frame_subset_mode must be 'first' or 'random'")
        return index[:count]

    @staticmethod
    def _build_index(records: List[List[dict]]) -> List[tuple[int, int]]:
        index: List[tuple[int, int]] = []
        for source_idx, scenario_records in enumerate(records):
            for frame_idx in range(len(scenario_records)):
                index.append((source_idx, frame_idx))
        return index


def create_environment_dataloaders(
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
    transform: Optional[Callable] = None,
    scenarios_root: Optional[str] = None,
    climate_to_idx: Optional[Dict[str, int]] = None,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader, Dict[str, int]]:
    if num_workers is None:
        cpu_count = os.cpu_count() or 0
        num_workers = min(8, max(0, cpu_count - 1))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_dataset = ScenarioEnvironmentDataset(
        split="train",
        scenarios_root=scenarios_root,
        transform=transform,
        climate_to_idx=climate_to_idx,
        **dataset_kwargs,
    )
    val_dataset = ScenarioEnvironmentDataset(
        split="val",
        scenarios_root=scenarios_root,
        transform=transform,
        climate_to_idx=train_dataset.climate_to_idx,
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

    train_loader = DataLoader(train_dataset, shuffle=shuffle, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_dataset.climate_to_idx
