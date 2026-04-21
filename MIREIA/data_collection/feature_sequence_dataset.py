from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from MIREIA.config import Config
from MIREIA.data_collection.dataset_utils import (
    DEFAULT_VAL_SCENARIO_TOKENS_CSV,
    compute_frame_split_boundary,
    load_jsonl_records,
    normalize_frame_train_ratio,
    normalize_partition_mode,
    normalize_validation_tokens,
    scenario_is_validation_split,
)


@dataclass(frozen=True)
class ScenarioFeatureSource:
    name: str
    jsonl_path: str


class ScenarioFeatureSequenceDataset(Dataset):
    """
    Sequence dataset that reads per-frame feature vectors from scenario JSONL files.

    Expected record keys:
    - feature_vector_32 (default): list[float] with length feature_dim
    - ground_truth_risk (default): scalar float target

        Notes:
        - Legacy labeled datasets may contain unnormalized depth/flow/change features.
            This dataset applies a defensive sanitization pass to keep features in stable
            ranges for recurrent models.
    """

    def __init__(
        self,
        seq_len: int,
        split: str,
        scenarios_root: Optional[str] = None,
        target_mode: str = "sequence",
        risk_key: str = "ground_truth_risk",
        feature_key: str = "feature_vector_32",
        feature_dim: int = 32,
        fallback_to_zeros: bool = True,
        expected_scenarios: Optional[int] = None,
        include_names: Optional[Iterable[str]] = None,
        exclude_names: Optional[Iterable[str]] = None,
        partition_mode: str = "scenario",
        val_scenario_tokens: str | Iterable[str] | None = None,
        town10hd_token: str = DEFAULT_VAL_SCENARIO_TOKENS_CSV,
        frame_train_ratio: float = 0.7,
        subset_ratio: Optional[float] = None,
        subset_seed: int = Config.RANDOM_SEED,
        subset_mode: str = "first",
        max_scenarios: Optional[int] = None,
        window_subset_ratio: Optional[float] = None,
        window_subset_seed: int = Config.RANDOM_SEED,
        window_subset_mode: str = "random",
        prefer_labeled_jsonl: bool = True,
        fallback_to_dataset_jsonl: bool = False,
        labeled_jsonl_name: str = "dataset_labeled.jsonl",
        dataset_jsonl_name: str = "dataset.jsonl",
        sanitize_features: bool = True,
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.seq_len = int(seq_len)
        self.split = split
        self.target_mode = str(target_mode)
        self.risk_key = str(risk_key)
        self.feature_key = str(feature_key)
        self.feature_dim = int(feature_dim)
        self.fallback_to_zeros = bool(fallback_to_zeros)

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
        self.fallback_to_dataset_jsonl = bool(fallback_to_dataset_jsonl)
        self.labeled_jsonl_name = str(labeled_jsonl_name)
        self.dataset_jsonl_name = str(dataset_jsonl_name)
        self.sanitize_features = bool(sanitize_features)

        self.scenarios_root = scenarios_root or Config.PATH_TO_SCENARIOS
        self.include_names = set(include_names or [])
        self.exclude_names = set(exclude_names or [])

        sources = self._discover_scenarios()
        if expected_scenarios is not None and len(sources) != expected_scenarios:
            print(
                f"Warning ({self.split}): expected {expected_scenarios} scenarios "
                f"but found {len(sources)} under {self.scenarios_root}"
            )

        self._sources: List[ScenarioFeatureSource] = sources
        records_by_scenario: List[List[dict]] = [load_jsonl_records(s.jsonl_path) for s in sources]
        self._scenario_features, self._scenario_risks = self._materialize_records(records_by_scenario)
        lengths = [int(feats.shape[0]) for feats in self._scenario_features]
        self._index: List[tuple[int, int]] = self._build_index_for_split(lengths, self.seq_len)
        self._index = self._apply_window_subset(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scenario_idx, start = self._index[index]
        seq_tensor = self._scenario_features[scenario_idx][start : start + self.seq_len]
        risk_window = self._scenario_risks[scenario_idx][start : start + self.seq_len]
        target = self._build_target_from_risk_window(risk_window)
        return seq_tensor, target

    def _materialize_records(
        self,
        records_by_scenario: List[List[dict]],
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        scenario_features: List[torch.Tensor] = []
        scenario_risks: List[torch.Tensor] = []

        for scenario_idx, records in enumerate(records_by_scenario):
            scenario_name = self._sources[scenario_idx].name

            features = [self._feature_from_record(rec, scenario_name) for rec in records]
            if features:
                feature_tensor = torch.stack(features, dim=0)
            else:
                feature_tensor = torch.empty((0, self.feature_dim), dtype=torch.float32)

            risks: List[float] = []
            for rec in records:
                raw = rec.get(self.risk_key, 0.0)
                try:
                    risks.append(float(raw))
                except (TypeError, ValueError):
                    risks.append(0.0)
            risk_tensor = torch.tensor(risks, dtype=torch.float32)

            scenario_features.append(feature_tensor)
            scenario_risks.append(risk_tensor)

        return scenario_features, scenario_risks

    def _discover_scenarios(self) -> List[ScenarioFeatureSource]:
        if not os.path.isdir(self.scenarios_root):
            raise FileNotFoundError(f"Scenarios root not found: {self.scenarios_root}")

        candidates: List[ScenarioFeatureSource] = []
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
            elif self.fallback_to_dataset_jsonl and os.path.isfile(default_jsonl_path):
                jsonl_path = default_jsonl_path
            else:
                continue

            if self.partition_mode == "scenario":
                is_val = scenario_is_validation_split(entry, self.val_scenario_tokens)
                if self.split == "val" and not is_val:
                    continue
                if self.split == "train" and is_val:
                    continue

            candidates.append(ScenarioFeatureSource(name=entry, jsonl_path=jsonl_path))

        candidates = self._apply_subset(candidates)
        return candidates

    def _apply_subset(self, candidates: List[ScenarioFeatureSource]) -> List[ScenarioFeatureSource]:
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

    def _select_subset(self, candidates: List[ScenarioFeatureSource], count: int) -> List[ScenarioFeatureSource]:
        if count >= len(candidates):
            return candidates
        if self.subset_mode == "random":
            import random

            rng = random.Random(self.subset_seed)
            return rng.sample(candidates, count)
        if self.subset_mode != "first":
            raise ValueError("subset_mode must be 'first' or 'random'")
        return candidates[:count]

    def _apply_window_subset(self, index: List[tuple[int, int]]) -> List[tuple[int, int]]:
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

    def _feature_from_record(self, record: dict, scenario_name: str) -> torch.Tensor:
        raw = record.get(self.feature_key, None)
        if isinstance(raw, (list, tuple)) and len(raw) == self.feature_dim:
            try:
                values = [float(v) for v in raw]
                vec = torch.tensor(values, dtype=torch.float32)
                if self.sanitize_features:
                    vec = self._sanitize_feature_vector(vec)
                return vec
            except (TypeError, ValueError):
                pass

        if self.fallback_to_zeros:
            return torch.zeros(self.feature_dim, dtype=torch.float32)

        frame_id = record.get("frame_id", "?")
        raise ValueError(
            f"Invalid {self.feature_key} in scenario={scenario_name}, frame_id={frame_id}. "
            f"Expected list/tuple with length={self.feature_dim}."
        )

    def _sanitize_feature_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """Keep legacy and outlier-heavy vectors numerically safe for GRU training."""
        if vec.numel() != self.feature_dim:
            return vec

        out = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).clone()

        # 1) Count and size-style features.
        out[0] = torch.clamp(out[0], min=0.0, max=100.0)
        out[1] = torch.clamp(out[1], min=0.0, max=1.0)
        out[2] = torch.clamp(out[2], min=0.0, max=1.0)

        # 2) Legacy depth was stored in meters [0, 100]. Normalize to [0, 1].
        if float(out[3]) > 1.5:
            out[3] = torch.clamp(out[3] / 100.0, min=0.0, max=1.0)
        else:
            out[3] = torch.clamp(out[3], min=0.0, max=1.0)
        if float(out[4]) > 1.5:
            out[4] = torch.clamp(out[4] / 100.0, min=0.0, max=1.0)
        else:
            out[4] = torch.clamp(out[4], min=0.0, max=1.0)

        # 3) Ratio features are valid but can spike with tiny denominators.
        out[5] = torch.clamp(out[5], min=0.0, max=5.0)
        out[6] = torch.clamp(out[6], min=0.0, max=5.0)
        out[7] = torch.clamp(out[7], min=0.0, max=5.0)
        out[8] = torch.clamp(out[8], min=0.0, max=5.0)

        # 4) Legacy flow may be raw pixels (roughly up to image width/height).
        if float(torch.abs(out[9])) > 2.0:
            out[9] = out[9] / 512.0
        if float(torch.abs(out[10])) > 2.0:
            out[10] = out[10] / 512.0
        out[9] = torch.clamp(out[9], min=-1.0, max=1.0)
        out[10] = torch.clamp(out[10], min=-1.0, max=1.0)

        # 5) Threat channels are non-negative by definition; clamp outliers.
        for i in range(11, 17):
            out[i] = torch.clamp(out[i], min=0.0, max=10.0)

        # 6) Probability-like channels.
        for i in range(17, 32):
            out[i] = torch.clamp(out[i], min=0.0, max=1.0)

        return out

    def _build_target_from_risk_window(self, risk_window: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "sequence":
            return risk_window.unsqueeze(1)
        if self.target_mode == "mean":
            value = float(risk_window.mean().item()) if risk_window.numel() else 0.0
            return torch.tensor([value], dtype=torch.float32)
        value = float(risk_window[-1].item()) if risk_window.numel() else 0.0
        return torch.tensor([value], dtype=torch.float32)

    def _build_index_for_split(self, lengths: list[int], seq_len: int) -> list[tuple[int, int]]:
        if self.partition_mode != "frame":
            return self._build_index_from_lengths(lengths, seq_len)

        index: list[tuple[int, int]] = []
        for scenario_idx, scenario_len in enumerate(lengths):
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

    @staticmethod
    def _build_index_from_lengths(lengths: list[int], seq_len: int) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for scenario_idx, scenario_len in enumerate(lengths):
            max_start = scenario_len - seq_len
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                index.append((scenario_idx, start))
        return index


def create_feature_sequence_dataloaders(
    seq_len: int,
    batch_size: int = 8,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
    scenarios_root: Optional[str] = None,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    if num_workers is None:
        if os.name == "nt":
            # On Windows, process spawn can dominate time when dataset state is large.
            num_workers = 0
        else:
            cpu_count = os.cpu_count() or 0
            num_workers = min(8, max(0, cpu_count - 1))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_dataset = ScenarioFeatureSequenceDataset(
        seq_len=seq_len,
        split="train",
        scenarios_root=scenarios_root,
        **dataset_kwargs,
    )
    val_dataset = ScenarioFeatureSequenceDataset(
        seq_len=seq_len,
        split="val",
        scenarios_root=scenarios_root,
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


__all__ = [
    "ScenarioFeatureSource",
    "ScenarioFeatureSequenceDataset",
    "create_feature_sequence_dataloaders",
]
