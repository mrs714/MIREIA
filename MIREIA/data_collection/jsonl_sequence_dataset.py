from __future__ import annotations

import os
from typing import Callable, List, Sequence

import torch

from MIREIA.data_collection.dataset_utils import (
    BaseSequenceDataset,
    load_jsonl_records,
    resolve_image_path,
)


class JsonlSequenceDataset(BaseSequenceDataset):
    def __init__(
        self,
        jsonl_path: str,
        seq_len: int,
        image_root: str | None = None,
        image_size: tuple[int, int] = (512, 512),
        transform: Callable | None = None,
        target_mode: str = "last",
        risk_key: str = "ground_truth_risk",
        normalize_paths: bool = True,
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
        self.jsonl_path = jsonl_path
        self.image_root = image_root or os.path.dirname(jsonl_path)
        self.normalize_paths = normalize_paths
        self.records = load_jsonl_records(jsonl_path)

    def __len__(self) -> int:
        return max(0, len(self.records) - self.seq_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.records[index : index + self.seq_len]
        images = [self._load_record_image(rec) for rec in window]
        seq_tensor = torch.stack(images, dim=0)
        target = self._build_target(window)
        return seq_tensor, target

    def _load_record_image(self, record: dict) -> torch.Tensor:
        rel_path = record.get("rgb_image_path", "")
        if not rel_path:
            raise ValueError(f"Missing rgb_image_path in {self.jsonl_path}")
        full_path = resolve_image_path(self.image_root, rel_path, self.normalize_paths)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Dashcam image not found: {full_path}")
        crop_bbox = self._resolve_record_crop_bbox(record)
        return self._load_image_tensor(full_path, crop_bbox_xyxy=crop_bbox)
