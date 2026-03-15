from __future__ import annotations

import json
import os
from typing import Callable, List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class E2ESequenceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        seq_len: int,
        image_root: str | None = None,
        image_size: tuple[int, int] = (512, 512),
        transform: Callable | None = None,
        target_mode: str = "last",
        risk_key: str = "ground_truth_risk",
    ):
        self.jsonl_path = jsonl_path
        self.seq_len = seq_len
        self.image_root = image_root or os.path.dirname(jsonl_path)
        self.target_mode = target_mode
        self.risk_key = risk_key
        self.records = self._load_records(jsonl_path)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return max(0, len(self.records) - self.seq_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.records[index : index + self.seq_len]
        images = [self._load_image(rec["rgb_image_path"]) for rec in window]
        seq_tensor = torch.stack(images, dim=0)
        target = self._build_target(window)
        return seq_tensor, target

    def _build_target(self, window: Sequence[dict]) -> torch.Tensor:
        if self.target_mode == "mean":
            value = sum(rec[self.risk_key] for rec in window) / len(window)
        else:
            value = window[-1][self.risk_key]
        return torch.tensor([value], dtype=torch.float32)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        full_path = os.path.join(self.image_root, rel_path)
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            return self.transform(img)

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
