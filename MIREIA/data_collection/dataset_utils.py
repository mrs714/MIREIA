from __future__ import annotations

import json
import os
from typing import Callable, List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_IMAGE_SIZE = (512, 512)


def build_default_transform(image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )


def load_jsonl_records(jsonl_path: str) -> List[dict]:
    records: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_image_path(image_root: str, rel_path: str, normalize_paths: bool = True) -> str:
    if os.path.isabs(rel_path):
        path = rel_path
    else:
        path = os.path.join(image_root, rel_path)
    return os.path.normpath(path) if normalize_paths else path


def load_rgb_image(path: str, transform: Callable) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return transform(img)


class BaseSequenceDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        transform: Callable | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        target_mode: str = "last",
        risk_key: str = "ground_truth_risk",
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        self.seq_len = seq_len
        self.target_mode = target_mode
        self.risk_key = risk_key
        self.transform = transform or build_default_transform(image_size)

    def _build_target(self, window: Sequence[dict]) -> torch.Tensor:
        if self.target_mode == "sequence":
            values = [rec[self.risk_key] for rec in window]
            return torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        if self.target_mode == "mean":
            value = sum(rec[self.risk_key] for rec in window) / len(window)
        else:
            value = window[-1][self.risk_key]
        return torch.tensor([value], dtype=torch.float32)

    def _load_image_tensor(self, full_path: str) -> torch.Tensor:
        return load_rgb_image(full_path, self.transform)
