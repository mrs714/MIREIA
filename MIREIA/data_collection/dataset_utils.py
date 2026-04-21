from __future__ import annotations

import json
import os
from typing import Callable, Iterable, List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_IMAGE_SIZE = (512, 512)
VALID_PARTITION_MODES = {"scenario", "frame"}
DEFAULT_VAL_SCENARIO_TOKENS_CSV = "Town05"


def normalize_partition_mode(partition_mode: str) -> str:
    mode = str(partition_mode).strip().lower()
    if mode not in VALID_PARTITION_MODES:
        raise ValueError("partition_mode must be 'scenario' or 'frame'")
    return mode


def normalize_validation_tokens(
    tokens: str | Iterable[str] | None,
    fallback_token: str | None = DEFAULT_VAL_SCENARIO_TOKENS_CSV,
) -> tuple[str, ...]:
    parsed: list[str] = []

    def _append(raw: str) -> None:
        for chunk in str(raw).split(","):
            token = chunk.strip()
            if token:
                parsed.append(token)

    if tokens is None:
        if fallback_token:
            _append(fallback_token)
    elif isinstance(tokens, str):
        _append(tokens)
    else:
        for item in tokens:
            _append(str(item))

    deduped: list[str] = []
    seen: set[str] = set()
    for token in parsed:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return tuple(deduped)


def scenario_is_validation_split(scenario_name: str, validation_tokens: Sequence[str]) -> bool:
    normalized_name = scenario_name.lower()
    for token in validation_tokens:
        normalized_token = str(token).strip().lower()
        if normalized_token and normalized_token in normalized_name:
            return True
    return False


def normalize_frame_train_ratio(frame_train_ratio: float) -> float:
    ratio = float(frame_train_ratio)
    if ratio > 1.0:
        ratio = ratio / 100.0
    if not (0.0 < ratio < 1.0):
        raise ValueError("frame_train_ratio must be in (0, 1) or (0, 100)")
    return ratio


def compute_frame_split_boundary(total_count: int, frame_train_ratio: float) -> int:
    if total_count <= 0:
        return 0
    if total_count == 1:
        return 1
    boundary = int(total_count * frame_train_ratio)
    return min(max(1, boundary), total_count - 1)


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


def normalize_crop_bbox_xyxy(bbox_xyxy: Sequence[float] | None) -> tuple[int, int, int, int] | None:
    if bbox_xyxy is None:
        return None
    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    except (TypeError, ValueError):
        return None

    return (x1, y1, x2, y2)


def resolve_record_crop_bbox(
    record: dict,
    crop_bbox_key: str | None = "crop_bbox_xyxy",
    manual_crop_bbox: Sequence[float] | None = None,
) -> tuple[int, int, int, int] | None:
    manual = normalize_crop_bbox_xyxy(manual_crop_bbox)
    if manual is not None:
        return manual

    if not crop_bbox_key:
        return None

    return normalize_crop_bbox_xyxy(record.get(crop_bbox_key))


def _clip_bbox_to_image(
    bbox_xyxy: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if bbox_xyxy is None:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def load_rgb_image(
    path: str,
    transform: Callable,
    crop_bbox_xyxy: Sequence[float] | None = None,
) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        bbox = normalize_crop_bbox_xyxy(crop_bbox_xyxy)
        clipped = _clip_bbox_to_image(bbox, width=img.width, height=img.height)
        if clipped is not None:
            img = img.crop(clipped)
        return transform(img)


class BaseSequenceDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        transform: Callable | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        target_mode: str = "last",
        risk_key: str = "ground_truth_risk",
        crop_bbox_key: str | None = "crop_bbox_xyxy",
        manual_crop_bbox: Sequence[float] | None = None,
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        self.seq_len = seq_len
        self.target_mode = target_mode
        self.risk_key = risk_key
        self.transform = transform or build_default_transform(image_size)
        self.crop_bbox_key = crop_bbox_key
        self.manual_crop_bbox = normalize_crop_bbox_xyxy(manual_crop_bbox)

    def _build_target(self, window: Sequence[dict]) -> torch.Tensor:
        if self.target_mode == "sequence":
            values = [rec[self.risk_key] for rec in window]
            return torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        if self.target_mode == "mean":
            value = sum(rec[self.risk_key] for rec in window) / len(window)
        else:
            value = window[-1][self.risk_key]
        return torch.tensor([value], dtype=torch.float32)

    def _resolve_record_crop_bbox(self, record: dict) -> tuple[int, int, int, int] | None:
        return resolve_record_crop_bbox(
            record=record,
            crop_bbox_key=self.crop_bbox_key,
            manual_crop_bbox=self.manual_crop_bbox,
        )

    def _load_image_tensor(
        self,
        full_path: str,
        crop_bbox_xyxy: Sequence[float] | None = None,
    ) -> torch.Tensor:
        return load_rgb_image(full_path, self.transform, crop_bbox_xyxy=crop_bbox_xyxy)
