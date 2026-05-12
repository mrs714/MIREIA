from __future__ import annotations

import os
from typing import Callable, Sequence

import torch

from MIREIA.data_collection.dataset_utils import (
    DEFAULT_IMAGE_SIZE,
    build_default_transform,
    load_rgb_image,
    normalize_crop_bbox_xyxy,
    resolve_image_path,
)


class InferenceFrameLoader:
    """Load single RGB frames for online or offline model inference.

    Pass `manual_crop_bbox` (xyxy) to crop every loaded frame before the
    resize/normalize transform — same path used by training datasets so trial
    inference matches the preprocessing the model was trained on (e.g. cutting
    the dashboard out of the dashcam frame).
    """

    def __init__(
        self,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        transform: Callable | None = None,
        normalize_paths: bool = True,
        manual_crop_bbox: Sequence[float] | None = None,
    ):
        self.normalize_paths = normalize_paths
        self.transform = transform or build_default_transform(image_size)
        self.manual_crop_bbox = normalize_crop_bbox_xyxy(manual_crop_bbox)

    def __call__(self, image_path: str) -> torch.Tensor:
        return self.load_from_path(image_path)

    def load_from_path(self, image_path: str) -> torch.Tensor:
        full_path = os.path.normpath(image_path) if self.normalize_paths else image_path
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Inference image not found: {full_path}")
        return load_rgb_image(full_path, self.transform, crop_bbox_xyxy=self.manual_crop_bbox)

    def resolve_record_image_path(
        self,
        record: dict,
        image_root: str | None = None,
        rgb_key: str = "rgb_image_path",
    ) -> str:
        rel_path = str(record.get(rgb_key, "")).strip()
        if not rel_path:
            raise ValueError(f"Record does not contain a non-empty '{rgb_key}' value")

        if os.path.isabs(rel_path):
            return os.path.normpath(rel_path) if self.normalize_paths else rel_path

        if image_root is None:
            raise ValueError("image_root is required when record image path is relative")

        return resolve_image_path(image_root, rel_path, normalize_paths=self.normalize_paths)

    def load_from_record(
        self,
        record: dict,
        image_root: str | None = None,
        rgb_key: str = "rgb_image_path",
    ) -> torch.Tensor:
        image_path = self.resolve_record_image_path(record, image_root=image_root, rgb_key=rgb_key)
        return self.load_from_path(image_path)
