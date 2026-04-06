from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from MIREIA.config import Config

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = exc
else:
    _ULTRALYTICS_IMPORT_ERROR = None


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class YoloObstacleDetector:
    """
    Thin inference wrapper for Ultralytics YOLO checkpoints.

    Default filtering is focused on obstacle and self-driving relevant objects.
    """

    DEFAULT_CHECKPOINT_NAME = "yolo11s.pt"
    DEFAULT_TARGET_CLASSES: tuple[str, ...] = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "traffic light",
    )

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Sequence[str] | None = None,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "Ultralytics is required for YOLO inference. "
                "Install it with `pip install ultralytics`."
            ) from _ULTRALYTICS_IMPORT_ERROR

        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.device = self._resolve_device(device)
        self.confidence_threshold = self._validate_threshold(
            value=confidence_threshold,
            name="confidence_threshold",
        )
        self.iou_threshold = self._validate_threshold(
            value=iou_threshold,
            name="iou_threshold",
        )

        self.model = YOLO(str(self.checkpoint_path))
        self.class_names = self._normalize_class_names(getattr(self.model, "names", None))

        if not self.class_names:
            raise ValueError("Could not read class names from the YOLO checkpoint")

        self.target_classes = tuple(target_classes) if target_classes is not None else self.DEFAULT_TARGET_CLASSES
        self.target_class_ids = self._resolve_target_class_ids(self.target_classes)

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> str:
        if device is None:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if isinstance(device, torch.device):
            return str(device)
        return str(device)

    @staticmethod
    def _validate_threshold(value: float, name: str) -> float:
        validated = float(value)
        if validated < 0.0 or validated > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {validated}")
        return validated

    @classmethod
    def _resolve_checkpoint_path(cls, checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path is None:
            resolved = Path(Config.PATH_TO_MODELS) / cls.DEFAULT_CHECKPOINT_NAME
        else:
            resolved = Path(checkpoint_path)

        if not resolved.is_file():
            raise FileNotFoundError(
                f"YOLO checkpoint not found: {resolved}. "
                f"Place `{cls.DEFAULT_CHECKPOINT_NAME}` in `{Config.PATH_TO_MODELS}` "
                "or pass checkpoint_path explicitly."
            )

        return resolved.resolve()

    @staticmethod
    def _normalize_class_names(raw_names: Any) -> dict[int, str]:
        if isinstance(raw_names, dict):
            names: dict[int, str] = {}
            for key, value in raw_names.items():
                try:
                    class_id = int(key)
                except (TypeError, ValueError):
                    continue
                names[class_id] = str(value)
            return names

        if isinstance(raw_names, (list, tuple)):
            return {int(index): str(name) for index, name in enumerate(raw_names)}

        return {}

    def _resolve_target_class_ids(self, target_classes: Sequence[str]) -> list[int]:
        if not target_classes:
            return []

        name_to_id = {name.lower(): class_id for class_id, name in self.class_names.items()}
        resolved_ids: list[int] = []
        missing: list[str] = []

        for class_name in target_classes:
            lookup = str(class_name).strip().lower()
            if lookup in name_to_id:
                resolved_ids.append(int(name_to_id[lookup]))
            else:
                missing.append(str(class_name))

        if missing:
            available = ", ".join(self.class_names.values())
            raise ValueError(
                "Unknown target classes: "
                f"{missing}. Available YOLO classes: {available}"
            )

        return sorted(set(resolved_ids))

    def detect_from_image_path(
        self,
        image_path: str | Path,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int = 300,
        target_classes: Sequence[str] | None = None,
    ) -> list[YoloDetection]:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return self.detect(
            source=str(image_path),
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            target_classes=target_classes,
        )

    def detect(
        self,
        source: Any,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int = 300,
        target_classes: Sequence[str] | None = None,
    ) -> list[YoloDetection]:
        if int(max_detections) <= 0:
            raise ValueError(f"max_detections must be > 0, got {max_detections}")

        conf = self.confidence_threshold if confidence_threshold is None else self._validate_threshold(
            value=confidence_threshold,
            name="confidence_threshold",
        )
        iou = self.iou_threshold if iou_threshold is None else self._validate_threshold(
            value=iou_threshold,
            name="iou_threshold",
        )

        class_ids = self.target_class_ids
        if target_classes is not None:
            class_ids = self._resolve_target_class_ids(target_classes)

        prediction_results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=int(max_detections),
            classes=class_ids if class_ids else None,
            device=self.device,
            verbose=False,
        )

        if not prediction_results:
            return []

        result = prediction_results[0]
        result_names = self._normalize_class_names(getattr(result, "names", None)) or self.class_names
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            return []

        detections: list[YoloDetection] = []
        for box in boxes:
            xyxy_raw = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (float(value) for value in xyxy_raw)

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = result_names.get(class_id, str(class_id))

            detections.append(
                YoloDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    @staticmethod
    def detection_counts(detections: Iterable[YoloDetection]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts


def create_yolo_obstacle_detector(
    checkpoint_path: str | Path | None = None,
    device: torch.device | str | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_classes: Sequence[str] | None = None,
) -> YoloObstacleDetector:
    return YoloObstacleDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        target_classes=target_classes,
    )


__all__ = [
    "YoloDetection",
    "YoloObstacleDetector",
    "create_yolo_obstacle_detector",
]
