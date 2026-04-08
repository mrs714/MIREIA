from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np


class EgoMotionEstimator:
    """Estimate global ego-motion between two RGB frames using phase correlation.

    The method converts frames to grayscale, keeps a center crop to reduce border
    artifacts, and uses cv2.phaseCorrelate to estimate a single global shift.
    """

    def __init__(self, crop_ratio: float = 0.9) -> None:
        if crop_ratio <= 0.0 or crop_ratio > 1.0:
            raise ValueError("crop_ratio must be in (0, 1]")
        self.crop_ratio = float(crop_ratio)

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        crop_h = max(1, int(round(height * self.crop_ratio)))
        crop_w = max(1, int(round(width * self.crop_ratio)))
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        return image[y0 : y0 + crop_h, x0 : x0 + crop_w]

    def estimate_motion(self, frame1_rgb: np.ndarray, frame2_rgb: np.ndarray) -> Tuple[float, float]:
        if frame1_rgb.ndim != 3 or frame1_rgb.shape[-1] < 3:
            raise ValueError("frame1_rgb must have shape (H, W, 3)")
        if frame2_rgb.ndim != 3 or frame2_rgb.shape[-1] < 3:
            raise ValueError("frame2_rgb must have shape (H, W, 3)")
        if frame1_rgb.shape[:2] != frame2_rgb.shape[:2]:
            raise ValueError(
                "Both frames must have the same spatial shape. "
                f"Got {frame1_rgb.shape[:2]} and {frame2_rgb.shape[:2]}"
            )

        gray1 = cv2.cvtColor(frame1_rgb[..., :3], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2_rgb[..., :3], cv2.COLOR_RGB2GRAY)

        gray1 = np.asarray(gray1, dtype=np.float32)
        gray2 = np.asarray(gray2, dtype=np.float32)

        gray1_crop = self._center_crop(gray1)
        gray2_crop = self._center_crop(gray2)

        (x_shift, y_shift), _ = cv2.phaseCorrelate(gray1_crop, gray2_crop)
        return float(x_shift), float(y_shift)


def track_objects(model: Any, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
    """Track objects in an RGB frame using Ultralytics YOLO ByteTrack.

    Args:
        model: Ultralytics YOLO model instance.
        frame_rgb: RGB image as a numpy array with shape (H, W, 3).

    Returns:
        A list of dictionaries with keys:
            - "id": tracking id (int)
            - "bbox": [x1, y1, x2, y2] (list[float])
            - "class": class id (int)
        Detections without an assigned tracking id are ignored.
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] < 3:
        raise ValueError("frame_rgb must have shape (H, W, 3)")

    results = model.track(
        source=frame_rgb,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    track_ids = getattr(boxes, "id", None)
    if track_ids is None:
        return []

    ids_list = track_ids.int().cpu().tolist()
    xyxy_list = boxes.xyxy.cpu().tolist()
    class_list = boxes.cls.int().cpu().tolist()

    tracked: list[dict[str, Any]] = []
    for track_id, xyxy, class_id in zip(ids_list, xyxy_list, class_list):
        if track_id is None:
            continue

        tracked.append(
            {
                "id": int(track_id),
                "bbox": [float(value) for value in xyxy],
                "class": int(class_id),
            }
        )

    return tracked


__all__ = ["EgoMotionEstimator", "track_objects"]