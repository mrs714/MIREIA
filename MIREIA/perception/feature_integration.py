from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image

from MIREIA.perception.flow import EgoMotionEstimator, track_objects


@dataclass(frozen=True)
class _ObjectMetrics:
    object_id: int | None
    area: float
    avg_depth: float
    cx: float
    cy: float


@dataclass(frozen=True)
class FramePerception:
    frame_rgb: np.ndarray
    yolo_tracks: list[dict[str, Any]]
    depth_map: np.ndarray
    road_relative_size: float
    day_prob: float
    climate_probs: list[float]


class FeatureIntegrator:
    """Build a strict 32-feature state tensor from multi-model perception outputs.

    Feature order:
    [
        1. num_objects,
        2. bb_size_max,
        3. bb_size_avg,
        4. depth_min_norm,
        5. depth_avg_norm,
        6. size_change_max,
        7. size_change_avg,
        8. depth_change_max,
        9. depth_change_avg,
        10. bg_flow_x_norm,
        11. bg_flow_y_norm,
        12. left_threat_max,
        13. left_threat_avg,
        14. center_threat_max,
        15. center_threat_avg,
        16. right_threat_max,
        17. right_threat_avg,
        18. road_relative_size,
        19. day_prob,
        20. climate_1, ..., 32. climate_13,
    ]
    """

    EPS = 1e-6
    CLEAR_ROAD_DEPTH = 100.0
    MAX_SIZE_CHANGE = 5.0
    MAX_DEPTH_CHANGE = 5.0
    MAX_THREAT_SCORE = 10.0

    def extract_state_vector_from_sources(
        self,
        source_frame1: Any,
        source_frame2: Any,
        *,
        yolo_model: Any,
        depth_estimator: Any,
        environment_predictor: Any | None = None,
        road_segmentation: Mapping[str, Any] | Any | None = None,
        ego_motion_estimator: EgoMotionEstimator | None = None,
    ) -> torch.Tensor:
        """Run all perception models and return the strict 32-feature tensor.

        Args:
            source_frame1: Frame 1 source (path, PIL image, or RGB-like array).
            source_frame2: Frame 2 source (path, PIL image, or RGB-like array).
            yolo_model: Ultralytics YOLO model instance used for ByteTrack.
            depth_estimator: Metric depth estimator exposing predict(...).depth_map.
            environment_predictor: Optional EnvironmentClassifierPredictor-like instance.
            road_segmentation: Optional road-seg source. Can be either:
                - dict from load_road_segmentation_model(...)
                - model instance returning (lane_logits, road_logits)
            ego_motion_estimator: Optional EgoMotionEstimator. If None, one is created.
        """

        frame1 = self.extract_frame_perception(
            source_frame=source_frame1,
            yolo_model=yolo_model,
            depth_estimator=depth_estimator,
            environment_predictor=environment_predictor,
            road_segmentation=road_segmentation,
        )
        frame2 = self.extract_frame_perception(
            source_frame=source_frame2,
            yolo_model=yolo_model,
            depth_estimator=depth_estimator,
            environment_predictor=environment_predictor,
            road_segmentation=road_segmentation,
        )

        return self.extract_state_vector_from_frame_perception(
            frame1=frame1,
            frame2=frame2,
            ego_motion_estimator=ego_motion_estimator,
        )

    def extract_frame_perception(
        self,
        source_frame: Any,
        *,
        yolo_model: Any,
        depth_estimator: Any,
        environment_predictor: Any | None = None,
        road_segmentation: Mapping[str, Any] | Any | None = None,
    ) -> FramePerception:
        frame_rgb, _ = self._to_rgb_array(source_frame)

        yolo_tracks = track_objects(model=yolo_model, frame_rgb=frame_rgb)
        depth_map = np.asarray(depth_estimator.predict(frame_rgb).depth_map, dtype=np.float32)

        road_relative_size = self._infer_road_relative_size(
            frame_rgb=frame_rgb,
            road_segmentation=road_segmentation,
        )
        day_prob, climate_probs = self._infer_environment_probs(
            frame_rgb=frame_rgb,
            environment_predictor=environment_predictor,
        )

        return FramePerception(
            frame_rgb=frame_rgb,
            yolo_tracks=yolo_tracks,
            depth_map=depth_map,
            road_relative_size=float(road_relative_size),
            day_prob=float(day_prob),
            climate_probs=[float(v) for v in climate_probs],
        )

    def extract_state_vector_from_frame_perception(
        self,
        frame1: FramePerception,
        frame2: FramePerception,
        *,
        ego_motion_estimator: EgoMotionEstimator | None = None,
    ) -> torch.Tensor:
        if frame1.frame_rgb.shape[:2] != frame2.frame_rgb.shape[:2]:
            raise ValueError(
                "Both input frames must have the same spatial shape. "
                f"Got {frame1.frame_rgb.shape[:2]} and {frame2.frame_rgb.shape[:2]}"
            )

        ego_estimator = ego_motion_estimator or EgoMotionEstimator(crop_ratio=0.9)
        bg_flow_x, bg_flow_y = ego_estimator.estimate_motion(frame1.frame_rgb, frame2.frame_rgb)

        return self.extract_state_vector(
            yolo_tracks1=frame1.yolo_tracks,
            yolo_tracks2=frame2.yolo_tracks,
            depth_map1=frame1.depth_map,
            depth_map2=frame2.depth_map,
            bg_flow_x=bg_flow_x,
            bg_flow_y=bg_flow_y,
            road_relative_size=frame2.road_relative_size,
            day_prob=frame2.day_prob,
            climate_probs=frame2.climate_probs,
        )

    def extract_state_vector(
        self,
        yolo_tracks1: list[dict[str, Any]],
        yolo_tracks2: list[dict[str, Any]],
        depth_map1: np.ndarray,
        depth_map2: np.ndarray,
        bg_flow_x: float,
        bg_flow_y: float,
        road_relative_size: float,
        day_prob: float,
        climate_probs: list[float],
    ) -> torch.Tensor:
        depth_map1_arr = self._as_depth_map(depth_map1)
        depth_map2_arr = self._as_depth_map(depth_map2)

        frame1_objects = self._extract_frame_objects(yolo_tracks1, depth_map1_arr)
        frame2_objects = self._extract_frame_objects(yolo_tracks2, depth_map2_arr)

        num_objects = float(len(frame2_objects))

        frame2_areas = np.asarray([obj.area for obj in frame2_objects], dtype=np.float32)
        frame2_depths = np.asarray([obj.avg_depth for obj in frame2_objects], dtype=np.float32)

        image_area = float(max(1, depth_map2_arr.shape[0] * depth_map2_arr.shape[1]))
        frame2_areas_norm = (
            np.clip(frame2_areas / image_area, 0.0, 1.0)
            if frame2_areas.size
            else frame2_areas
        )
        frame2_areas_sqrt = (
            np.sqrt(frame2_areas_norm)
            if frame2_areas_norm.size
            else frame2_areas_norm
        )

        bb_size_max = float(frame2_areas_sqrt.max()) if frame2_areas_sqrt.size else 0.0
        bb_size_avg = float(frame2_areas_sqrt.mean()) if frame2_areas_sqrt.size else 0.0
        depth_min_raw = float(frame2_depths.min()) if frame2_depths.size else self.CLEAR_ROAD_DEPTH
        depth_avg_raw = float(frame2_depths.mean()) if frame2_depths.size else 0.0
        depth_min = self._normalize_depth(depth_min_raw)
        depth_avg = self._normalize_depth(depth_avg_raw)

        frame1_by_id = {
            obj.object_id: obj for obj in frame1_objects if obj.object_id is not None
        }
        frame2_by_id = {
            obj.object_id: obj for obj in frame2_objects if obj.object_id is not None
        }
        matched_ids = sorted(set(frame1_by_id).intersection(frame2_by_id))

        size_changes: list[float] = []
        depth_changes: list[float] = []

        zone_scores: dict[str, list[float]] = {
            "left": [],
            "center": [],
            "right": [],
        }

        height = float(depth_map2_arr.shape[0])
        width = float(depth_map2_arr.shape[1])
        left_cut = width / 3.0
        right_cut = (2.0 * width) / 3.0

        norm_bg_flow_x, norm_bg_flow_y = self._normalize_flow(
            flow_x=bg_flow_x,
            flow_y=bg_flow_y,
            width=width,
            height=height,
        )

        for track_id in matched_ids:
            obj1 = frame1_by_id[track_id]
            obj2 = frame2_by_id[track_id]

            size_change = obj2.area / (obj1.area + self.EPS)
            depth_change = obj2.avg_depth / (obj1.avg_depth + self.EPS)
            size_change = self._clip_non_negative(size_change, self.MAX_SIZE_CHANGE)
            depth_change = self._clip_non_negative(depth_change, self.MAX_DEPTH_CHANGE)
            centroid_shift = float(np.hypot(obj2.cx - obj1.cx, obj2.cy - obj1.cy))
            centroid_shift = self._normalize_centroid_shift(
                shift_px=centroid_shift,
                width=width,
                height=height,
            )

            threat_score = (size_change + centroid_shift) * (1.0 / (obj2.avg_depth + self.EPS))
            threat_score = self._clip_non_negative(threat_score, self.MAX_THREAT_SCORE)

            if obj2.cx < left_cut:
                zone_scores["left"].append(threat_score)
            elif obj2.cx < right_cut:
                zone_scores["center"].append(threat_score)
            else:
                zone_scores["right"].append(threat_score)

            size_changes.append(float(size_change))
            depth_changes.append(float(depth_change))

        has_matched_objects = len(matched_ids) > 0

        if has_matched_objects:
            size_change_arr = np.asarray(size_changes, dtype=np.float32)
            depth_change_arr = np.asarray(depth_changes, dtype=np.float32)

            size_change_max = float(size_change_arr.max())
            size_change_avg = float(size_change_arr.mean())
            depth_change_max = float(depth_change_arr.max())
            depth_change_avg = float(depth_change_arr.mean())

            left_threat_max, left_threat_avg = self._max_avg(zone_scores["left"])
            center_threat_max, center_threat_avg = self._max_avg(zone_scores["center"])
            right_threat_max, right_threat_avg = self._max_avg(zone_scores["right"])
        else:
            size_change_max = 0.0
            size_change_avg = 0.0
            depth_change_max = 0.0
            depth_change_avg = 0.0
            left_threat_max = 0.0
            left_threat_avg = 0.0
            center_threat_max = 0.0
            center_threat_avg = 0.0
            right_threat_max = 0.0
            right_threat_avg = 0.0

        # Critical defaulting policy requested for no objects or no matched objects.
        if num_objects == 0.0 or not has_matched_objects:
            bb_size_max = 0.0
            bb_size_avg = 0.0
            depth_min = self._normalize_depth(self.CLEAR_ROAD_DEPTH)
            depth_avg = 0.0
            size_change_max = 0.0
            size_change_avg = 0.0
            depth_change_max = 0.0
            depth_change_avg = 0.0
            left_threat_max = 0.0
            left_threat_avg = 0.0
            center_threat_max = 0.0
            center_threat_avg = 0.0
            right_threat_max = 0.0
            right_threat_avg = 0.0

        climate_baseline_dropped = self._normalize_climate(climate_probs)

        features = np.asarray(
            [
                num_objects,
                bb_size_max,
                bb_size_avg,
                depth_min,
                depth_avg,
                size_change_max,
                size_change_avg,
                depth_change_max,
                depth_change_avg,
                norm_bg_flow_x,
                norm_bg_flow_y,
                left_threat_max,
                left_threat_avg,
                center_threat_max,
                center_threat_avg,
                right_threat_max,
                right_threat_avg,
                float(road_relative_size),
                float(day_prob),
                *climate_baseline_dropped.tolist(),
            ],
            dtype=np.float32,
        )

        if features.shape[0] != 32:
            raise RuntimeError(
                f"Feature vector length must be exactly 32, got {features.shape[0]}"
            )

        return torch.as_tensor(features, dtype=torch.float32)

    def _normalize_flow(
        self,
        flow_x: float,
        flow_y: float,
        width: float,
        height: float,
    ) -> tuple[float, float]:
        width_safe = max(1.0, float(width))
        height_safe = max(1.0, float(height))
        norm_x = float(np.clip(float(flow_x) / width_safe, -1.0, 1.0))
        norm_y = float(np.clip(float(flow_y) / height_safe, -1.0, 1.0))
        return norm_x, norm_y

    def _normalize_centroid_shift(self, shift_px: float, width: float, height: float) -> float:
        diagonal = max(self.EPS, float(np.hypot(width, height)))
        return float(np.clip(float(shift_px) / diagonal, 0.0, 1.0))

    def _normalize_depth(self, depth_value: float) -> float:
        return float(np.clip(float(depth_value) / self.CLEAR_ROAD_DEPTH, 0.0, 1.0))

    def _clip_non_negative(self, value: float, max_value: float) -> float:
        return float(np.clip(float(value), 0.0, float(max_value)))

    def _extract_frame_objects(
        self,
        tracks: list[dict[str, Any]],
        depth_map: np.ndarray,
    ) -> list[_ObjectMetrics]:
        height, width = depth_map.shape
        objects: list[_ObjectMetrics] = []

        for track in tracks:
            if not isinstance(track, dict):
                continue

            bbox_raw = track.get("bbox")
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                continue

            x1, y1, x2, y2 = self._clip_bbox(
                bbox_raw=bbox_raw,
                width=width,
                height=height,
            )

            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            area = float(box_w * box_h)

            roi = self._depth_roi(depth_map, x1, y1, x2, y2)
            avg_depth = self._avg_depth(roi)

            object_id = self._safe_id(track.get("id"))
            cx = float((x1 + x2) * 0.5)
            cy = float((y1 + y2) * 0.5)

            objects.append(
                _ObjectMetrics(
                    object_id=object_id,
                    area=area,
                    avg_depth=avg_depth,
                    cx=cx,
                    cy=cy,
                )
            )

        return objects

    def _as_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth_map, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"depth_map must be a 2D array, got shape {arr.shape}"
            )
        return arr

    def _to_rgb_array(self, source: Any) -> tuple[np.ndarray, str]:
        if isinstance(source, (str, Path)):
            image_path = Path(source)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8), str(image_path)

        if isinstance(source, Image.Image):
            return np.asarray(source.convert("RGB"), dtype=np.uint8), "PIL.Image"

        arr = np.asarray(source)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, repeats=3, axis=-1)

        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError("Expected RGB-like input with shape (H, W, 3)")

        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return rgb, "array"

    def _infer_road_relative_size(
        self,
        frame_rgb: np.ndarray,
        road_segmentation: Mapping[str, Any] | Any | None,
    ) -> float:
        if road_segmentation is None:
            return 0.0

        if isinstance(road_segmentation, Mapping):
            model = road_segmentation.get("model")
            image_size_raw = road_segmentation.get("image_size", (256, 256))
            pred_threshold = float(road_segmentation.get("pred_threshold", 0.5))
        else:
            model = road_segmentation
            image_size_raw = (256, 256)
            pred_threshold = 0.5

        if model is None:
            return 0.0

        if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
            seg_h, seg_w = int(image_size_raw[0]), int(image_size_raw[1])
        else:
            seg_h, seg_w = 256, 256

        pred_threshold = float(np.clip(pred_threshold, 0.0, 1.0))

        resized = Image.fromarray(frame_rgb).resize((seg_w, seg_h), resample=Image.BILINEAR)
        img_np = np.asarray(resized, dtype=np.float32) / 255.0

        # Match ImageNet normalization used by the segmentation model setup.
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        seg_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        seg_device = self._infer_model_device(model)

        with torch.inference_mode():
            lane_logits, road_logits = model(seg_tensor.to(seg_device, non_blocking=True))
            road_probs = torch.sigmoid(road_logits)[0, 0].detach().cpu().numpy()

        road_mask = road_probs > pred_threshold
        if road_mask.size == 0:
            return 0.0

        return float(np.mean(road_mask.astype(np.float32)))

    def _infer_environment_probs(
        self,
        frame_rgb: np.ndarray,
        environment_predictor: Any | None,
    ) -> tuple[float, list[float]]:
        if environment_predictor is None:
            return 0.0, [0.0] * 14

        # Preferred path: run predictor model logits directly and keep full softmax outputs.
        try:
            frame_tensor = self._build_environment_tensor(frame_rgb, environment_predictor)
            model = getattr(environment_predictor, "model")
            device = getattr(environment_predictor, "device", torch.device("cpu"))

            with torch.inference_mode():
                day_logits, climate_logits = model(frame_tensor.to(device, non_blocking=True))
                day_probs = torch.softmax(day_logits, dim=1).squeeze(0)
                climate_probs = torch.softmax(climate_logits, dim=1).squeeze(0)

            day_prob = float(day_probs[0].item()) if day_probs.numel() >= 1 else 0.0
            climate_list = [float(v) for v in climate_probs.detach().cpu().tolist()]
            return day_prob, self._ensure_len(climate_list, target_len=14)
        except Exception:
            pass

        # Fallback path for looser predictor-like wrappers.
        try:
            frame_tensor = self._build_environment_tensor(frame_rgb, environment_predictor)
            prediction = environment_predictor.predict_from_frame_tensor(frame_tensor.squeeze(0))

            if getattr(prediction, "day_night_label", "") == "day":
                day_prob = float(getattr(prediction, "day_night_confidence", 0.0))
            else:
                day_prob = 1.0 - float(getattr(prediction, "day_night_confidence", 0.0))

            climate_probs = [0.0] * 14
            climate_index = int(getattr(prediction, "climate_index", 0))
            if 0 <= climate_index < 14:
                climate_probs[climate_index] = float(
                    getattr(prediction, "climate_confidence", 0.0)
                )

            return float(day_prob), climate_probs
        except Exception:
            return 0.0, [0.0] * 14

    def _build_environment_tensor(self, frame_rgb: np.ndarray, environment_predictor: Any) -> torch.Tensor:
        frame_loader = getattr(environment_predictor, "frame_loader", None)
        transform = getattr(frame_loader, "transform", None)

        if callable(transform):
            tensor = transform(Image.fromarray(frame_rgb).convert("RGB"))
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            return tensor

        # Generic fallback with ImageNet normalization at a common runtime size.
        image = Image.fromarray(frame_rgb).convert("RGB").resize((224, 224), Image.BILINEAR)
        img_np = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _infer_model_device(self, model: Any) -> torch.device:
        try:
            first_param = next(model.parameters())
            return first_param.device
        except Exception:
            return torch.device("cpu")

    def _ensure_len(self, values: list[float], target_len: int) -> list[float]:
        if len(values) < target_len:
            return values + [0.0] * (target_len - len(values))
        if len(values) > target_len:
            return values[:target_len]
        return values

    def _clip_bbox(
        self,
        bbox_raw: list[float] | tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox_raw]

        x1 = float(np.clip(x1, 0.0, float(width)))
        y1 = float(np.clip(y1, 0.0, float(height)))
        x2 = float(np.clip(x2, 0.0, float(width)))
        y2 = float(np.clip(y2, 0.0, float(height)))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return x1, y1, x2, y2

    def _depth_roi(
        self,
        depth_map: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        h, w = depth_map.shape
        x1i = int(np.clip(np.floor(x1), 0, w))
        y1i = int(np.clip(np.floor(y1), 0, h))
        x2i = int(np.clip(np.ceil(x2), 0, w))
        y2i = int(np.clip(np.ceil(y2), 0, h))

        if x2i <= x1i or y2i <= y1i:
            return np.empty((0, 0), dtype=np.float32)

        return depth_map[y1i:y2i, x1i:x2i]

    def _avg_depth(self, roi: np.ndarray) -> float:
        if roi.size == 0:
            return self.CLEAR_ROAD_DEPTH

        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size == 0:
            return self.CLEAR_ROAD_DEPTH

        return float(valid.mean())

    def _safe_id(self, raw_id: Any) -> int | None:
        if raw_id is None:
            return None
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            return None

    def _max_avg(self, values: Iterable[float]) -> tuple[float, float]:
        vals = np.asarray(list(values), dtype=np.float32)
        if vals.size == 0:
            return 0.0, 0.0
        return float(vals.max()), float(vals.mean())

    def _normalize_climate(self, climate_probs: list[float]) -> np.ndarray:
        arr = np.asarray(climate_probs, dtype=np.float32).reshape(-1)

        # Keep the interface robust in chained pipelines while guaranteeing 32 features.
        if arr.size < 14:
            arr = np.pad(arr, (0, 14 - arr.size), mode="constant", constant_values=0.0)
        elif arr.size > 14:
            arr = arr[:14]

        # Baseline climate class is removed; keep first 13 entries.
        return arr[:13]


__all__ = ["FeatureIntegrator", "FramePerception"]
