from __future__ import annotations

import os
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from MIREIA.config import Config
from MIREIA.data_collection.dataset_utils import (
    _clip_bbox_to_image,
    normalize_crop_bbox_xyxy,
)
from MIREIA.data_collection.inference_loader import InferenceFrameLoader
from MIREIA.perception.bdu_gru_model import (
    BDUGRUModelConfig,
    BDUGRURiskPredictor,
    Seq2SeqBDUGRURiskPredictor,
)
from MIREIA.perception.e2e_model import E2EModelConfig, Seq2SeqRiskPredictor
from MIREIA.perception.feature_integration import FeatureIntegrator, FramePerception
from MIREIA.perception.flow import EgoMotionEstimator


@dataclass(frozen=True)
class QueuedTemporalConfig:
    sequence_len: int = Config.INFERENCE_SEQUENCE_LENGTH
    burn_in_frames: int = Config.INFERENCE_BURN_IN_FRAMES
    eval_frames: int = Config.INFERENCE_EVAL_FRAMES

    def __post_init__(self) -> None:
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be > 0")
        if self.burn_in_frames < 0:
            raise ValueError("burn_in_frames must be >= 0")
        if self.eval_frames <= 0:
            raise ValueError("eval_frames must be > 0")
        if self.burn_in_frames + self.eval_frames != self.sequence_len:
            raise ValueError("burn_in_frames + eval_frames must equal sequence_len")


@dataclass(frozen=True)
class QueuedRiskPrediction:
    ready: bool
    latest_risk: float | None
    risk_window: list[float]
    queue_size: int
    cache_hit: bool


class _LRUCache:
    """Simple bounded LRU cache for preprocessed tensors/features."""

    def __init__(self, max_entries: int | None = None):
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be > 0 or None")
        self.max_entries = max_entries
        self._store: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> tuple[bool, Any | None]:
        if key not in self._store:
            return False, None
        value = self._store.pop(key)
        self._store[key] = value
        return True, value

    def put(self, key: Any, value: Any) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if self.max_entries is not None:
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_frame_key(source: Any, frame_key: str | None = None) -> str:
    if frame_key is not None and str(frame_key).strip():
        return str(frame_key).strip()

    if isinstance(source, (str, Path)):
        return os.path.normpath(os.path.abspath(str(source)))

    raise ValueError(
        "frame_key is required for non-path sources so preprocessing can be reused."
    )


def _normalize_bdu_model_type(raw_type: str | None) -> str:
    normalized = str(raw_type or "").strip().lower()
    if normalized in {"seq2seq", "bdu_gru", "e2e"}:
        return "seq2seq"
    if normalized in {"single", "bdu_gru_single"}:
        return "single"
    return "seq2seq"


def _assert_full_e2e_state_dict(
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: str,
) -> None:
    has_spatial = any(k.startswith("spatial_backbone.") for k in state_dict)
    has_temporal = any(k.startswith("temporal_bdugru.") for k in state_dict)
    has_head = any(k.startswith("regression_head.") for k in state_dict)

    if not (has_spatial and has_temporal and has_head):
        raise RuntimeError(
            "Checkpoint is not a full E2E Seq2Seq checkpoint. "
            "This queued E2E inference expects end-to-end weights "
            "(spatial_backbone + temporal_bdugru + regression_head). "
            f"Checkpoint: {checkpoint_path}"
        )


class QueuedE2ERiskInference:
    """Queue-based E2E temporal inference with one-time per-frame preprocessing."""

    def __init__(
        self,
        model: Seq2SeqRiskPredictor,
        temporal_config: QueuedTemporalConfig | None = None,
        frame_loader: InferenceFrameLoader | None = None,
        device: torch.device | str | None = None,
        max_feature_cache_entries: int | None = 4096,
        manual_crop_bbox: Sequence[float] | None = None,
    ):
        self.temporal_config = temporal_config or QueuedTemporalConfig()
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        # If a frame_loader was passed, honor it as-is; otherwise build one with
        # the requested manual crop so trial inference matches training preprocess.
        self.frame_loader = frame_loader or InferenceFrameLoader(
            image_size=self.model.config.input_size,
            manual_crop_bbox=manual_crop_bbox,
        )

        self._feature_cache = _LRUCache(max_entries=max_feature_cache_entries)
        self._feature_queue: deque[torch.Tensor] = deque(maxlen=self.temporal_config.sequence_len)
        self._key_queue: deque[str] = deque(maxlen=self.temporal_config.sequence_len)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: E2EModelConfig | None = None,
        temporal_config: QueuedTemporalConfig | None = None,
        frame_loader: InferenceFrameLoader | None = None,
        device: torch.device | str | None = None,
        strict: bool = True,
        max_feature_cache_entries: int | None = 4096,
        manual_crop_bbox: Sequence[float] | None = None,
    ) -> "QueuedE2ERiskInference":
        resolved_device = _resolve_device(device)
        model = Seq2SeqRiskPredictor(config=model_config)

        payload = torch.load(checkpoint_path, map_location=resolved_device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}")

        _assert_full_e2e_state_dict(state_dict=state_dict, checkpoint_path=checkpoint_path)

        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is not compatible with Seq2SeqRiskPredictor. "
                "Use a full end-to-end seq2seq e2e checkpoint."
            ) from exc

        return cls(
            model=model,
            temporal_config=temporal_config,
            frame_loader=frame_loader,
            device=resolved_device,
            max_feature_cache_entries=max_feature_cache_entries,
            manual_crop_bbox=manual_crop_bbox,
        )

    def reset_queue(self) -> None:
        self._feature_queue.clear()
        self._key_queue.clear()

    def clear_preprocess_cache(self) -> None:
        self._feature_cache.clear()

    def add_image_path(self, image_path: str, frame_key: str | None = None) -> QueuedRiskPrediction:
        frame_key_resolved = _resolve_frame_key(image_path, frame_key=frame_key)
        hit, feature = self._feature_cache.get(frame_key_resolved)

        if not hit:
            frame_tensor = self.frame_loader.load_from_path(image_path)
            feature = self._extract_spatial_feature(frame_tensor)
            self._feature_cache.put(frame_key_resolved, feature)

        self._key_queue.append(frame_key_resolved)
        self._feature_queue.append(feature)

        return self._predict_from_queue(cache_hit=hit)

    def add_record(
        self,
        record: dict,
        image_root: str | None = None,
        rgb_key: str = "rgb_image_path",
        frame_key: str | None = None,
    ) -> QueuedRiskPrediction:
        image_path = self.frame_loader.resolve_record_image_path(
            record,
            image_root=image_root,
            rgb_key=rgb_key,
        )
        return self.add_image_path(image_path=image_path, frame_key=frame_key)

    def warm_start_from_paths(self, image_paths: Iterable[str]) -> None:
        for image_path in image_paths:
            self.add_image_path(image_path)

    def _extract_spatial_feature(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
        if frame_tensor.ndim != 4 or frame_tensor.shape[0] != 1:
            raise ValueError("Expected frame tensor shape (C, H, W) or (1, C, H, W)")

        with torch.inference_mode():
            frame_batch = frame_tensor.to(self.device, non_blocking=True)
            feature = self.model.spatial_backbone(frame_batch).squeeze(0)
        return feature.detach().cpu().float()

    def _predict_from_queue(self, cache_hit: bool) -> QueuedRiskPrediction:
        if len(self._feature_queue) < self.temporal_config.sequence_len:
            return QueuedRiskPrediction(
                ready=False,
                latest_risk=None,
                risk_window=[],
                queue_size=len(self._feature_queue),
                cache_hit=cache_hit,
            )

        with torch.inference_mode():
            feature_seq = torch.stack(tuple(self._feature_queue), dim=0).unsqueeze(0)
            feature_seq = feature_seq.to(self.device, non_blocking=True)
            temporal_seq = self.model.temporal_bdugru(feature_seq)

            start = self.temporal_config.burn_in_frames
            end = start + self.temporal_config.eval_frames
            eval_seq = temporal_seq[:, start:end, :]
            risk_seq = self.model.regression_head(eval_seq).squeeze(0).squeeze(-1)

        risk_window = [float(v) for v in risk_seq.detach().cpu().tolist()]
        latest_risk = risk_window[-1] if risk_window else None

        return QueuedRiskPrediction(
            ready=True,
            latest_risk=latest_risk,
            risk_window=risk_window,
            queue_size=len(self._feature_queue),
            cache_hit=cache_hit,
        )


class QueuedComposedBDUGRURiskInference:
    """Queue-based inference for the composed 32D-feature BDU-GRU model.

    This implementation caches per-frame perception outputs so overlapping frame pairs
    in streaming inference do not recompute detector/depth/environment/road passes for
    the previous frame at every step.
    """

    def __init__(
        self,
        model: Seq2SeqBDUGRURiskPredictor | BDUGRURiskPredictor,
        feature_integrator: FeatureIntegrator,
        yolo_model: Any,
        depth_estimator: Any,
        environment_predictor: Any | None = None,
        road_segmentation: Any | None = None,
        temporal_config: QueuedTemporalConfig | None = None,
        device: torch.device | str | None = None,
        max_pair_feature_cache_entries: int | None = 8192,
        max_source_cache_entries: int | None = 4096,
        max_frame_perception_cache_entries: int | None = 512,
        manual_crop_bbox: Sequence[float] | None = None,
    ):
        self.temporal_config = temporal_config or QueuedTemporalConfig()
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.feature_integrator = feature_integrator
        self.yolo_model = yolo_model
        self.depth_estimator = depth_estimator
        self.environment_predictor = environment_predictor
        self.road_segmentation = road_segmentation
        self._ego_motion_estimator = EgoMotionEstimator(crop_ratio=0.9)
        self.manual_crop_bbox = normalize_crop_bbox_xyxy(manual_crop_bbox)

        self._source_cache = _LRUCache(max_entries=max_source_cache_entries)
        self._frame_perception_cache = _LRUCache(max_entries=max_frame_perception_cache_entries)
        self._pair_feature_cache = _LRUCache(max_entries=max_pair_feature_cache_entries)

        self._feature_queue: deque[torch.Tensor] = deque(maxlen=self.temporal_config.sequence_len)
        self._key_queue: deque[str] = deque(maxlen=self.temporal_config.sequence_len)
        self._frame_queue: deque[FramePerception] = deque(maxlen=self.temporal_config.sequence_len)

        self._is_seq2seq = isinstance(model, Seq2SeqBDUGRURiskPredictor)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        feature_integrator: FeatureIntegrator,
        yolo_model: Any,
        depth_estimator: Any,
        environment_predictor: Any | None = None,
        road_segmentation: Any | None = None,
        temporal_config: QueuedTemporalConfig | None = None,
        device: torch.device | str | None = None,
        strict: bool = True,
        model_type: str | None = None,
        max_pair_feature_cache_entries: int | None = 8192,
        max_source_cache_entries: int | None = 4096,
        max_frame_perception_cache_entries: int | None = 512,
        manual_crop_bbox: Sequence[float] | None = None,
    ) -> "QueuedComposedBDUGRURiskInference":
        resolved_device = _resolve_device(device)
        payload = torch.load(checkpoint_path, map_location=resolved_device)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
            inferred_type = payload.get("model_type_internal", payload.get("model_type"))
            feature_dim = int(payload.get("feature_dim", 32))
        else:
            state_dict = payload
            inferred_type = None
            feature_dim = 32

        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}")

        internal_type = _normalize_bdu_model_type(model_type or inferred_type)
        config = BDUGRUModelConfig(feature_dim=feature_dim)

        model: Seq2SeqBDUGRURiskPredictor | BDUGRURiskPredictor
        if internal_type == "seq2seq":
            model = Seq2SeqBDUGRURiskPredictor(config=config)
        else:
            model = BDUGRURiskPredictor(config=config)

        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is not compatible with the selected BDU-GRU architecture."
            ) from exc

        return cls(
            model=model,
            feature_integrator=feature_integrator,
            yolo_model=yolo_model,
            depth_estimator=depth_estimator,
            environment_predictor=environment_predictor,
            road_segmentation=road_segmentation,
            temporal_config=temporal_config,
            device=resolved_device,
            max_pair_feature_cache_entries=max_pair_feature_cache_entries,
            max_source_cache_entries=max_source_cache_entries,
            max_frame_perception_cache_entries=max_frame_perception_cache_entries,
            manual_crop_bbox=manual_crop_bbox,
        )

    def reset_queue(self) -> None:
        self._feature_queue.clear()
        self._key_queue.clear()
        self._frame_queue.clear()

    def clear_preprocess_cache(self) -> None:
        self._source_cache.clear()
        self._frame_perception_cache.clear()
        self._pair_feature_cache.clear()

    def add_image_path(self, image_path: str, frame_key: str | None = None) -> QueuedRiskPrediction:
        return self.add_frame_source(source=image_path, frame_key=frame_key)

    def add_frame_source(self, source: Any, frame_key: str | None = None) -> QueuedRiskPrediction:
        key = _resolve_frame_key(source, frame_key=frame_key)
        source_hit, source_rgb = self._source_cache.get(key)
        if not source_hit:
            source_rgb = self._to_rgb_array(source)
            self._source_cache.put(key, source_rgb)

        frame_hit, frame_perception = self._frame_perception_cache.get(key)
        if not frame_hit:
            frame_perception = self.feature_integrator.extract_frame_perception(
                source_frame=source_rgb,
                yolo_model=self.yolo_model,
                depth_estimator=self.depth_estimator,
                environment_predictor=self.environment_predictor,
                road_segmentation=self.road_segmentation,
            )
            self._frame_perception_cache.put(key, frame_perception)

        if len(self._key_queue) == 0:
            prev_key = key
            prev_frame = frame_perception
        else:
            prev_key = self._key_queue[-1]
            prev_frame = self._frame_queue[-1]

        pair_key = (prev_key, key)
        feat_hit, feature_vec = self._pair_feature_cache.get(pair_key)
        if not feat_hit:
            feature_vec = self.feature_integrator.extract_state_vector_from_frame_perception(
                frame1=prev_frame,
                frame2=frame_perception,
                ego_motion_estimator=self._ego_motion_estimator,
            )
            if feature_vec.ndim != 1:
                feature_vec = feature_vec.reshape(-1)
            feature_vec = feature_vec.detach().cpu().float()
            self._pair_feature_cache.put(pair_key, feature_vec)

        self._key_queue.append(key)
        self._frame_queue.append(frame_perception)
        self._feature_queue.append(feature_vec)

        return self._predict_from_queue(cache_hit=bool(feat_hit and frame_hit))

    def _predict_from_queue(self, cache_hit: bool) -> QueuedRiskPrediction:
        if len(self._feature_queue) < self.temporal_config.sequence_len:
            return QueuedRiskPrediction(
                ready=False,
                latest_risk=None,
                risk_window=[],
                queue_size=len(self._feature_queue),
                cache_hit=cache_hit,
            )

        with torch.inference_mode():
            feature_seq = torch.stack(tuple(self._feature_queue), dim=0).unsqueeze(0)
            feature_seq = feature_seq.to(self.device, non_blocking=True)

            if self._is_seq2seq:
                temporal_seq = self.model.temporal_bdugru(feature_seq)
                start = self.temporal_config.burn_in_frames
                end = start + self.temporal_config.eval_frames
                eval_seq = temporal_seq[:, start:end, :]
                risk_seq = self.model.regression_head(eval_seq).squeeze(0).squeeze(-1)
                risk_window = [float(v) for v in risk_seq.detach().cpu().tolist()]
                latest_risk = risk_window[-1] if risk_window else None
            else:
                temporal_last = self.model.temporal_bdugru(feature_seq)
                risk = self.model.regression_head(temporal_last).reshape(-1)
                latest_risk = float(risk[-1].detach().cpu().item())
                risk_window = [latest_risk]

        return QueuedRiskPrediction(
            ready=True,
            latest_risk=latest_risk,
            risk_window=risk_window,
            queue_size=len(self._feature_queue),
            cache_hit=cache_hit,
        )

    def _to_rgb_array(self, source: Any) -> np.ndarray:
        if isinstance(source, (str, Path)):
            image_path = Path(source)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return self._apply_manual_crop(rgb)

        if isinstance(source, Image.Image):
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            return self._apply_manual_crop(rgb)

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

        return self._apply_manual_crop(rgb)

    def _apply_manual_crop(self, rgb: np.ndarray) -> np.ndarray:
        # No-op unless manual_crop_bbox was set at construction. Keeps the
        # composed predictor's preprocessing aligned with training (dashboard cut).
        if self.manual_crop_bbox is None:
            return rgb
        h, w = rgb.shape[:2]
        clipped = _clip_bbox_to_image(self.manual_crop_bbox, width=w, height=h)
        if clipped is None:
            return rgb
        x1, y1, x2, y2 = clipped
        return rgb[y1:y2, x1:x2]


__all__ = [
    "QueuedTemporalConfig",
    "QueuedRiskPrediction",
    "QueuedE2ERiskInference",
    "QueuedComposedBDUGRURiskInference",
]
