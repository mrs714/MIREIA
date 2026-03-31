from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from MIREIA.config import Config
from MIREIA.data_collection.inference_loader import InferenceFrameLoader
from MIREIA.perception.climate_model import MireiaEnvironmentClassifier
from MIREIA.perception.e2e_model import E2EModelConfig, Seq2SeqRiskPredictor


@dataclass(frozen=True)
class TemporalInferenceConfig:
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
class TemporalRiskPrediction:
    ready: bool
    latest_risk: float | None
    risk_window: list[float]
    buffer_size: int


@dataclass(frozen=True)
class EnvironmentPrediction:
    day_night_index: int
    day_night_label: str
    day_night_confidence: float
    climate_index: int
    climate_label: str
    climate_confidence: float


class StreamingRiskPredictor:
    """
    Online predictor with O(1) spatial extraction and FIFO temporal buffering.

    Each new frame runs only one CNN forward pass. The temporal model runs on the
    stacked feature buffer of fixed length.
    """

    def __init__(
        self,
        model: Seq2SeqRiskPredictor,
        temporal_config: TemporalInferenceConfig | None = None,
        frame_loader: InferenceFrameLoader | None = None,
        device: torch.device | str | None = None,
    ):
        self.temporal_config = temporal_config or TemporalInferenceConfig()
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.frame_loader = frame_loader or InferenceFrameLoader(
            image_size=self.model.config.input_size
        )
        self._feature_buffer: deque[torch.Tensor] = deque(maxlen=self.temporal_config.sequence_len)

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: E2EModelConfig | None = None,
        temporal_config: TemporalInferenceConfig | None = None,
        frame_loader: InferenceFrameLoader | None = None,
        device: torch.device | str | None = None,
        strict: bool = True,
    ) -> "StreamingRiskPredictor":
        resolved_device = cls._resolve_device(device)
        model = Seq2SeqRiskPredictor(config=model_config)

        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}")

        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is not compatible with Seq2SeqRiskPredictor. "
                "Use a seq2seq-trained checkpoint or pass strict=False with matching weights."
            ) from exc

        return cls(
            model=model,
            temporal_config=temporal_config,
            frame_loader=frame_loader,
            device=resolved_device,
        )

    def reset(self) -> None:
        self._feature_buffer.clear()

    def warm_start_from_paths(self, image_paths: Iterable[str]) -> None:
        for image_path in image_paths:
            self.predict_from_image_path(image_path)

    def predict_from_image_path(self, image_path: str) -> TemporalRiskPrediction:
        frame_tensor = self.frame_loader.load_from_path(image_path)
        return self.predict_from_frame_tensor(frame_tensor)

    def predict_from_record(
        self,
        record: dict,
        image_root: str | None = None,
        rgb_key: str = "rgb_image_path",
    ) -> TemporalRiskPrediction:
        frame_tensor = self.frame_loader.load_from_record(
            record,
            image_root=image_root,
            rgb_key=rgb_key,
        )
        return self.predict_from_frame_tensor(frame_tensor)

    def predict_from_frame_tensor(self, frame_tensor: torch.Tensor) -> TemporalRiskPrediction:
        feature = self._extract_spatial_feature(frame_tensor)
        self._feature_buffer.append(feature)

        if len(self._feature_buffer) < self.temporal_config.sequence_len:
            return TemporalRiskPrediction(
                ready=False,
                latest_risk=None,
                risk_window=[],
                buffer_size=len(self._feature_buffer),
            )

        with torch.inference_mode():
            feature_seq = torch.stack(tuple(self._feature_buffer), dim=0).unsqueeze(0)
            temporal_seq = self.model.temporal_bdugru(feature_seq)

            start = self.temporal_config.burn_in_frames
            end = start + self.temporal_config.eval_frames
            eval_seq = temporal_seq[:, start:end, :]
            risk_seq = self.model.regression_head(eval_seq).squeeze(0).squeeze(-1)

        risk_window = [float(value) for value in risk_seq.detach().cpu().tolist()]
        latest_risk = risk_window[-1] if risk_window else None

        return TemporalRiskPrediction(
            ready=True,
            latest_risk=latest_risk,
            risk_window=risk_window,
            buffer_size=len(self._feature_buffer),
        )

    def _extract_spatial_feature(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
        if frame_tensor.ndim != 4 or frame_tensor.shape[0] != 1:
            raise ValueError("Expected frame tensor shape (C, H, W) or (1, C, H, W)")

        with torch.inference_mode():
            frame_batch = frame_tensor.to(self.device, non_blocking=True)
            features = self.model.spatial_backbone(frame_batch)

        return features.squeeze(0)


class EnvironmentClassifierPredictor:
    """Single-frame predictor for day/night and climate classification."""

    DAY_NIGHT_LABELS: tuple[str, str] = ("day", "night")

    def __init__(
        self,
        model: MireiaEnvironmentClassifier,
        climate_labels: Sequence[str],
        frame_loader: InferenceFrameLoader | None = None,
        device: torch.device | str | None = None,
    ):
        if not climate_labels:
            raise ValueError("climate_labels must not be empty")

        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.climate_labels = [str(label) for label in climate_labels]
        self.frame_loader = frame_loader or InferenceFrameLoader(
            image_size=self.model.input_size
        )

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _infer_num_weather_classes(state_dict: dict[str, torch.Tensor]) -> int:
        for key, tensor in state_dict.items():
            if key.endswith("weather_head.3.weight") and tensor.ndim == 2:
                return int(tensor.shape[0])

        for key, tensor in state_dict.items():
            if "weather_head" in key and key.endswith("weight") and tensor.ndim == 2:
                return int(tensor.shape[0])

        raise ValueError(
            "Could not infer weather head output dimension from checkpoint state_dict"
        )

    @staticmethod
    def _labels_from_checkpoint_payload(payload: dict[str, object]) -> list[str] | None:
        idx_to_climate = payload.get("idx_to_climate")
        if isinstance(idx_to_climate, list) and idx_to_climate:
            return [str(label) for label in idx_to_climate]

        climate_to_idx = payload.get("climate_to_idx")
        if isinstance(climate_to_idx, dict) and climate_to_idx:
            try:
                pairs = sorted(
                    ((int(idx), str(label)) for label, idx in climate_to_idx.items()),
                    key=lambda item: item[0],
                )
            except (TypeError, ValueError):
                return None
            return [label for _, label in pairs]

        return None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        climate_labels: Sequence[str] | None = None,
        frame_loader: InferenceFrameLoader | None = None,
        device: torch.device | str | None = None,
        strict: bool = True,
    ) -> "EnvironmentClassifierPredictor":
        resolved_device = cls._resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            payload = checkpoint
        else:
            state_dict = checkpoint
            payload = None

        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}")

        labels = [str(label) for label in climate_labels] if climate_labels is not None else None
        if labels is None and isinstance(payload, dict):
            labels = cls._labels_from_checkpoint_payload(payload)

        if labels is None:
            inferred_classes = cls._infer_num_weather_classes(state_dict)
            labels = [f"class_{idx}" for idx in range(inferred_classes)]

        model = MireiaEnvironmentClassifier(num_weather_classes=len(labels))
        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is not compatible with MireiaEnvironmentClassifier. "
                "Use a multitask environment checkpoint or pass strict=False with matching weights."
            ) from exc

        return cls(
            model=model,
            climate_labels=labels,
            frame_loader=frame_loader,
            device=resolved_device,
        )

    def predict_from_image_path(self, image_path: str) -> EnvironmentPrediction:
        frame_tensor = self.frame_loader.load_from_path(image_path)
        return self.predict_from_frame_tensor(frame_tensor)

    def predict_from_record(
        self,
        record: dict,
        image_root: str | None = None,
        rgb_key: str = "rgb_image_path",
    ) -> EnvironmentPrediction:
        frame_tensor = self.frame_loader.load_from_record(
            record,
            image_root=image_root,
            rgb_key=rgb_key,
        )
        return self.predict_from_frame_tensor(frame_tensor)

    def predict_from_frame_tensor(self, frame_tensor: torch.Tensor) -> EnvironmentPrediction:
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
        if frame_tensor.ndim != 4 or frame_tensor.shape[0] != 1:
            raise ValueError("Expected frame tensor shape (C, H, W) or (1, C, H, W)")

        with torch.inference_mode():
            batch = frame_tensor.to(self.device, non_blocking=True)
            day_logits, climate_logits = self.model(batch)
            day_probs = torch.softmax(day_logits, dim=1).squeeze(0)
            climate_probs = torch.softmax(climate_logits, dim=1).squeeze(0)

        day_idx = int(torch.argmax(day_probs).item())
        climate_idx = int(torch.argmax(climate_probs).item())

        day_label = self.DAY_NIGHT_LABELS[day_idx]
        climate_label = (
            self.climate_labels[climate_idx]
            if climate_idx < len(self.climate_labels)
            else f"class_{climate_idx}"
        )

        return EnvironmentPrediction(
            day_night_index=day_idx,
            day_night_label=day_label,
            day_night_confidence=float(day_probs[day_idx].item()),
            climate_index=climate_idx,
            climate_label=climate_label,
            climate_confidence=float(climate_probs[climate_idx].item()),
        )


def create_streaming_predictor(
    checkpoint_path: str,
    model_config: E2EModelConfig | None = None,
    temporal_config: TemporalInferenceConfig | None = None,
    frame_loader: InferenceFrameLoader | None = None,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> StreamingRiskPredictor:
    """Convenience factory for online runtime inference."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return StreamingRiskPredictor.from_checkpoint(
        checkpoint_path=str(checkpoint),
        model_config=model_config,
        temporal_config=temporal_config,
        frame_loader=frame_loader,
        device=device,
        strict=strict,
    )


def create_environment_classifier_predictor(
    checkpoint_path: str,
    climate_labels: Sequence[str] | None = None,
    frame_loader: InferenceFrameLoader | None = None,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> EnvironmentClassifierPredictor:
    """Convenience factory for multitask environment classification."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return EnvironmentClassifierPredictor.from_checkpoint(
        checkpoint_path=str(checkpoint),
        climate_labels=climate_labels,
        frame_loader=frame_loader,
        device=device,
        strict=strict,
    )
