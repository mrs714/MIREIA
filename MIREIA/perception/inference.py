from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from MIREIA.config import Config
from MIREIA.data_collection.inference_loader import InferenceFrameLoader
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
