from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from MIREIA.config import Config
from MIREIA.perception._raft_backbone import InputPadder, RAFT, flow_to_image


@dataclass(frozen=True)
class FlowPrediction:
    source_first: str
    source_second: str
    first_rgb: np.ndarray
    second_rgb: np.ndarray
    flow_xy: np.ndarray
    flow_rgb: np.ndarray
    magnitude: np.ndarray
    mean_magnitude: float
    max_magnitude: float

    def colorize(self) -> np.ndarray:
        return self.flow_rgb


@dataclass
class _RaftArgs:
    small: bool = False
    mixed_precision: bool = False
    alternate_corr: bool = False
    dropout: float = 0.0

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class RaftOpticalFlowEstimator:
    """
    Thin inference wrapper around an internal RAFT implementation.
    """

    DEFAULT_CHECKPOINT_NAME = "raft-kitti.pth"
    DEFAULT_CHECKPOINT_SUBDIR = "raft"
    DEFAULT_ITERS = 20

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        raft_repo_root: str | Path | None = None,
        device: torch.device | str | None = None,
        small: bool = False,
        mixed_precision: bool = False,
        alternate_corr: bool = False,
        dropout: float = 0.0,
    ) -> None:
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.device = self._resolve_device(device)
        # Kept for API compatibility with previous versions. No longer used.
        self.raft_repo_root = None if raft_repo_root is None else Path(raft_repo_root)

        raft_args = _RaftArgs(
            small=bool(small),
            mixed_precision=bool(mixed_precision),
            alternate_corr=bool(alternate_corr),
            dropout=float(dropout),
        )

        base_model = RAFT(raft_args)
        state_dict = self._load_state_dict(self.checkpoint_path)
        self.model = self._load_weights(base_model=base_model, state_dict=state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> str:
        if device is None:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if isinstance(device, torch.device):
            return str(device)
        return str(device)

    @classmethod
    def _resolve_checkpoint_path(cls, checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path is None:
            resolved = Path(Config.PATH_TO_MODELS) / cls.DEFAULT_CHECKPOINT_SUBDIR / cls.DEFAULT_CHECKPOINT_NAME
        else:
            resolved = Path(checkpoint_path)

        if not resolved.is_file():
            raise FileNotFoundError(
                f"RAFT checkpoint not found: {resolved}. "
                f"Place `{cls.DEFAULT_CHECKPOINT_NAME}` in "
                f"`{Path(Config.PATH_TO_MODELS) / cls.DEFAULT_CHECKPOINT_SUBDIR}` "
                "or pass checkpoint_path explicitly."
            )

        return resolved.resolve()

    @staticmethod
    def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
        if isinstance(payload, Mapping):
            for candidate_key in ("state_dict", "model", "model_state_dict"):
                maybe_state = payload.get(candidate_key)
                if isinstance(maybe_state, Mapping):
                    payload = maybe_state
                    break

        if not isinstance(payload, Mapping):
            raise ValueError("Unsupported RAFT checkpoint format")

        state_dict: dict[str, torch.Tensor] = {}
        for raw_key, raw_value in payload.items():
            if torch.is_tensor(raw_value):
                state_dict[str(raw_key)] = raw_value

        if not state_dict:
            raise ValueError("No tensor weights were found in RAFT checkpoint payload")

        return state_dict

    @classmethod
    def _load_state_dict(cls, checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        return cls._extract_state_dict(payload)

    @staticmethod
    def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stripped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                stripped[key[len("module.") :]] = value
            else:
                stripped[key] = value
        return stripped

    @classmethod
    def _load_weights(cls, base_model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> torch.nn.Module:
        # Most official checkpoints are saved from DataParallel, so try that first.
        parallel_model = torch.nn.DataParallel(base_model)
        try:
            parallel_model.load_state_dict(state_dict, strict=True)
            return parallel_model.module
        except RuntimeError:
            normalized = cls._strip_module_prefix(state_dict)
            base_model.load_state_dict(normalized, strict=True)
            return base_model

    @staticmethod
    def _to_rgb_array(source: Any) -> tuple[np.ndarray, str]:
        if isinstance(source, (str, Path)):
            image_path = Path(source)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB")), str(image_path)

        if isinstance(source, Image.Image):
            return np.asarray(source.convert("RGB")), "PIL.Image"

        array = np.asarray(source)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, repeats=3, axis=-1)

        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError("Expected an RGB-like image with shape (H, W, 3)")

        rgb = array[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return rgb, "array"

    def _to_model_tensor(self, rgb_image: np.ndarray) -> torch.Tensor:
        contiguous = np.ascontiguousarray(rgb_image)
        tensor = torch.from_numpy(contiguous).permute(2, 0, 1).float().unsqueeze(0)
        return tensor.to(self.device)

    def _flow_to_rgb(self, flow_xy: np.ndarray) -> np.ndarray:
        flow_rgb = np.asarray(flow_to_image(flow_xy))
        if flow_rgb.ndim == 3 and flow_rgb.shape[-1] >= 3:
            flow_rgb = flow_rgb[..., :3]
        return flow_rgb.astype(np.uint8)

    def predict(
        self,
        source_first: Any,
        source_second: Any,
        iters: int = DEFAULT_ITERS,
        pad_mode: str = "sintel",
    ) -> FlowPrediction:
        iters = int(iters)
        if iters <= 0:
            raise ValueError(f"iters must be > 0, got {iters}")

        image1_rgb, source1_name = self._to_rgb_array(source_first)
        image2_rgb, source2_name = self._to_rgb_array(source_second)

        if image1_rgb.shape[:2] != image2_rgb.shape[:2]:
            raise ValueError(
                "Both images must have the same spatial shape for RAFT inference. "
                f"Got {image1_rgb.shape[:2]} and {image2_rgb.shape[:2]}"
            )

        image1_tensor = self._to_model_tensor(image1_rgb)
        image2_tensor = self._to_model_tensor(image2_rgb)

        padder = InputPadder(image1_tensor.shape, mode=str(pad_mode))
        image1_tensor, image2_tensor = padder.pad(image1_tensor, image2_tensor)

        with torch.no_grad():
            _, flow_up = self.model(image1_tensor, image2_tensor, iters=iters, test_mode=True)

        flow_up = padder.unpad(flow_up[0]).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        magnitude = np.linalg.norm(flow_up, axis=-1).astype(np.float32)
        flow_rgb = self._flow_to_rgb(flow_up)

        return FlowPrediction(
            source_first=source1_name,
            source_second=source2_name,
            first_rgb=image1_rgb,
            second_rgb=image2_rgb,
            flow_xy=flow_up,
            flow_rgb=flow_rgb,
            magnitude=magnitude,
            mean_magnitude=float(np.mean(magnitude)),
            max_magnitude=float(np.max(magnitude)),
        )

    def predict_from_image_paths(
        self,
        image1_path: str | Path,
        image2_path: str | Path,
        iters: int = DEFAULT_ITERS,
        pad_mode: str = "sintel",
    ) -> FlowPrediction:
        return self.predict(
            source_first=image1_path,
            source_second=image2_path,
            iters=iters,
            pad_mode=pad_mode,
        )


def create_raft_optical_flow_estimator(
    checkpoint_path: str | Path | None = None,
    raft_repo_root: str | Path | None = None,
    device: torch.device | str | None = None,
    small: bool = False,
    mixed_precision: bool = False,
    alternate_corr: bool = False,
    dropout: float = 0.0,
) -> RaftOpticalFlowEstimator:
    return RaftOpticalFlowEstimator(
        checkpoint_path=checkpoint_path,
        raft_repo_root=raft_repo_root,
        device=device,
        small=small,
        mixed_precision=mixed_precision,
        alternate_corr=alternate_corr,
        dropout=dropout,
    )


__all__ = [
    "FlowPrediction",
    "RaftOpticalFlowEstimator",
    "create_raft_optical_flow_estimator",
]
