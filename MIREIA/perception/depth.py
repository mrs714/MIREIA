from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from MIREIA.config import Config

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    DepthAnythingV2 = None
    _DEPTH_ANYTHING_IMPORT_ERROR = exc
else:
    _DEPTH_ANYTHING_IMPORT_ERROR = None


@dataclass(frozen=True)
class DepthPrediction:
    source: str
    input_rgb: np.ndarray
    depth_map: np.ndarray
    normalized_depth: np.ndarray
    min_depth: float
    max_depth: float

    def colorize(self, colormap: str = "magma") -> np.ndarray:
        try:
            import matplotlib.cm as cm
        except ImportError as exc:  # pragma: no cover - optional visualization dependency
            raise ImportError(
                "Matplotlib is required to colorize depth maps. "
                "Install it with `pip install matplotlib`."
            ) from exc

        color_fn = cm.get_cmap(colormap)
        colored = color_fn(np.clip(self.normalized_depth, 0.0, 1.0))[..., :3]
        return (colored * 255.0).astype(np.uint8)


class DepthAnythingV2Estimator:
    """
    Inference wrapper for Depth Anything V2 checkpoints.

    This class expects a local checkpoint in MIREIA/models, by default
    `depth_anything_v2_vits.pth`.
    """

    DEFAULT_CHECKPOINT_NAME = "depth_anything_v2_vits.pth"
    MODEL_CONFIGS: dict[str, dict[str, Any]] = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        encoder: str | None = None,
        device: torch.device | str | None = None,
        input_size: int = 518,
    ) -> None:
        if DepthAnythingV2 is None:
            raise ImportError(
                "Depth Anything V2 is required for depth inference. "
                "Install it with `pip install depth-anything-v2`."
            ) from _DEPTH_ANYTHING_IMPORT_ERROR

        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.encoder = self._resolve_encoder(encoder=encoder, checkpoint_path=self.checkpoint_path)
        self.device = self._resolve_device(device)
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError(f"input_size must be > 0, got {self.input_size}")

        model_kwargs = dict(self.MODEL_CONFIGS[self.encoder])
        self.model = DepthAnythingV2(**model_kwargs)
        state_dict = self._load_state_dict(self.checkpoint_path)

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not load Depth Anything V2 checkpoint. "
                f"Checkpoint: {self.checkpoint_path}. "
                f"Selected encoder: {self.encoder}. "
                "Make sure the checkpoint architecture matches the encoder."
            ) from exc

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
            resolved = Path(Config.PATH_TO_MODELS) / cls.DEFAULT_CHECKPOINT_NAME
        else:
            resolved = Path(checkpoint_path)

        if not resolved.is_file():
            raise FileNotFoundError(
                f"Depth checkpoint not found: {resolved}. "
                f"Place `{cls.DEFAULT_CHECKPOINT_NAME}` in `{Config.PATH_TO_MODELS}` "
                "or pass checkpoint_path explicitly."
            )

        return resolved.resolve()

    @classmethod
    def _resolve_encoder(cls, encoder: str | None, checkpoint_path: Path) -> str:
        if encoder is not None:
            resolved = str(encoder).strip().lower()
        else:
            filename = checkpoint_path.name.lower()
            resolved = ""
            for candidate in cls.MODEL_CONFIGS:
                if candidate in filename:
                    resolved = candidate
                    break
            if not resolved:
                resolved = "vits"

        if resolved not in cls.MODEL_CONFIGS:
            allowed = ", ".join(sorted(cls.MODEL_CONFIGS))
            raise ValueError(f"Unsupported encoder '{resolved}'. Allowed: {allowed}")

        return resolved

    @staticmethod
    def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
        if isinstance(payload, Mapping):
            for candidate_key in ("state_dict", "model", "model_state_dict"):
                maybe_state = payload.get(candidate_key)
                if isinstance(maybe_state, Mapping):
                    payload = maybe_state
                    break

        if not isinstance(payload, Mapping):
            raise ValueError("Unsupported checkpoint format for Depth Anything V2")

        state_dict: dict[str, torch.Tensor] = {}
        for raw_key, raw_value in payload.items():
            if not torch.is_tensor(raw_value):
                continue

            key = str(raw_key)
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("model."):
                key = key[len("model.") :]

            state_dict[key] = raw_value

        if not state_dict:
            raise ValueError("No tensor weights were found in checkpoint payload")

        return state_dict

    @classmethod
    def _load_state_dict(cls, checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        return cls._extract_state_dict(payload)

    @staticmethod
    def _to_rgb_array(source: Any) -> tuple[np.ndarray, str]:
        if isinstance(source, (str, Path)):
            image_path = Path(source)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
            return np.asarray(image), str(image_path)

        if isinstance(source, Image.Image):
            image = source.convert("RGB")
            return np.asarray(image), "PIL.Image"

        array = np.asarray(source)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, repeats=3, axis=-1)

        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError("Expected an RGB-like image input with shape (H, W, 3)")

        rgb = array[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return rgb, "array"

    def _infer_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        bgr_image = rgb_image[..., ::-1]

        infer_image = getattr(self.model, "infer_image", None)
        if not callable(infer_image):
            raise RuntimeError(
                "DepthAnythingV2 backend does not expose infer_image. "
                "Install the official Depth Anything V2 package and retry."
            )

        try:
            depth_raw = infer_image(bgr_image, input_size=self.input_size)
        except TypeError:
            depth_raw = infer_image(bgr_image)

        if torch.is_tensor(depth_raw):
            depth_raw = depth_raw.detach().cpu().numpy()

        depth_map = np.asarray(depth_raw, dtype=np.float32)
        if depth_map.ndim == 3 and depth_map.shape[0] == 1:
            depth_map = depth_map[0]
        elif depth_map.ndim == 3 and depth_map.shape[-1] == 1:
            depth_map = depth_map[..., 0]

        if depth_map.ndim != 2:
            raise ValueError(f"Unexpected depth output shape: {depth_map.shape}")

        return depth_map

    @staticmethod
    def _normalize_depth(depth_map: np.ndarray) -> tuple[np.ndarray, float, float]:
        min_depth = float(np.nanmin(depth_map))
        max_depth = float(np.nanmax(depth_map))

        if not np.isfinite(min_depth) or not np.isfinite(max_depth):
            raise ValueError("Depth output contains non-finite values")

        scale = max_depth - min_depth
        if scale <= 1e-8:
            normalized = np.zeros_like(depth_map, dtype=np.float32)
        else:
            normalized = (depth_map - min_depth) / scale
            normalized = normalized.astype(np.float32)

        return normalized, min_depth, max_depth

    def predict(self, source: Any) -> DepthPrediction:
        rgb_image, source_name = self._to_rgb_array(source)
        depth_map = self._infer_depth(rgb_image)
        normalized_depth, min_depth, max_depth = self._normalize_depth(depth_map)

        return DepthPrediction(
            source=source_name,
            input_rgb=rgb_image,
            depth_map=depth_map,
            normalized_depth=normalized_depth,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    def predict_from_image_path(self, image_path: str | Path) -> DepthPrediction:
        return self.predict(source=image_path)


def create_depth_anything_v2_estimator(
    checkpoint_path: str | Path | None = None,
    encoder: str | None = None,
    device: torch.device | str | None = None,
    input_size: int = 518,
) -> DepthAnythingV2Estimator:
    return DepthAnythingV2Estimator(
        checkpoint_path=checkpoint_path,
        encoder=encoder,
        device=device,
        input_size=input_size,
    )


__all__ = [
    "DepthPrediction",
    "DepthAnythingV2Estimator",
    "create_depth_anything_v2_estimator",
]