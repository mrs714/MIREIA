from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from MIREIA.config import Config

try:
    _sam2_auto_module = importlib.import_module("sam2.automatic_mask_generator")
    _sam2_build_module = importlib.import_module("sam2.build_sam")
    SAM2AutomaticMaskGenerator = getattr(_sam2_auto_module, "SAM2AutomaticMaskGenerator")
    build_sam2 = getattr(_sam2_build_module, "build_sam2")
except Exception as exc:  # pragma: no cover - runtime dependency guard
    SAM2AutomaticMaskGenerator = None
    build_sam2 = None
    _SAM2_IMPORT_ERROR = exc
else:
    _SAM2_IMPORT_ERROR = None


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned pixel bounding box using exclusive x2/y2."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, int(self.x2) - int(self.x1))

    @property
    def height(self) -> int:
        return max(0, int(self.y2) - int(self.y1))

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass(frozen=True)
class DashBBoxResult:
    source: str
    instruction: str
    image_rgb: np.ndarray
    dashboard_mask: np.ndarray
    dashboard_bbox: BoundingBox | None
    inverse_bbox: BoundingBox | None
    candidate_count: int
    selected_area: int
    selected_score: float

    def crop_inverse_view(self) -> np.ndarray | None:
        if self.inverse_bbox is None:
            return None
        return self.image_rgb[
            self.inverse_bbox.y1 : self.inverse_bbox.y2,
            self.inverse_bbox.x1 : self.inverse_bbox.x2,
        ]


@dataclass(frozen=True)
class _InstructionPreferences:
    bottom_bias: float
    center_bias: float
    min_top_ratio: float
    min_bottom_overlap: float
    max_height_ratio: float


class Sam2DashboardSegmenter:
    """
    SAM2 wrapper for dashboard/car-hood segmentation and downstream safe crop region.

    SAM2 itself is promptable with points/boxes, not direct text. The `instruction`
    string is treated as a semantic hint that adjusts candidate-scoring heuristics.
    """

    DEFAULT_CHECKPOINT_NAME = "sam2.1_hiera_small.pt"
    DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
    DEFAULT_INSTRUCTION = """
        Detect ONLY the dashboard, car hood, wipers, mirrors, and steering wheel of the ego vehicle. 
        EXCLUDE the road completely (do not include any road pixels, even if they are close to the hood). 
        The road must NOT be part of the mask. If unsure, prefer to miss part of the hood rather than include any road. 
        Do NOT include the sky or anything outside the car.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_cfg: str | Path | None = None,
        device: torch.device | str | None = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 0,
    ) -> None:
        if SAM2AutomaticMaskGenerator is None or build_sam2 is None:
            raise ImportError(
                "SAM2 is required for dashboard segmentation. "
                "Install it with `pip install -e git+https://github.com/facebookresearch/sam2.git#egg=SAM-2`."
            ) from _SAM2_IMPORT_ERROR

        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.model_cfg = self._resolve_model_cfg(model_cfg)
        self.device = self._resolve_device(device)

        # Keep SAM2 constructor usage compatible with minor API variations.
        try:
            self.model = build_sam2(
                self.model_cfg,
                str(self.checkpoint_path),
                device=self.device,
                apply_postprocessing=False,
            )
        except TypeError:
            self.model = build_sam2(self.model_cfg, str(self.checkpoint_path), self.device)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=int(points_per_side),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh),
            min_mask_region_area=int(min_mask_region_area),
        )

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
                f"SAM2 checkpoint not found: {resolved}. "
                f"Place `{cls.DEFAULT_CHECKPOINT_NAME}` in `{Config.PATH_TO_MODELS}` "
                "or pass checkpoint_path explicitly."
            )

        return resolved.resolve()

    @classmethod
    def _resolve_model_cfg(cls, model_cfg: str | Path | None) -> str:
        if model_cfg is None:
            return cls.DEFAULT_MODEL_CFG

        cfg_path = Path(model_cfg)
        if cfg_path.is_file():
            return str(cfg_path.resolve())

        return str(model_cfg)

    @staticmethod
    def _to_rgb_array(source: Any) -> tuple[np.ndarray, str]:
        if isinstance(source, (str, Path)):
            image_path = Path(source)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB")), str(image_path)

        if isinstance(source, Image.Image):
            image = source.convert("RGB")
            return np.asarray(image), "PIL.Image"

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

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> BoundingBox | None:
        ys, xs = np.nonzero(mask)
        if ys.size == 0 or xs.size == 0:
            return None

        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _instruction_preferences(instruction: str) -> _InstructionPreferences:
        lower_instruction = instruction.lower()
        ego_terms = ("ego", "dashboard", "hood", "bonnet", "windshield")
        has_ego_hint = any(term in lower_instruction for term in ego_terms)

        if has_ego_hint:
            return _InstructionPreferences(
                bottom_bias=2.5,
                center_bias=1.1,
                min_top_ratio=0.42,
                min_bottom_overlap=0.45,
                max_height_ratio=0.55,
            )

        return _InstructionPreferences(
            bottom_bias=1.6,
            center_bias=0.6,
            min_top_ratio=0.20,
            min_bottom_overlap=0.25,
            max_height_ratio=0.85,
        )

    @staticmethod
    def _score_dashboard_candidate(
        mask: np.ndarray,
        bbox: BoundingBox,
        prefs: _InstructionPreferences,
    ) -> float:
        height, width = mask.shape
        area = float(mask.sum())
        if area <= 0.0:
            return float("-inf")

        bottom_start = int(height * 0.55)
        center_left = int(width * 0.20)
        center_right = int(width * 0.80)

        bottom_overlap = float(mask[bottom_start:, :].sum()) / area
        center_overlap = float(mask[:, center_left:center_right].sum()) / area
        top_overlap = float(mask[: int(height * 0.45), :].sum()) / area
        height_ratio = float(bbox.height) / float(height)

        score = area
        score *= 1.0 + prefs.bottom_bias * bottom_overlap
        score *= 1.0 + prefs.center_bias * center_overlap

        if bool(mask[-1, :].any()):
            score *= 1.4

        if bbox.y1 < int(height * prefs.min_top_ratio):
            score *= 0.55

        if bottom_overlap < prefs.min_bottom_overlap:
            score *= 0.20

        if top_overlap > 0.10:
            score *= 0.30

        if height_ratio > prefs.max_height_ratio:
            score *= 0.35

        if bbox.height < int(height * 0.05):
            score *= 0.4

        if bbox.width < int(width * 0.10):
            score *= 0.5

        return float(score)

    @staticmethod
    def _largest_true_rectangle(mask: np.ndarray) -> BoundingBox | None:
        """
        Largest axis-aligned rectangle fully contained in a True/False mask.

        This uses the standard monotonic-stack histogram algorithm row-by-row.
        """

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        height, width = mask.shape
        if height == 0 or width == 0:
            return None

        histogram = np.zeros(width, dtype=np.int32)
        best_area = 0
        best_bbox: BoundingBox | None = None

        for y in range(height):
            histogram = np.where(mask[y], histogram + 1, 0)

            stack: list[tuple[int, int]] = []
            for x in range(width + 1):
                current_height = int(histogram[x]) if x < width else 0
                start = x

                while stack and stack[-1][1] > current_height:
                    index, popped_height = stack.pop()
                    rect_width = x - index
                    rect_area = popped_height * rect_width

                    if rect_area > best_area:
                        y2 = y + 1
                        y1 = y2 - popped_height
                        x1 = index
                        x2 = x
                        best_area = rect_area
                        best_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

                    start = index

                if not stack or stack[-1][1] < current_height:
                    stack.append((start, current_height))

        return best_bbox

    def create_dash_bb(
        self,
        source: Any,
        instruction: str | None = None,
        min_mask_area_ratio: float = 0.001,
    ) -> DashBBoxResult:
        instruction_text = str(instruction or self.DEFAULT_INSTRUCTION)
        if not (0.0 < float(min_mask_area_ratio) <= 1.0):
            raise ValueError("min_mask_area_ratio must be in (0, 1]")

        image_rgb, source_name = self._to_rgb_array(source)
        image_height, image_width = image_rgb.shape[:2]
        image_area = image_height * image_width
        min_mask_pixels = max(1, int(image_area * float(min_mask_area_ratio)))

        raw_masks = self.mask_generator.generate(image_rgb)

        candidate_count = 0
        best_mask: np.ndarray | None = None
        best_bbox: BoundingBox | None = None
        best_score = float("-inf")
        best_area = 0

        prefs = self._instruction_preferences(instruction_text)

        for payload in raw_masks:
            if not isinstance(payload, dict):
                continue

            segmentation = payload.get("segmentation")
            if segmentation is None:
                continue

            mask = np.asarray(segmentation).astype(bool)
            if mask.shape != (image_height, image_width):
                continue

            area = int(mask.sum())
            if area < min_mask_pixels:
                continue

            bbox = self._bbox_from_mask(mask)
            if bbox is None:
                continue

            candidate_count += 1
            score = self._score_dashboard_candidate(mask=mask, bbox=bbox, prefs=prefs)

            if score > best_score:
                best_score = score
                best_mask = mask
                best_bbox = bbox
                best_area = area

        if best_mask is None:
            # If all candidates were filtered out, try the largest raw mask as a fallback.
            fallback_area = 0
            for payload in raw_masks:
                if not isinstance(payload, dict):
                    continue
                segmentation = payload.get("segmentation")
                if segmentation is None:
                    continue
                mask = np.asarray(segmentation).astype(bool)
                if mask.shape != (image_height, image_width):
                    continue
                area = int(mask.sum())
                if area > fallback_area:
                    bbox = self._bbox_from_mask(mask)
                    if bbox is None:
                        continue
                    best_mask = mask
                    best_bbox = bbox
                    fallback_area = area

            best_area = fallback_area
            best_score = float(best_area)

        if best_mask is None:
            empty_mask = np.zeros((image_height, image_width), dtype=bool)
            return DashBBoxResult(
                source=source_name,
                instruction=instruction_text,
                image_rgb=image_rgb,
                dashboard_mask=empty_mask,
                dashboard_bbox=None,
                inverse_bbox=BoundingBox(0, 0, image_width, image_height),
                candidate_count=0,
                selected_area=0,
                selected_score=float("-inf"),
            )

        free_view_mask = np.logical_not(best_mask)
        inverse_bbox = self._largest_true_rectangle(free_view_mask)

        return DashBBoxResult(
            source=source_name,
            instruction=instruction_text,
            image_rgb=image_rgb,
            dashboard_mask=best_mask,
            dashboard_bbox=best_bbox,
            inverse_bbox=inverse_bbox,
            candidate_count=candidate_count,
            selected_area=int(best_area),
            selected_score=float(best_score),
        )

    def create_dash_bb_from_image_path(
        self,
        image_path: str | Path,
        instruction: str | None = None,
        min_mask_area_ratio: float = 0.001,
    ) -> DashBBoxResult:
        return self.create_dash_bb(
            source=image_path,
            instruction=instruction,
            min_mask_area_ratio=min_mask_area_ratio,
        )


def create_dash_bb(
    source: Any,
    checkpoint_path: str | Path | None = None,
    model_cfg: str | Path | None = None,
    instruction: str | None = None,
    device: torch.device | str | None = None,
    min_mask_area_ratio: float = 0.001,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 0,
) -> DashBBoxResult:
    """Convenience single-call API for dashboard and inverse bounding boxes."""

    segmenter = Sam2DashboardSegmenter(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        device=device,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    return segmenter.create_dash_bb(
        source=source,
        instruction=instruction,
        min_mask_area_ratio=min_mask_area_ratio,
    )


def create_sam2_dashboard_segmenter(
    checkpoint_path: str | Path | None = None,
    model_cfg: str | Path | None = None,
    device: torch.device | str | None = None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 0,
) -> Sam2DashboardSegmenter:
    return Sam2DashboardSegmenter(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        device=device,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )


class DashBBCleanCropTransform:
    """
    Callable preprocessing transform for DataLoader pipelines.

    It applies `create_dash_bb` logic, keeps the inverse clean-view crop, and then
    runs an optional `post_transform` (by default: Resize + ToTensor).
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_cfg: str | Path | None = None,
        instruction: str | None = None,
        device: torch.device | str | None = None,
        min_mask_area_ratio: float = 0.001,
        min_inverse_area_ratio: float = 0.05,
        fallback_to_original: bool = True,
        output_size: tuple[int, int] = (512, 512),
        post_transform: Any | None = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 0,
    ) -> None:
        if not (0.0 < float(min_mask_area_ratio) <= 1.0):
            raise ValueError("min_mask_area_ratio must be in (0, 1]")
        if not (0.0 <= float(min_inverse_area_ratio) <= 1.0):
            raise ValueError("min_inverse_area_ratio must be in [0, 1]")

        self.segmenter = Sam2DashboardSegmenter(
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            device=device,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
        self.instruction = str(instruction or Sam2DashboardSegmenter.DEFAULT_INSTRUCTION)
        self.min_mask_area_ratio = float(min_mask_area_ratio)
        self.min_inverse_area_ratio = float(min_inverse_area_ratio)
        self.fallback_to_original = bool(fallback_to_original)

        if post_transform is None:
            try:
                from torchvision import transforms as tv_transforms
            except ImportError as exc:  # pragma: no cover - runtime dependency guard
                raise ImportError(
                    "torchvision is required for the default loader transform. "
                    "Install it with `pip install torchvision` or pass post_transform explicitly."
                ) from exc

            post_transform = tv_transforms.Compose(
                [
                    tv_transforms.Resize(output_size),
                    tv_transforms.ToTensor(),
                ]
            )

        self.post_transform = post_transform

    def _select_clean_crop(self, result: DashBBoxResult) -> np.ndarray:
        image_height, image_width = result.image_rgb.shape[:2]
        image_area = max(1, image_height * image_width)

        crop = result.crop_inverse_view()
        if crop is None or crop.size == 0 or result.inverse_bbox is None:
            if self.fallback_to_original:
                return result.image_rgb
            raise ValueError("SAM2 inverse crop is empty and fallback_to_original is False")

        inverse_area_ratio = float(result.inverse_bbox.area) / float(image_area)
        if inverse_area_ratio < self.min_inverse_area_ratio:
            if self.fallback_to_original:
                return result.image_rgb
            raise ValueError(
                "SAM2 inverse crop area is below min_inverse_area_ratio and "
                "fallback_to_original is False"
            )

        return crop

    def apply_with_metadata(self, source: Any) -> tuple[Any, DashBBoxResult]:
        result = self.segmenter.create_dash_bb(
            source=source,
            instruction=self.instruction,
            min_mask_area_ratio=self.min_mask_area_ratio,
        )

        clean_crop = self._select_clean_crop(result)
        clean_pil = Image.fromarray(np.asarray(clean_crop, dtype=np.uint8))

        if self.post_transform is None:
            return clean_pil, result
        return self.post_transform(clean_pil), result

    def __call__(self, source: Any) -> Any:
        transformed, _ = self.apply_with_metadata(source)
        return transformed


def create_dash_bb_transform(
    checkpoint_path: str | Path | None = None,
    model_cfg: str | Path | None = None,
    instruction: str | None = None,
    device: torch.device | str | None = None,
    min_mask_area_ratio: float = 0.001,
    min_inverse_area_ratio: float = 0.05,
    fallback_to_original: bool = True,
    output_size: tuple[int, int] = (512, 512),
    post_transform: Any | None = None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 0,
) -> DashBBCleanCropTransform:
    """
    Factory for a DataLoader-ready callable transform.

    The returned object can be passed directly as the `transform` argument of your
    datasets/loaders that expect a callable from PIL image to tensor.
    """

    return DashBBCleanCropTransform(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        instruction=instruction,
        device=device,
        min_mask_area_ratio=min_mask_area_ratio,
        min_inverse_area_ratio=min_inverse_area_ratio,
        fallback_to_original=fallback_to_original,
        output_size=output_size,
        post_transform=post_transform,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )


__all__ = [
    "BoundingBox",
    "DashBBCleanCropTransform",
    "DashBBoxResult",
    "Sam2DashboardSegmenter",
    "create_dash_bb",
    "create_dash_bb_transform",
    "create_sam2_dashboard_segmenter",
]
