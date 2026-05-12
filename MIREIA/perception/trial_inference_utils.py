"""Shared helpers for trial batch runners: build the E2E + Composed
predictors with a consistent dashboard crop and preview the crop on a sample
frame so the bbox can be tuned before kicking off a long batch.

Both `QueuedE2ERiskInference` and `QueuedComposedBDUGRURiskInference` accept a
`manual_crop_bbox` argument that crops every loaded RGB frame *before* the
resize/normalize transform. This matches the way training datasets are
preprocessed (`crop_bbox_xyxy` cuts the dashboard out of the dashcam frame),
so trial inference no longer sees the squashed full-frame distribution shift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from MIREIA.config import Config
from MIREIA.perception.feature_integration import FeatureIntegrator
from MIREIA.perception.queued_inference import (
    QueuedComposedBDUGRURiskInference,
    QueuedE2ERiskInference,
)


def build_trial_predictors(
    *,
    device: str | torch.device | None = None,
    manual_crop_bbox: Sequence[float] | None,
    e2e_checkpoint: str | Path | None = None,
    composed_checkpoint: str | Path | None = None,
    yolo_checkpoint: str | Path | None = None,
    depth_checkpoint: str | Path | None = None,
    climate_checkpoint: str | Path | None = None,
    road_checkpoint: str | Path | None = None,
) -> tuple[QueuedE2ERiskInference | None, QueuedComposedBDUGRURiskInference | None]:
    """Load E2E + Composed predictors once, both wired with the same crop.

    Any checkpoint path that is None or missing causes that predictor to be
    skipped (returned as None). Defaults pull from `Config.PATH_TO_MODELS`.
    """
    models_root = Path(Config.PATH_TO_MODELS)
    e2e_path      = Path(e2e_checkpoint)      if e2e_checkpoint      else models_root / "e2e_risk_checkpoint.pt"
    composed_path = Path(composed_checkpoint) if composed_checkpoint else models_root / "bdu_gru_search_02.pt"
    yolo_path     = Path(yolo_checkpoint)     if yolo_checkpoint     else models_root / "yolo11s.pt"
    depth_path    = Path(depth_checkpoint)    if depth_checkpoint    else models_root / "depth_anything_v2_vits.pth"
    climate_path  = Path(climate_checkpoint)  if climate_checkpoint  else models_root / "environment_multitask_checkpoint.pt"
    road_path     = Path(road_checkpoint)     if road_checkpoint     else models_root / "road_segmentation_multitask_checkpoint.pt"

    device_name = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    e2e_predictor: QueuedE2ERiskInference | None = None
    if e2e_path.is_file():
        e2e_predictor = QueuedE2ERiskInference.from_checkpoint(
            checkpoint_path=str(e2e_path),
            device=device_name,
            manual_crop_bbox=manual_crop_bbox,
        )

    composed_predictor: QueuedComposedBDUGRURiskInference | None = None
    required = {"composed": composed_path, "yolo": yolo_path, "depth": depth_path}
    if all(p.is_file() for p in required.values()):
        # Lazy imports keep the module importable when these heavy deps are missing.
        from ultralytics import YOLO
        from MIREIA.perception import (
            DepthAnythingV2Estimator,
            create_environment_classifier_predictor,
            load_road_segmentation_model,
        )

        yolo_model = YOLO(str(yolo_path))
        depth_estimator = DepthAnythingV2Estimator(
            checkpoint_path=depth_path,
            encoder="vits",
            device=device_name,
        )
        environment_predictor = (
            create_environment_classifier_predictor(checkpoint_path=str(climate_path), device=device_name)
            if climate_path.is_file() else None
        )
        road_segmentation = (
            load_road_segmentation_model(checkpoint_path=str(road_path), device=device_name)
            if road_path.is_file() else None
        )

        composed_predictor = QueuedComposedBDUGRURiskInference.from_checkpoint(
            checkpoint_path=str(composed_path),
            feature_integrator=FeatureIntegrator(),
            yolo_model=yolo_model,
            depth_estimator=depth_estimator,
            environment_predictor=environment_predictor,
            road_segmentation=road_segmentation,
            device=device_name,
            manual_crop_bbox=manual_crop_bbox,
        )

    return e2e_predictor, composed_predictor


def find_sample_dashcam_frame(search_roots: Sequence[str | Path] | None = None) -> Path | None:
    """Return the path to an existing dashcam frame to preview the crop on.

    Searches `MIREIA/trials/**/runs/**/images/rgb_*.png` and
    `MIREIA/scenarios/**/dataset/images/rgb_*.png` in that order. Returns the
    first match (sorted), or None if no captured frame exists yet.
    """
    if search_roots is None:
        search_roots = [
            Path(Config.PATH_TO_TRIALS),
            Path(Config.PATH_TO_SCENARIOS) if hasattr(Config, "PATH_TO_SCENARIOS") else None,
        ]
    for root in search_roots:
        if root is None:
            continue
        root = Path(root)
        if not root.is_dir():
            continue
        for pattern in ("**/runs/**/images/rgb_*.png", "**/dataset/images/rgb_*.png", "**/images/rgb_*.png"):
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def preview_manual_crop(
    manual_crop_bbox: Sequence[float] | None,
    sample_frame_path: str | Path | None = None,
) -> Path | None:
    """Render a side-by-side preview of (raw frame, cropped frame) with matplotlib.

    Returns the path of the sample frame used, or None if none was found.
    Intended to be called from a Jupyter notebook so the user can iterate on
    `MANUAL_CROP_BBOX_XYXY` before launching a batch.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import Image

    from MIREIA.data_collection.dataset_utils import (
        _clip_bbox_to_image,
        normalize_crop_bbox_xyxy,
    )

    frame_path = Path(sample_frame_path) if sample_frame_path else find_sample_dashcam_frame()
    if frame_path is None or not frame_path.is_file():
        print("[preview] No captured RGB frame found yet. Run any batch once to populate one.")
        return None

    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        raw = img.copy()
        bbox = normalize_crop_bbox_xyxy(manual_crop_bbox)
        clipped = _clip_bbox_to_image(bbox, width=w, height=h) if bbox else None
        cropped = img.crop(clipped) if clipped is not None else img.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(raw)
    axes[0].set_title(f"Raw  ({w} x {h})")
    axes[0].axis("off")
    if clipped is not None:
        x1, y1, x2, y2 = clipped
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor="lime", facecolor="none")
        axes[0].add_patch(rect)

    axes[1].imshow(cropped)
    if clipped is not None:
        cw, ch = clipped[2] - clipped[0], clipped[3] - clipped[1]
        axes[1].set_title(f"Cropped  ({cw} x {ch})  bbox={list(clipped)}")
    else:
        axes[1].set_title("No crop applied (manual_crop_bbox=None)")
    axes[1].axis("off")

    fig.suptitle(f"Crop preview — {frame_path.name}", fontsize=11)
    fig.tight_layout()
    plt.show()
    return frame_path


__all__ = [
    "build_trial_predictors",
    "find_sample_dashcam_frame",
    "preview_manual_crop",
]
