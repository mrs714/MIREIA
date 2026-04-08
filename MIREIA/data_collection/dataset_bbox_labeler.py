from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from MIREIA.config import Config
from MIREIA.data_collection.dataset_utils import (
    load_jsonl_records,
    normalize_crop_bbox_xyxy,
    resolve_image_path,
)


@dataclass(frozen=True)
class DatasetLabelSummary:
    scenario_name: str
    output_jsonl_path: str
    bbox_xyxy: tuple[int, int, int, int]
    bbox_source: str
    num_records: int


def _clip_bbox_to_image(
    bbox_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return (0, 0, width, height)

    return (x1, y1, x2, y2)


def _first_valid_record_path(records: list[dict], image_root: str) -> tuple[dict, str]:
    for record in records:
        rel_path = str(record.get("rgb_image_path", "")).strip()
        if not rel_path:
            continue

        image_path = resolve_image_path(image_root=image_root, rel_path=rel_path, normalize_paths=True)
        if os.path.isfile(image_path):
            return record, image_path

    raise FileNotFoundError("No valid rgb_image_path entries found in source JSONL")


def _bbox_from_sam(
    anchor_image_path: str,
    sam_segmenter: Any,
    bbox_kind: str,
    sam_instruction: str | None,
    min_mask_area_ratio: float,
) -> tuple[int, int, int, int]:
    with Image.open(anchor_image_path) as image:
        width, height = image.size

    result = sam_segmenter.create_dash_bb(
        source=anchor_image_path,
        instruction=sam_instruction,
        min_mask_area_ratio=min_mask_area_ratio,
    )

    kind = bbox_kind.strip().lower()
    if kind == "inverse":
        bbox_obj = result.inverse_bbox
    elif kind == "dashboard":
        bbox_obj = result.dashboard_bbox
    else:
        raise ValueError("bbox_kind must be 'inverse' or 'dashboard'")

    if bbox_obj is None:
        return (0, 0, width, height)

    return _clip_bbox_to_image(
        bbox_xyxy=(bbox_obj.x1, bbox_obj.y1, bbox_obj.x2, bbox_obj.y2),
        width=width,
        height=height,
    )


def label_scenario_dataset_with_bbox(
    scenario_dir: str,
    mode: str = "sam",
    manual_bbox_xyxy: Sequence[float] | None = None,
    source_jsonl_name: str = "dataset.jsonl",
    output_jsonl_name: str = "dataset_labeled.jsonl",
    bbox_key: str = "crop_bbox_xyxy",
    bbox_kind: str = "inverse",
    sam_segmenter: Any | None = None,
    sam_checkpoint_path: str | Path | None = None,
    sam_model_cfg: str | Path | None = None,
    sam_device: str | None = None,
    sam_instruction: str | None = None,
    min_mask_area_ratio: float = 0.001,
) -> DatasetLabelSummary:
    scenario_path = Path(scenario_dir)
    if not scenario_path.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_path}")

    source_jsonl_path = scenario_path / source_jsonl_name
    if not source_jsonl_path.is_file():
        raise FileNotFoundError(f"Source JSONL not found: {source_jsonl_path}")

    records = load_jsonl_records(str(source_jsonl_path))
    if not records:
        raise ValueError(f"Source JSONL has no records: {source_jsonl_path}")

    anchor_record, anchor_image_path = _first_valid_record_path(records, str(scenario_path))

    with Image.open(anchor_image_path) as image:
        image_width, image_height = image.size

    normalized_manual_bbox = normalize_crop_bbox_xyxy(manual_bbox_xyxy)

    mode_norm = mode.strip().lower()
    if mode_norm == "manual":
        if normalized_manual_bbox is None:
            raise ValueError("manual_bbox_xyxy must be provided when mode='manual'")
        bbox_xyxy = _clip_bbox_to_image(
            bbox_xyxy=normalized_manual_bbox,
            width=image_width,
            height=image_height,
        )
        bbox_source = "manual"
    elif mode_norm == "sam":
        if sam_segmenter is None:
            from MIREIA.perception.sam2_dashboard import Sam2DashboardSegmenter

            sam_segmenter = Sam2DashboardSegmenter(
                checkpoint_path=sam_checkpoint_path,
                model_cfg=sam_model_cfg,
                device=sam_device,
            )

        bbox_xyxy = _bbox_from_sam(
            anchor_image_path=anchor_image_path,
            sam_segmenter=sam_segmenter,
            bbox_kind=bbox_kind,
            sam_instruction=sam_instruction,
            min_mask_area_ratio=min_mask_area_ratio,
        )
        bbox_source = "sam"
    else:
        raise ValueError("mode must be 'sam' or 'manual'")

    output_jsonl_path = scenario_path / output_jsonl_name
    anchor_frame_id = int(anchor_record.get("frame_id", 0))

    with open(output_jsonl_path, "w", encoding="utf-8") as handle:
        for record in records:
            updated = dict(record)
            updated[bbox_key] = [int(v) for v in bbox_xyxy]
            updated["crop_bbox_source"] = bbox_source
            updated["crop_bbox_kind"] = bbox_kind
            updated["crop_bbox_anchor_frame_id"] = anchor_frame_id
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")

    return DatasetLabelSummary(
        scenario_name=scenario_path.name,
        output_jsonl_path=str(output_jsonl_path),
        bbox_xyxy=bbox_xyxy,
        bbox_source=bbox_source,
        num_records=len(records),
    )


def label_all_scenarios_datasets_with_bbox(
    scenarios_root: str | None = None,
    mode: str = "sam",
    manual_bbox_xyxy: Sequence[float] | None = None,
    source_jsonl_name: str = "dataset.jsonl",
    output_jsonl_name: str = "dataset_labeled.jsonl",
    bbox_key: str = "crop_bbox_xyxy",
    bbox_kind: str = "inverse",
    sam_checkpoint_path: str | Path | None = None,
    sam_model_cfg: str | Path | None = None,
    sam_device: str | None = None,
    sam_instruction: str | None = None,
    min_mask_area_ratio: float = 0.001,
    include_scenarios: Sequence[str] | None = None,
    exclude_scenarios: Sequence[str] | None = None,
) -> list[DatasetLabelSummary]:
    root = Path(scenarios_root or Config.PATH_TO_SCENARIOS)
    if not root.is_dir():
        raise FileNotFoundError(f"Scenarios root not found: {root}")

    include_set = set(include_scenarios or [])
    exclude_set = set(exclude_scenarios or [])

    mode_norm = mode.strip().lower()
    sam_segmenter = None
    if mode_norm == "sam":
        from MIREIA.perception.sam2_dashboard import Sam2DashboardSegmenter

        sam_segmenter = Sam2DashboardSegmenter(
            checkpoint_path=sam_checkpoint_path,
            model_cfg=sam_model_cfg,
            device=sam_device,
        )

    summaries: list[DatasetLabelSummary] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"videos", "__pycache__"}:
            continue
        if include_set and entry.name not in include_set:
            continue
        if entry.name in exclude_set:
            continue

        source_jsonl_path = entry / source_jsonl_name
        if not source_jsonl_path.is_file():
            continue

        summary = label_scenario_dataset_with_bbox(
            scenario_dir=str(entry),
            mode=mode_norm,
            manual_bbox_xyxy=manual_bbox_xyxy,
            source_jsonl_name=source_jsonl_name,
            output_jsonl_name=output_jsonl_name,
            bbox_key=bbox_key,
            bbox_kind=bbox_kind,
            sam_segmenter=sam_segmenter,
            sam_checkpoint_path=sam_checkpoint_path,
            sam_model_cfg=sam_model_cfg,
            sam_device=sam_device,
            sam_instruction=sam_instruction,
            min_mask_area_ratio=min_mask_area_ratio,
        )
        summaries.append(summary)

    return summaries


def _parse_manual_bbox(raw: str | None) -> tuple[int, int, int, int] | None:
    if raw is None:
        return None
    parts = [token.strip() for token in raw.split(",") if token.strip()]
    if len(parts) != 4:
        raise ValueError("--manual-bbox must have four comma-separated values: x1,y1,x2,y2")
    return tuple(int(float(v)) for v in parts)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label scenario datasets with a fixed crop bbox")
    parser.add_argument("--scenarios-root", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["sam", "manual"], default="sam")
    parser.add_argument("--manual-bbox", type=str, default=None, help="x1,y1,x2,y2 for mode=manual")
    parser.add_argument("--source-jsonl", type=str, default="dataset.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="dataset_labeled.jsonl")
    parser.add_argument("--bbox-key", type=str, default="crop_bbox_xyxy")
    parser.add_argument("--bbox-kind", type=str, choices=["inverse", "dashboard"], default="inverse")
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--sam-model-cfg", type=str, default=None)
    parser.add_argument("--sam-device", type=str, default=None)
    parser.add_argument("--sam-instruction", type=str, default=None)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.001)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    manual_bbox = _parse_manual_bbox(args.manual_bbox)

    summaries = label_all_scenarios_datasets_with_bbox(
        scenarios_root=args.scenarios_root,
        mode=args.mode,
        manual_bbox_xyxy=manual_bbox,
        source_jsonl_name=args.source_jsonl,
        output_jsonl_name=args.output_jsonl,
        bbox_key=args.bbox_key,
        bbox_kind=args.bbox_kind,
        sam_checkpoint_path=args.sam_checkpoint,
        sam_model_cfg=args.sam_model_cfg,
        sam_device=args.sam_device,
        sam_instruction=args.sam_instruction,
        min_mask_area_ratio=args.min_mask_area_ratio,
    )

    print(f"Labeled scenarios: {len(summaries)}")
    for summary in summaries:
        print(
            f" - {summary.scenario_name}: records={summary.num_records}, "
            f"bbox={summary.bbox_xyxy}, output={summary.output_jsonl_path}"
        )


if __name__ == "__main__":
    main()
