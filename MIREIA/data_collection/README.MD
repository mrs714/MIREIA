# MIREIA Data Collection

Dataset recording, labeling, loading, and split management.

## Modules

| Module | Key Class / Function | Description |
|--------|---------------------|-------------|
| `recorder.py` | `DatasetLogger` | Append-only JSONL writer; logs one line per captured frame with RGB path, risk label, ego state, sensor data, and metadata |
| `dataset_utils.py` | `BaseSequenceDataset` | Shared utilities: image loading, path normalization, validation-token parsing, partition-mode handling |
| `database.py` | `create_dataloaders` | Creates train/val DataLoaders; dispatches to scenario-name or frame-index split logic |
| `dataset_bbox_labeler.py` | `label_scenario_dataset_with_bbox` | Adds `crop_bbox_xyxy` and optional dashboard mask coords to scenario JSONL files |
| `feature_sequence_dataset.py` | `ScenarioFeatureSequenceDataset` | Loads pre-extracted 32D feature vectors from `dataset_labeled.jsonl` as temporal sequences for BDU-GRU training |
| `jsonl_sequence_dataset.py` | `JsonlSequenceDataset` | Generic temporal sequence loader from JSONL (with optional crop-bbox support) — used by E2E training |
| `scenario_multitask_dataset.py` | `ScenarioEnvironmentSource` | Loads RGB frames + day/weather labels for environment classifier training |
| `inference_loader.py` | `InferenceFrameLoader` | Single-frame RGB loader with transform pipeline for online inference |

## JSONL Format

### `dataset.jsonl` (raw — produced by simulator)

```jsonc
{
  "frame_id": 150,
  "rgb_image_path": "dataset/images/rgb_000150.png",
  "ground_truth_risk": 0.23,
  "ego": { "x": 12.3, "y": -45.1, "speed": 8.4, "heading": 91.2 },
  "environment": { "weather": "ClearNoon", "day_prob": 0.98 }
}
```

### `dataset_labeled.jsonl` (labeled — produced by Step 9 of `train_demo.ipynb`)

Extends each record with:

```jsonc
{
  "crop_bbox_xyxy": [0, 305, 512, 512],           // dashboard mask region
  "feature_vector_32": [2.0, 0.08, ..., 0.0],    // 32D perception feature vector
  "feature_vector_source": "all_models_masked"    // extraction method tag
}
```

## Dataset Split Modes

### Scenario-based split (`partition_mode="scenario"`)

Scenarios whose names contain any `val_scenario_tokens` substring go entirely to validation. All frames from those scenarios are excluded from training. Used for strict domain generalization evaluation.

```python
create_dataloaders(partition_mode="scenario", val_scenario_tokens=["Town05"])
```

### Frame-based split (`partition_mode="frame"`)

All scenarios are pooled. The first `frame_train_ratio` fraction of frames (by index) goes to training, the rest to validation. No scenario-level leakage prevention.

```python
create_dataloaders(partition_mode="frame", frame_train_ratio=0.7)
```

## Dashboard Masking

Step 9 of `train_demo.ipynb` applies a per-scenario letter mask before feature extraction to black out the dashboard area (which varies by camera mount/letter slot):

| Letter | Mask region (x1, y1, x2, y2) |
|--------|-------------------------------|
| A | `(0, 305, 512, 512)` |
| B | `(0, 320, 512, 512)` |
| C | `(0, 300, 512, 512)` |
| D | `(0, 440, 512, 512)` |

The masked frames are used for YOLO, depth, and road segmentation inference. The feature vector is written to `dataset_labeled.jsonl` with `feature_vector_source="all_models_masked"`.
