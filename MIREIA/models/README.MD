# MIREIA Models

Trained checkpoints and auxiliary weights used by the MIREIA perception pipeline.

## Risk Predictors

| File | Size | Description |
|------|------|-------------|
| `e2e_risk_checkpoint.pt` | ~170 MB | Main E2E risk model (ResNet18 + BDU-GRU, seq2seq) |
| `e2e_risk_checkpoint_1full_epoch.pt` | ~170 MB | E2E variant trained for 1 full epoch |
| `e2e_risk_checkpoint_high_accum.pt` | ~170 MB | E2E variant with high gradient accumulation |
| `e2e_risk_checkpoint_low_accum.pt` | ~170 MB | E2E variant with low gradient accumulation |
| `e2e_risk_checkpoint_old.pt` | ~170 MB | Older E2E checkpoint (kept for comparison) |
| `bdu_gru_risk_checkpoint_stable.pt` | ~21 MB | Stable BDU-GRU checkpoint (32D feature input) |
| `bdu_gru_search_01.pt` … `bdu_gru_search_08.pt` | ~21 MB each | BDU-GRU hyperparameter search variants |

## Perception Models

| File | Size | Description |
|------|------|-------------|
| `environment_multitask_checkpoint.pt` | ~13 MB | Day/night + 5-class weather classifier (MobileNetV3-small) |
| `road_segmentation_multitask_checkpoint.pt` | ~23 MB | Road + lane segmentation (latest checkpoint) |
| `road_segmentation_multitask_best.pt` | ~23 MB | Road + lane segmentation (best mean IoU checkpoint) |
| `speed_fusion_checkpoint.pt` | ~2.3 MB | Two-frame ego speed regressor (experimental) |

## External Weights

| File | Size | Description |
|------|------|-------------|
| `yolo11s.pt` | ~19 MB | YOLOv11-small object detector (Ultralytics) |
| `depth_anything_v2_vits.pth` | ~99 MB | Depth Anything V2 — ViT-small encoder |
| `sam2.1_hiera_small.pt` | ~184 MB | SAM 2.1 — Hiera-small instance segmentation |

## Training Split Controls

Used in `train_demo.ipynb` and `train_bdu_gru_model` / `train_e2e_model`.

### `partition_mode`

- `"scenario"` — validation set is a subset of named scenarios; all frames from those scenarios go to val.
- `"frame"` — all scenarios are pooled; frames are split by index order (first N% train, rest val).

### `val_scenario_tokens`

Comma-separated tokens (or a list) matched against scenario folder names. A scenario is assigned to val if its name contains any token.

```
val_scenario_tokens=["Town05"]   # Town05 scenarios → validation
val_scenario_tokens=["Town10HD"] # Town10HD scenarios → validation (default)
```

### `frame_train_ratio`

Fraction of frames used for training when `partition_mode="frame"`. Accepts floats (0.7) or percentages (70).

### Examples

```python
# Scenario-token split — Town05 as validation
train_bdu_gru_model(partition_mode="scenario", val_scenario_tokens=["Town05"])

# Frame-based split over all scenarios, 70/30
train_bdu_gru_model(partition_mode="frame", frame_train_ratio=0.7)
```

## Dataset Layout Alignment

The BDU-GRU models expect `dataset_labeled.jsonl` (with `feature_vector_32` keys) in each scenario folder.
The E2E models expect `dataset.jsonl` (with `rgb_image_path` keys) in each scenario folder.
Run `train_demo.ipynb` → Step 9 to generate `dataset_labeled.jsonl` from raw `dataset.jsonl`.
