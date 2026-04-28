# MIREIA Perception

ML models and feature extraction pipeline for risk prediction.

## Pipeline Overview

```
RGB frame(s)
    │
    ├─────────────────────────────────────────── E2E path ──┐
    │                                                        │
    ├── YOLOv11s          → obstacle bounding boxes         ResNet18
    ├── Depth Anything V2 → per-pixel metric depth          spatial
    ├── Road Segmentation → road + lane masks               backbone
    ├── EgoMotionEstimator→ frame-to-frame optical flow         │
    └── EnvironmentClassifier → day_prob + 13-class logits      │
            │                                                    │
            ▼                                               visual embed
    FeatureIntegrator                                            │
            │                                                    │
            ▼                                                    ▼
    32D feature vector ──────────────────────────────────► BDU-GRU
                                                           temporal head
    Composed path:                                              │
    perception → 32D → BDU-GRU                                 │
                                                               ▼
    E2E path:                                          risk prediction
    raw RGB → ResNet18 → BDU-GRU
```

## Modules

### Models

| Module | Class | Description |
|--------|-------|-------------|
| `e2e_model.py` | `Seq2SeqRiskPredictor` | **E2E path**: ResNet18 encodes each raw RGB frame into a visual embedding; a BDU-GRU temporal head runs over the embedding sequence to produce risk predictions |
| `bdu_gru_model.py` | `BDUGRURiskPredictor` | **Composed path**: BDU-GRU + MLP operating directly on pre-extracted 32D feature vector sequences (no raw pixels) |
| `climate_model.py` | `MireiaEnvironmentClassifier` | MobileNetV3-small multi-task: day/night binary + 5-class weather |
| `road_segmentation.py` | `MireiaRoadSegmentationModel` | U-Net (MobileNetV3 backbone): dual-head lane + road segmentation |
| `depth.py` | `DepthAnythingV2Estimator` | Depth Anything V2 wrapper (ViT-small, local `.pth` checkpoint) |
| `yolo.py` | `YoloObstacleDetector` | YOLOv11s wrapper filtered to driving-relevant COCO classes |
| `sam2_dashboard.py` | `Sam2DashboardSegmenter` | SAM 2.1 wrapper for dashboard region segmentation |

### Feature Integration

| Module | Class | Description |
|--------|-------|-------------|
| `feature_integration.py` | `FeatureIntegrator` | Assembles the strict 32D feature vector from all perception outputs |
| `flow.py` | `EgoMotionEstimator` | Phase-correlation optical flow for ego motion estimation |

**32D feature vector layout:**
```
[0]     num_objects
[1–4]   bbox_size stats (w, h, area, aspect)
[5–8]   depth stats (mean, min, max, std)
[9–12]  size/depth change (Δsize, Δdepth, Δsize_norm, Δdepth_norm)
[13–16] ego motion (dx, dy, speed, heading_change)
[17–18] threat scores (near, weighted)
[19]    road_size (fraction of image)
[20]    day_prob
[21–31] climate class probabilities (13 classes)
```

### Training

| Module | Function | Description |
|--------|----------|-------------|
| `e2e_model_train.py` | `train_e2e_model` | E2E training with seq_len/burn_in/eval_frames config |
| `bdu_gru_model_train.py` | `train_bdu_gru_model` | BDU-GRU training with dynamic weighted loss options |
| `climate_model_train.py` | `train_environment_model` | Multi-task day/night + weather classifier training |
| `road_segmentation_train.py` | `train_road_segmentation_model` | Dual-head road segmentation training with IoU/Dice metrics |
| `training_utils.py` | — | Shared checkpoint I/O, scenario splits, dataloader builders |

Supported BDU-GRU loss types: `smooth_l1`, `mse`, `dynamic_smooth_l1`, `dynamic_mse`, `dynamic_huber`.

### Inference

| Module | Class | Description |
|--------|-------|-------------|
| `inference.py` | `StreamingRiskPredictor` | Online inference with FIFO frame buffer, burn-in/eval window |
| `queued_inference.py` | `QueuedE2ERiskInference` / `QueuedComposedBDUGRURiskInference` | Batched offline inference with per-stage caching |

## Temporal Configuration

All temporal models use a shared 3-parameter config (set in `Config`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `INFERENCE_SEQUENCE_LENGTH` | 16 | Total frames in the sliding window |
| `INFERENCE_BURN_IN_FRAMES` | 12 | Context frames (GRU warm-up, no loss) |
| `INFERENCE_EVAL_FRAMES` | 4 | Prediction frames (loss computed here) |

Constraint: `burn_in + eval == seq_len`.
