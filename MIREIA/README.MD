# MIREIA

**M**odel for **I**ntelligent **R**isk **E**valuation in **I**mperfect **A**utonomous driving — vision algorithms for real-time risk prediction in CARLA simulations.

## Architecture

```
CARLA Simulator
      │
      ▼
 WorldManager ──── SimulationBridge
      │                  │
      │         per-tick state snapshot
      │         (ego, obstacles, env)
      │
      ▼
 RiskOracle (DRF) ──► ground_truth_risk → DatasetLogger → dataset.jsonl
      │
      ▼
 SensorManager ──► RGB frames → dataset/images/

──────── Offline: Feature Extraction (train_demo.ipynb Step 9) ────────

 RGB frame pairs
      │
      ├── YOLOv11s ──────────► obstacle boxes
      ├── Depth Anything V2 ─► metric depth
      ├── Road Segmentation ─► road + lane masks
      ├── EgoMotionEstimator ► optical flow
      └── EnvironmentClassifier ► day_prob + weather
              │
              ▼
       FeatureIntegrator → 32D vector → dataset_labeled.jsonl

──────── Training ────────

 dataset_labeled.jsonl ──► BDU-GRU training  ──► bdu_gru_*.pt
 dataset.jsonl         ──► E2E training       ──► e2e_risk_*.pt
 HuggingFace dataset   ──► Road Seg training  ──► road_segmentation_*.pt
 CARLA scenarios       ──► Climate training   ──► environment_multitask_*.pt

──────── Inference ────────

 RGB stream ──► QueuedE2ERiskInference              (raw pixels → risk)
             ──► QueuedComposedBDUGRURiskInference  (perception → 32D → risk)
```

## Package Layout

| Directory | Description |
|-----------|-------------|
| [`core/`](core/README.MD) | Physics — Driving Risk Field (DRF), constants, geometry |
| [`perception/`](perception/README.MD) | ML models, feature integration, training, inference |
| [`simulation/`](simulation/README.MD) | CARLA integration — world, scenarios, sensors, trials |
| [`data_collection/`](data_collection/README.MD) | Dataset recording, JSONL labeling, DataLoaders |
| [`analysis/`](analysis/README.MD) | Risk visualization, comparison and dataset videos |
| [`models/`](models/README.MD) | Trained checkpoints and external weights |
| [`scenarios/`](scenarios/) | 64 CARLA scenario definitions (train + val + holdout) |
| [`trials/`](trials/) | Trial run output (per-run datasets and metadata) |

## Configuration

All paths and runtime defaults live in `config.py`, loaded from `.env` (copy `.env.example` to get started):

```env
PATH_TO_SCENARIOS=/path/to/MIREIA/scenarios
PATH_TO_TRIALS=/path/to/MIREIA/trials
PATH_TO_MODELS=/path/to/MIREIA/models
CARLA_HOST=127.0.0.1
CARLA_PORT=2000
```

## Scenario Dataset

64 scenarios across 4 towns, 5 weather conditions, day/night, and two traffic volumes:

| Split | Sets | Town(s) |
|-------|------|---------|
| Train | 01–16 | Town01, Town02, Town03, Town04 |
| Val | 17–21 | Town05 |
| Holdout | 22+ | Town10HD |

Each scenario slot has 4 variants (letters A–D) covering high/low traffic and alternate weather.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `model_demo.ipynb` | Individual model inference demo (YOLO, Depth, SAM2, Road Seg, Climate, E2E) |
| `feature_analysis.ipynb` | PCA, clustering, and diagnostics of the 32D feature space |
| `scenario_demo.ipynb` | Generate, save, and run CARLA scenarios; capture datasets |
| `train_demo.ipynb` | Full training pipeline (Climate → E2E → Road Seg → Speed Fusion → BDU-GRU search) |
| `queued_inference_demo.ipynb` | Offline inference comparison: E2E vs Composed BDU-GRU |
| `composed_inference_analysis.ipynb` | Benchmark composed model latency, FPS, and call counts |
| `trial_demo.ipynb` | Live CARLA trial execution with streaming predictor and route comparison |
