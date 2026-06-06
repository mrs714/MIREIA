# MIREIA

**M**onocular **I**nterpretable **R**isk **E**valuation for **I**ntelligent **A**utonomy — vision algorithms for real-time risk prediction in CARLA simulations.

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

──────── Offline: Feature Extraction (NB_04_training_pipeline.ipynb Step 9) ────────

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

The 15 workflow notebooks live at the repository root and are prefixed `NB_NN_`. See the [root README](../README.md) for the full ordered list and per-notebook purpose. In summary:

| Stage | Notebooks |
|-------|-----------|
| Data | `NB_01_scenario_demo` |
| Risk-field validation | `NB_02_risk_field_validation` |
| Perception & training | `NB_03_perception_modules_demo`, `NB_04_training_pipeline` |
| Analysis & inference | `NB_05_feature_analysis`, `NB_06_queued_inference_demo`, `NB_07_composed_inference_analysis` |
| Closed-loop trials | `NB_08_trial_builder`, `NB_09_trial_demo`, `NB_10`–`NB_13_*_batch_runner` |
| Aggregate analysis | `NB_14_trial_analysis`, `NB_15_trial_comparison` |
