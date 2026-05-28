# MIREIA

**M**achine-vision **I**ntelligence for **R**isk **E**valuation in **I**nteractive **A**utonomy — a monocular RGB-based Driving Risk Field (DRF) estimator and a risk-aware speed controller, built on the CARLA simulator.

This repository contains the full system: the analytical risk oracle that generates the ground-truth labels, two learned risk predictors (an end-to-end ResNet-18 → BDU-GRU and a modular composed model), the closed-loop trial machinery that drives them, and the analysis tooling used to compare runs.

## Quickstart

1. Install Python 3.12 and CARLA 0.9.16 (UE 4.26 build), then `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and set `PATH_TO_SCENARIOS`, `PATH_TO_TRIALS`, `PATH_TO_MODELS`, `CARLA_HOST`, `CARLA_PORT`.
3. Launch CARLA, then open the notebooks below in order.

## Notebooks

The notebooks at the repository root are numbered to reflect the workflow stage they belong to: data → validation → perception → training → inference → trials → analysis.

| # | Notebook | Stage | Purpose |
|---|----------|-------|---------|
| 01 | [`NB_01_scenario_demo.ipynb`](NB_01_scenario_demo.ipynb) | Data | Build a `Scenario`, run it in synchronous mode, capture an RGB + ground-truth-risk dataset. |
| 02 | [`NB_02_risk_field_validation.ipynb`](NB_02_risk_field_validation.ipynb) | Validation | Drive the `RiskOracle` against a live world and render top-down DRF heatmaps. Qualitative evidence for §4.2–4.3 of the thesis. |
| 03 | [`NB_03_perception_modules_demo.ipynb`](NB_03_perception_modules_demo.ipynb) | Perception | Single-frame demo of every perception module (YOLO, Depth, RoadSeg, Climate, SAM2, Phase-correlation flow, E2E). |
| 04 | [`NB_04_training_pipeline.ipynb`](NB_04_training_pipeline.ipynb) | Training | Full training driver: Climate → E2E → RoadSeg → Speed fusion → Dataset labelling → BDU-GRU hyperparameter search. |
| 05 | [`NB_05_feature_analysis.ipynb`](NB_05_feature_analysis.ipynb) | Analysis | PCA, clustering, and correlation diagnostics of the 32-D feature space. |
| 06 | [`NB_06_queued_inference_demo.ipynb`](NB_06_queued_inference_demo.ipynb) | Inference | Offline replay: E2E vs Composed model overlaid on ground-truth risk. |
| 07 | [`NB_07_composed_inference_analysis.ipynb`](NB_07_composed_inference_analysis.ipynb) | Inference | Per-stage timing, call-count profiling, and FPS comparison of the composed model. |
| 08 | [`NB_08_trial_builder.ipynb`](NB_08_trial_builder.ipynb) | Trials | Author and persist `TrialDefinition` objects with an interactive waypoint picker. |
| 09 | [`NB_09_trial_demo.ipynb`](NB_09_trial_demo.ipynb) | Trials | Single live trial run with baseline + streaming-predictor variants. |
| 10 | [`NB_10_trial_batch_runner.ipynb`](NB_10_trial_batch_runner.ipynb) | Trials | Sweep: `base` subtrial per trial (ground-truth-driven, no speed override). |
| 11 | [`NB_11_trial_slow_batch_runner.ipynb`](NB_11_trial_slow_batch_runner.ipynb) | Trials | Sweep: `slow` subtrial with a constant speed multiplier. |
| 12 | [`NB_12_trial_function_batch_runner.ipynb`](NB_12_trial_function_batch_runner.ipynb) | Trials | Sweep: oracle-in-the-loop risk-aware speed control. |
| 13 | [`NB_13_trial_models_batch_runner.ipynb`](NB_13_trial_models_batch_runner.ipynb) | Trials | Sweep: per-model risk-aware speed control (headline closed-loop result). |
| 14 | [`NB_14_trial_analysis.ipynb`](NB_14_trial_analysis.ipynb) | Analysis | Per-run visualisation: compile videos and plot the per-tick risk trace. |
| 15 | [`NB_15_trial_comparison.ipynb`](NB_15_trial_comparison.ipynb) | Analysis | Aggregate validation / test comparison: tables, route plots, efficiency bar charts, 3×3 comparison videos. |

Each notebook opens with a header cell stating its purpose, inputs, outputs, how to run it, and where it sits in the workflow. The `test_dashcam/` folder holds two real-dashcam preprocessing / inference notebooks that live outside the main pipeline.

## Repository Layout

| Path | Contents |
|------|----------|
| [`MIREIA/`](MIREIA/README.MD) | The Python package: `core` (DRF physics), `perception` (models + integrator), `simulation` (CARLA bridge), `data_collection`, `analysis`, `models`. |
| `NB_*.ipynb` | The 15 workflow notebooks listed above. |
| [`test_dashcam/`](test_dashcam/) | Real dashcam preprocessing and inference notebooks (out-of-distribution check). |
| [`tfg/`](tfg/) | The thesis source (`main.tex`, `refs.bib`, diagrams). |
| [`PythonAPI/`](PythonAPI/) | CARLA Python API utilities and examples. |

See [`MIREIA/README.MD`](MIREIA/README.MD) for the architecture diagram, per-subpackage documentation, and the scenario / trial dataset layout.
