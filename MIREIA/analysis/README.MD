# MIREIA Analysis

Visualization tools for risk grids, trial comparisons, and dataset videos.

## Modules

| Module | Key Class / Function | Description |
|--------|---------------------|-------------|
| `plotter.py` | `RiskGrid` | Coordinate-based risk grid backed by a numpy meshgrid; supports top-down map rendering of risk values |
| `visualizer.py` | `RiskGridVisualizer` | Renders risk heatmaps overlaid with ego vehicle and obstacle positions; produces per-frame images and MP4 videos |
| `visualizer.py` | `compose_trial_comparison_video` | Builds a 2×2 side-by-side video comparing two trial sub-runs |
| `visualizer.py` | `compose_trial_dataset_video` | Builds a tiled video (dashcam + top-down + risk graph) from a single run JSONL |

## Usage

### Risk grid rendering

```python
from MIREIA.analysis.visualizer import RiskGridVisualizer

viz = RiskGridVisualizer(world_bounds=..., resolution=2.0)
frame = viz.render_frame(ego_pos, obstacles, risk_grid)
```

### Trial comparison video

```python
from MIREIA.analysis.visualizer import compose_trial_comparison_video

compose_trial_comparison_video(
    jsonl_path_a="trials/run_a/dataset.jsonl",
    jsonl_path_b="trials/run_b/dataset.jsonl",
    output_name="comparison.mp4",
    fps=4,
    label_a="base_speed",
    label_b="fast_speed",
    calculated_risk_key="predicted_risk",
)
```

### Dataset video

```python
from MIREIA.analysis.visualizer import compose_trial_dataset_video

compose_trial_dataset_video(
    jsonl_path="trials/run/dataset.jsonl",
    output_name="dataset_video.mp4",
    fps=4,
    calculated_risk_key="predicted_risk",
)
```

Both video composers read `rgb_image_path`, `topdown_image_path`, `ground_truth_risk`, and optionally `predicted_risk` from the JSONL records.
