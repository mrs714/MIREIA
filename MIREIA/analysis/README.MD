# MIREIA Analysis

Visualisation tooling for the Driving Risk Field and for closed-loop trial runs: top-down risk heatmaps, per-run dataset videos, and side-by-side trial comparison videos.

## Modules

| Module | Key class / function | Description |
|--------|----------------------|-------------|
| `plotter.py` | `Grid`, `RiskGrid` | Coordinate-based grids over a square region (centre, size, resolution). `RiskGrid` stores per-cell risk values and is what the oracle and the visualiser exchange. |
| `plotter.py` | `draw_risk_heatmap_3d` | Optional 3-D surface rendering of a `RiskGrid` (matplotlib). |
| `visualizer.py` | `RiskGridVisualizer` | Renders a `RiskGrid` as a top-down heatmap with the ego vehicle, dynamic obstacles, static obstacles and pedestrians overlaid. Supports per-frame rendering and multi-frame MP4 export. |
| `visualizer.py` | `render_static_risk_map` | One-shot rendering of a pre-baked static-risk grid. |
| `visualizer.py` | `render_risk_map_with_actors` | One-shot rendering of a live `RiskGrid` with actors overlaid (no video loop). |
| `visualizer.py` | `DatasetVideoComposer` / `compose_trial_dataset_video` | Builds a tiled per-run video (dashcam + top-down + risk graph) from a `dataset.jsonl`. |
| `visualizer.py` | `TrialComparisonVideoComposer` / `compose_trial_comparison_video` | Builds a 2×2 side-by-side comparison video between two run JSONLs. |
| `visualizer.py` | `DummyRiskVisualizer` | No-op visualiser used when rendering is disabled (e.g. in batch runners). |

## Usage

### Top-down risk-grid rendering

`RiskGridVisualizer` is the class used in [`NB_02_risk_field_validation.ipynb`](../../NB_02_risk_field_validation.ipynb) to produce the qualitative top-down heatmaps in §4.3 of the thesis.

```python
from MIREIA.analysis.plotter import Grid
from MIREIA.analysis.visualizer import RiskGridVisualizer
from MIREIA.core.physics import RiskOracle

oracle = RiskOracle()
grid = Grid(center_x=ego_x, center_y=ego_y, size=100.0, resolution=0.25)
risk_grid = oracle.calculate_risk_map(grid, bridge, baked_static_risk=baked_static_risk)

viz = RiskGridVisualizer(risk_grid, world, bridge, oracle=oracle, vmax=2.0)
viz.render_video("output/risk_video.mp4", n_frames=300,
                 grid_size=200.0, grid_resolution=1.0,
                 baked_static_risk=baked_static_risk)
```

`vmax` fixes the colour-scale upper bound across frames; omit it to auto-scale per frame.

### Per-run dataset video

```python
from MIREIA.analysis.visualizer import compose_trial_dataset_video

compose_trial_dataset_video(
    jsonl_path="MIREIA/trials/<trial>/runs/<run>/dataset.jsonl",
    output_name="dataset_video.mp4",
    fps=4,
    calculated_risk_key="predicted_risk",
)
```

Used by [`NB_09_trial_demo.ipynb`](../../NB_09_trial_demo.ipynb) and [`NB_14_trial_analysis.ipynb`](../../NB_14_trial_analysis.ipynb) to assemble dashcam + top-down + risk-graph videos for a single run.

### Trial comparison video

```python
from MIREIA.analysis.visualizer import compose_trial_comparison_video

compose_trial_comparison_video(
    jsonl_path_a="MIREIA/trials/<trial>/runs/<base_run>/dataset.jsonl",
    jsonl_path_b="MIREIA/trials/<trial>/runs/<other_run>/dataset.jsonl",
    output_name="comparison.mp4",
    fps=4,
    label_a="base_speed",
    label_b="risk_aware",
    calculated_risk_key="predicted_risk",
)
```

Used by [`NB_15_trial_comparison.ipynb`](../../NB_15_trial_comparison.ipynb) to produce the side-by-side videos shown in §6 of the thesis.

Both composers read `rgb_image_path`, `topdown_image_path`, `ground_truth_risk`, and optionally `predicted_risk` from the JSONL records.
