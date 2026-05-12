"""Cross-run comparison helpers for trial batch outputs.

Given a single `TrialDefinition` that has multiple `runs/<timestamp>_<prefix>/`
subfolders (produced by the three batch runners), this module loads each run's
JSONL, computes summary metrics, and renders comparison plots/videos.

Prefixes recognised: `base`, `very_slow`, `slow`, `half_slow`, `e2e`,
`composed`. Any prefix that is missing for a trial leaves a blank tile in the
grids so the comparison still renders.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.
_np_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# Run prefixes recognised by the comparison notebook. Order is the column-major
# scan order through `RUN_GRID_LAYOUT` so any consumer that wants a flat list
# can iterate this tuple directly.
RUN_PREFIXES: tuple[str, ...] = (
    "base",
    "e2e",
    "composed",
    "half_slow",
    "slow",
    "very_slow",
    "riskfn1",
    "riskfn2",
    "riskfn3",
)

# 3x3 grid layout used by every visual artifact. Row-major:
#   row 0: baseline + risk-aware models (predicted risk)
#   row 1: constant speed ladder, faster -> slowest
#   row 2: ground-truth-risk-driven v functions (riskfn1..3)
RUN_GRID_LAYOUT: tuple[tuple[str, ...], ...] = (
    ("base", "e2e", "composed"),
    ("half_slow", "slow", "very_slow"),
    ("riskfn1", "riskfn2", "riskfn3"),
)

# Maps that belong to the test/val split per MIREIA/todo.
TEST_VAL_MAP_NAMES: tuple[str, ...] = ("Town04", "Town10HD")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def is_test_or_val_trial(trial_name: str, *, map_name: str | None = None) -> bool:
    """Heuristic: trial is in the test+val set if its map is Town04 or Town10HD.

    Either pass the explicit `map_name` (from the trial JSON) or match the
    `auto_*` trial-name convention which embeds the map.
    """
    if map_name is not None:
        return map_name in TEST_VAL_MAP_NAMES
    return any(name in trial_name for name in TEST_VAL_MAP_NAMES)


def find_latest_run_by_prefix(trial_dir: Path, prefix: str) -> Path | None:
    """Return the most recent `runs/<timestamp>_<prefix>/` for a trial, or None.

    Run folder names are `YYYYMMDD_HHMMSS_<prefix>`.  Splitting on the first two
    underscores gives [date, time, prefix] so a prefix like "slow" never
    accidentally matches "half_slow" or "very_slow".
    """
    runs_root = trial_dir / "runs"
    if not runs_root.is_dir():
        return None

    def _prefix_matches(p: Path) -> bool:
        parts = p.name.split("_", 2)
        return len(parts) == 3 and parts[2] == prefix

    matches = sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and _prefix_matches(p)),
        reverse=True,
    )
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    prefix: str
    run_path: Path
    records: list[dict]
    distances: np.ndarray         # cumulative distance per frame (meters), shape (N,)
    risk_gt: np.ndarray           # ground-truth risk per frame, shape (N,)
    speed_kmh: np.ndarray         # actual ego speed per frame in km/h, shape (N,)
    target_speed_kmh: np.ndarray  # target_speed_kmh logged by the controller, shape (N,)
    positions_xy: np.ndarray      # (N, 2) ego x,y in CARLA coordinates
    pred_risk: dict[str, np.ndarray] = field(default_factory=dict)
    sample_dt: float = 0.05       # fixed_delta * image_stride


def load_run_data(run_path: Path, prefix: str | None = None) -> RunData | None:
    """Load a single run's JSONL and compute the per-frame arrays we need.

    Returns None if `dataset.jsonl` is missing or empty.
    """
    if prefix is None:
        prefix = run_path.name.rsplit("_", 1)[-1]

    jsonl_path = run_path / "dataset.jsonl"
    if not jsonl_path.is_file():
        return None

    # Late import to avoid circular deps when this module is imported eagerly.
    from MIREIA.config import Config
    from MIREIA.data_collection.dataset_utils import load_jsonl_records

    records = load_jsonl_records(str(jsonl_path))
    if not records:
        return None

    positions: list[tuple[float, float]] = []
    speeds: list[float] = []
    target_speeds: list[float] = []
    risks: list[float] = []

    for rec in records:
        pos = rec.get("ego", {}).get("position", {}) if isinstance(rec.get("ego"), dict) else {}
        positions.append((float(pos.get("x", 0.0)), float(pos.get("y", 0.0))))

        ego = rec.get("ego", {}) if isinstance(rec.get("ego"), dict) else {}
        speed_mps = float(ego.get("speed", 0.0))
        speeds.append(speed_mps * 3.6)

        target_speeds.append(float(rec.get("target_speed_kmh", float("nan"))))
        risks.append(float(rec.get("ground_truth_risk", 0.0)))

    positions_xy = np.array(positions, dtype=np.float64)
    if positions_xy.shape[0] >= 2:
        seg = np.linalg.norm(np.diff(positions_xy, axis=0), axis=1)
        distances = np.concatenate([[0.0], np.cumsum(seg)])
    else:
        distances = np.array([0.0], dtype=np.float64)

    pred_risk: dict[str, np.ndarray] = {}
    for col in ("predicted_risk", "predicted_risk_e2e", "predicted_risk_composed"):
        vals = np.array(
            [
                float(rec[col]) if col in rec and rec[col] is not None else float("nan")
                for rec in records
            ],
            dtype=np.float64,
        )
        if np.isfinite(vals).any():
            pred_risk[col] = vals

    sample_dt = float(Config.SIM_FIXED_DELTA_SECONDS) * float(Config.RECORD_EVERY_N_TICKS)

    return RunData(
        prefix=prefix,
        run_path=run_path,
        records=records,
        distances=distances,
        risk_gt=np.array(risks, dtype=np.float64),
        speed_kmh=np.array(speeds, dtype=np.float64),
        target_speed_kmh=np.array(target_speeds, dtype=np.float64),
        positions_xy=positions_xy,
        pred_risk=pred_risk,
        sample_dt=sample_dt,
    )


def discover_trial_runs(trial_dir: Path) -> dict[str, RunData | None]:
    """Return `{prefix: RunData or None}` for every RUN_PREFIXES for one trial."""
    out: dict[str, RunData | None] = {}
    for prefix in RUN_PREFIXES:
        run_path = find_latest_run_by_prefix(trial_dir, prefix)
        out[prefix] = load_run_data(run_path, prefix=prefix) if run_path is not None else None
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_run_metrics(rd: RunData) -> dict:
    """Compute summary metrics for a single run."""
    if len(rd.records) == 0:
        return _empty_metrics()

    total_distance = float(rd.distances[-1]) if rd.distances.size else 0.0
    total_sim_time = float(len(rd.records) * rd.sample_dt)

    # risk_per_meter via per-segment trapezoidal integration of risk over distance.
    if rd.distances.size >= 2:
        seg = np.diff(rd.distances)
        midpoint_risk = 0.5 * (rd.risk_gt[:-1] + rd.risk_gt[1:])
        risk_distance_integral = float(np.sum(midpoint_risk * seg))
    else:
        risk_distance_integral = 0.0

    risk_per_meter = risk_distance_integral / total_distance if total_distance > 0 else 0.0
    avg_speed = float(np.mean(rd.speed_kmh)) if rd.speed_kmh.size else 0.0
    max_speed = float(np.max(rd.speed_kmh)) if rd.speed_kmh.size else 0.0
    speed_std = float(np.std(rd.speed_kmh)) if rd.speed_kmh.size else 0.0
    risk_auc = float(_np_trapezoid(rd.risk_gt, dx=rd.sample_dt)) if rd.risk_gt.size else 0.0

    return {
        "n_frames": len(rd.records),
        "total_distance_m": total_distance,
        "total_sim_time_s": total_sim_time,
        "avg_speed_kmh": avg_speed,
        "max_speed_kmh": max_speed,
        "speed_std_kmh": speed_std,
        "risk_auc": risk_auc,
        "risk_per_meter": risk_per_meter,
        "max_risk": float(np.max(rd.risk_gt)) if rd.risk_gt.size else 0.0,
        "mean_risk": float(np.mean(rd.risk_gt)) if rd.risk_gt.size else 0.0,
    }


def _empty_metrics() -> dict:
    return {
        "n_frames": 0,
        "total_distance_m": 0.0,
        "total_sim_time_s": 0.0,
        "avg_speed_kmh": 0.0,
        "max_speed_kmh": 0.0,
        "speed_std_kmh": 0.0,
        "risk_auc": 0.0,
        "risk_per_meter": 0.0,
        "max_risk": 0.0,
        "mean_risk": 0.0,
    }


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------

def _shared_xy_bounds(runs_by_prefix: dict[str, "RunData | None"]) -> tuple[float, float, float, float, float]:
    """Compute (x_min, y_min, x_max, y_max, pad) across every populated run."""
    pieces = [rd.positions_xy for rd in runs_by_prefix.values()
              if rd is not None and rd.positions_xy.shape[0] >= 2]
    if not pieces:
        return -1.0, -1.0, 1.0, 1.0, 0.1
    arr = np.concatenate(pieces, axis=0)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    pad = 0.05 * max(float(x_max - x_min), float(y_max - y_min), 1.0)
    return float(x_min), float(y_min), float(x_max), float(y_max), float(pad)


def _shared_scalar_range(
    runs_by_prefix: dict[str, "RunData | None"],
    attr: str,
) -> tuple[float, float]:
    """Return (vmin, vmax) across all runs for a given per-frame attribute."""
    finite_values: list[float] = []
    for rd in runs_by_prefix.values():
        if rd is None:
            continue
        values = getattr(rd, attr, None)
        if values is None or values.size == 0:
            continue
        f = values[np.isfinite(values)]
        if f.size:
            finite_values.append(float(f.min()))
            finite_values.append(float(f.max()))
    if not finite_values:
        return 0.0, 1.0
    vmin = min(finite_values)
    vmax = max(finite_values)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def render_route_grid(
    runs_by_prefix: dict[str, "RunData | None"],
    *,
    color_by: str = "risk_gt",
    title: str = "",
    output_path: Path | None = None,
    cmap_name: str = "viridis",
    figsize: tuple[float, float] | None = None,
):
    """Render a grid where each cell is one run's route colored by `color_by`.

    Grid dimensions follow `RUN_GRID_LAYOUT`. `color_by` must be the name of a
    per-frame array on RunData: 'risk_gt' or 'speed_kmh'.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n_rows = len(RUN_GRID_LAYOUT)
    n_cols = len(RUN_GRID_LAYOUT[0]) if RUN_GRID_LAYOUT else 0
    if figsize is None:
        figsize = (5.0 * n_cols, 5.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor("white")

    vmin, vmax = _shared_scalar_range(runs_by_prefix, color_by)
    x_min, y_min, x_max, y_max, pad = _shared_xy_bounds(runs_by_prefix)

    cbar_label = {"risk_gt": "ground-truth risk", "speed_kmh": "ego speed (km/h)"}.get(color_by, color_by)
    fig.suptitle(title or f"Route colored by {cbar_label}", fontsize=14)

    for r, row in enumerate(RUN_GRID_LAYOUT):
        for c, prefix in enumerate(row):
            ax = axes[r][c]
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(labelsize=7, colors="#666666")
            for spine in ax.spines.values():
                spine.set_color("#444444")

            rd = runs_by_prefix.get(prefix)
            if rd is None:
                ax.set_title(f"{prefix}\n(no run)", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            xy = rd.positions_xy
            values = getattr(rd, color_by, None)
            if xy.shape[0] < 2 or values is None or values.size == 0:
                ax.set_title(f"{prefix}\n(empty)", fontsize=10, color="gray")
                continue

            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            seg_values = 0.5 * (values[:-1] + values[1:])
            lc = LineCollection(segments, cmap=cmap_name, norm=plt.Normalize(vmin, vmax))
            lc.set_array(seg_values)
            lc.set_linewidth(2.5)
            ax.add_collection(lc)

            # Endpoint markers: green=start, red=end.
            ax.scatter(xy[0, 0], xy[0, 1], s=40, c="#00FF00", edgecolors="white", linewidths=0.5, zorder=3)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=40, c="#FF3030", edgecolors="white", linewidths=0.5, zorder=3)

            metrics = compute_run_metrics(rd)
            subtitle = (
                f"d={metrics['total_distance_m']:.0f}m  "
                f"t={metrics['total_sim_time_s']:.0f}s  "
                f"r/m={metrics['risk_per_meter']:.3f}"
            )
            ax.set_title(f"{prefix}\n{subtitle}", fontsize=10)

    import matplotlib.pyplot as plt
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label=cbar_label, pad=0.02)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Comparison video (animated grid, dimensions follow RUN_GRID_LAYOUT)
# ---------------------------------------------------------------------------

def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _encode_png_sequence_to_mp4(frames_dir: Path, output_path: Path, fps: int) -> Path:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or add it to PATH.")
    if output_path.exists():
        output_path.unlink()
    cmd = [
        ffmpeg,
        "-y",
        "-framerate", str(int(fps)),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def _interp_at_progress(values: np.ndarray, n_target: int) -> np.ndarray:
    """Resample a 1D array to `n_target` samples by linear interpolation on progress 0..1."""
    if values.size == 0:
        return np.zeros(n_target)
    if values.size == 1:
        return np.full(n_target, values[0])
    src_x = np.linspace(0.0, 1.0, num=values.size)
    dst_x = np.linspace(0.0, 1.0, num=n_target)
    return np.interp(dst_x, src_x, values)


def _interp_xy_at_progress(xy: np.ndarray, n_target: int) -> np.ndarray:
    """Resample (N,2) xy to (n_target, 2) by linear interpolation on progress 0..1."""
    if xy.shape[0] == 0:
        return np.zeros((n_target, 2))
    if xy.shape[0] == 1:
        return np.repeat(xy, n_target, axis=0)
    return np.stack([_interp_at_progress(xy[:, 0], n_target),
                     _interp_at_progress(xy[:, 1], n_target)], axis=1)


def render_comparison_video(
    runs_by_prefix: dict[str, "RunData | None"],
    output_path: Path,
    *,
    fps: int = 8,
    video_seconds: float | None = None,
    title: str = "",
    color_by: str = "risk_gt",
    cmap_name: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> Path | None:
    """Build an animated MP4 comparing every run of a trial.

    Grid dimensions follow `RUN_GRID_LAYOUT`. Each panel shows the route (colored
    by `color_by`) with an animated ego dot. All panels are synced by
    route-progress fraction so each run "finishes" at the same time.

    Returns the output path, or None if no run has data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n_rows = len(RUN_GRID_LAYOUT)
    n_cols = len(RUN_GRID_LAYOUT[0]) if RUN_GRID_LAYOUT else 0
    if figsize is None:
        figsize = (5.0 * n_cols, 5.0 * n_rows)

    populated = [rd for rd in runs_by_prefix.values() if rd is not None and rd.positions_xy.shape[0] >= 2]
    if not populated:
        return None

    if video_seconds is None:
        # Default: scale by longest run's wall-clock duration but clamp to a sane window.
        longest = max(len(rd.records) * rd.sample_dt for rd in populated)
        video_seconds = float(max(8.0, min(45.0, longest)))
    n_frames = max(2, int(round(fps * video_seconds)))

    vmin, vmax = _shared_scalar_range(runs_by_prefix, color_by)
    x_min, y_min, x_max, y_max, pad = _shared_xy_bounds(runs_by_prefix)

    # Pre-resample each run's series to n_frames so the animation loop is O(panels).
    resampled: dict[str, dict] = {}
    for prefix in RUN_PREFIXES:
        rd = runs_by_prefix.get(prefix)
        if rd is None or rd.positions_xy.shape[0] < 2:
            resampled[prefix] = {}
            continue
        resampled[prefix] = {
            "xy_anim": _interp_xy_at_progress(rd.positions_xy, n_frames),
            "speed_anim": _interp_at_progress(rd.speed_kmh, n_frames),
            "risk_anim": _interp_at_progress(rd.risk_gt, n_frames),
            "metrics": compute_run_metrics(rd),
        }

    frames_dir = Path(tempfile.mkdtemp(prefix="trial_cmp_frames_", dir=str(output_path.parent)))
    try:
        for i in range(n_frames):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
            fig.patch.set_facecolor("white")
            fig.suptitle(
                f"{title}    frame {i+1}/{n_frames}    progress={100.0 * i / max(1, n_frames - 1):.0f}%",
                fontsize=12,
            )
            for r, row in enumerate(RUN_GRID_LAYOUT):
                for c, prefix in enumerate(row):
                    ax = axes[r][c]
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlim(x_min - pad, x_max + pad)
                    ax.set_ylim(y_min - pad, y_max + pad)
                    ax.set_facecolor("#1a1a1a")
                    ax.tick_params(labelsize=6, colors="#666666")
                    for spine in ax.spines.values():
                        spine.set_color("#333333")

                    rd = runs_by_prefix.get(prefix)
                    if rd is None:
                        ax.set_title(f"{prefix}\n(no run)", fontsize=9, color="gray")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    xy = rd.positions_xy
                    values = getattr(rd, color_by, None)
                    if xy.shape[0] < 2 or values is None or values.size == 0:
                        ax.set_title(f"{prefix}\n(empty)", fontsize=9, color="gray")
                        continue

                    segments = np.stack([xy[:-1], xy[1:]], axis=1)
                    seg_values = 0.5 * (values[:-1] + values[1:])
                    lc = LineCollection(segments, cmap=cmap_name, norm=plt.Normalize(vmin, vmax))
                    lc.set_array(seg_values)
                    lc.set_linewidth(2.0)
                    ax.add_collection(lc)

                    info = resampled[prefix]
                    xy_now = info["xy_anim"][i]
                    speed_now = info["speed_anim"][i]
                    risk_now = info["risk_anim"][i]
                    metrics = info["metrics"]
                    ax.scatter(xy[0, 0], xy[0, 1], s=20, c="#00FF00",
                               edgecolors="white", linewidths=0.4, zorder=3)
                    ax.scatter(xy[-1, 0], xy[-1, 1], s=20, c="#FF3030",
                               edgecolors="white", linewidths=0.4, zorder=3)
                    ax.scatter(xy_now[0], xy_now[1], s=70, c="#FFFFFF",
                               edgecolors="black", linewidths=1.0, zorder=4)

                    subtitle = (
                        f"d={metrics['total_distance_m']:.0f}m  "
                        f"t={metrics['total_sim_time_s']:.0f}s  "
                        f"r/m={metrics['risk_per_meter']:.3f}\n"
                        f"now: v={speed_now:5.1f} km/h   risk={risk_now:.2f}"
                    )
                    ax.set_title(f"{prefix}\n{subtitle}", fontsize=8)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            frame_path = frames_dir / f"frame_{i:06d}.png"
            fig.savefig(frame_path, dpi=100, facecolor="white")
            plt.close(fig)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return _encode_png_sequence_to_mp4(frames_dir, output_path, fps=fps)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cross-trial aggregation helpers
# ---------------------------------------------------------------------------

def compute_trial_metrics_table(
    runs_by_trial: dict[str, dict[str, "RunData | None"]],
) -> list[dict]:
    """Flatten the nested {trial: {prefix: run}} into a list of metric rows."""
    rows: list[dict] = []
    for trial_name, prefix_map in runs_by_trial.items():
        for prefix in RUN_PREFIXES:
            rd = prefix_map.get(prefix)
            base_row = {"trial": trial_name, "prefix": prefix, "available": rd is not None}
            if rd is None:
                base_row.update(_empty_metrics())
            else:
                base_row.update(compute_run_metrics(rd))
            rows.append(base_row)
    return rows


def aggregate_by_prefix(rows: list[dict]) -> list[dict]:
    """Aggregate the table from compute_trial_metrics_table by run prefix.

    Returns mean / median across available runs of that prefix for each metric.
    """
    by_prefix: dict[str, list[dict]] = {p: [] for p in RUN_PREFIXES}
    for row in rows:
        if row.get("available"):
            by_prefix[row["prefix"]].append(row)

    out: list[dict] = []
    metric_keys = (
        "total_distance_m",
        "total_sim_time_s",
        "avg_speed_kmh",
        "max_speed_kmh",
        "risk_per_meter",
        "risk_auc",
        "mean_risk",
        "max_risk",
    )
    for prefix in RUN_PREFIXES:
        bucket = by_prefix[prefix]
        n = len(bucket)
        row = {"prefix": prefix, "n_runs": n}
        if n == 0:
            for k in metric_keys:
                row[f"{k}_mean"] = float("nan")
                row[f"{k}_median"] = float("nan")
        else:
            for k in metric_keys:
                vals = np.array([float(r[k]) for r in bucket], dtype=np.float64)
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_median"] = float(np.median(vals))
        out.append(row)
    return out


def render_efficiency_barplot(
    runs_by_prefix: dict[str, "RunData | None"],
    *,
    title: str = "",
    output_path: Path | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> "plt.Figure":  # type: ignore[name-defined]
    """Render a 3-panel bar chart comparing every prefix to the 'base' run.

    Panels (left → right):
      1. Δ risk/m  = prefix.risk_per_meter − base.risk_per_meter  (negative = safer)
      2. Δ time (s) = prefix.total_sim_time_s − base.total_sim_time_s  (positive = slower)
      3. Efficiency = Δ risk/m ÷ Δ time (s)  — risk reduction per additional second;
         only shown for prefixes where |Δ time| > 0.1 s; otherwise set to NaN.

    The 'base' bar is always zero in panels 1 & 2 and NaN in panel 3.  Bars
    are coloured green for improvement (safer / more efficient) and red for
    worsening.  Missing runs are skipped with a cross-hatched empty bar.
    """
    import matplotlib.pyplot as plt

    base_rd = runs_by_prefix.get("base")
    base_metrics = compute_run_metrics(base_rd) if base_rd is not None else None

    prefixes = [p for p in RUN_PREFIXES if p != "base"]
    labels = ["base"] + prefixes
    all_prefixes = labels  # order for display

    def _metrics_for(p: str) -> dict | None:
        rd = runs_by_prefix.get(p)
        return compute_run_metrics(rd) if rd is not None else None

    metrics_map = {p: _metrics_for(p) for p in all_prefixes}

    delta_rpm: list[float] = []
    delta_time: list[float] = []
    efficiency: list[float] = []
    available: list[bool] = []

    base_rpm  = base_metrics["risk_per_meter"]  if base_metrics else float("nan")
    base_time = base_metrics["total_sim_time_s"] if base_metrics else float("nan")

    for p in all_prefixes:
        m = metrics_map[p]
        if m is None or base_metrics is None:
            delta_rpm.append(float("nan"))
            delta_time.append(float("nan"))
            efficiency.append(float("nan"))
            available.append(False)
            continue
        available.append(True)
        d_rpm  = m["risk_per_meter"]   - base_rpm
        d_time = m["total_sim_time_s"] - base_time
        delta_rpm.append(d_rpm)
        delta_time.append(d_time)
        eff = (d_rpm / d_time) if abs(d_time) > 0.1 else float("nan")
        efficiency.append(eff)

    x = np.arange(len(all_prefixes))
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.suptitle(title or "Efficiency vs. base run", fontsize=13)

    panel_data = [
        (axes[0], delta_rpm,  "Δ risk/m (vs. base)", "Δ risk/metre"),
        (axes[1], delta_time, "Δ time / s (vs. base)", "Δ seconds"),
        (axes[2], efficiency, "Δ risk/m  ÷  Δ time (s)", "risk/m per extra second"),
    ]

    for panel_idx, (ax, values, panel_title, ylabel) in enumerate(panel_data):
        ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
        for i, (v, lbl, avail) in enumerate(zip(values, all_prefixes, available)):
            if not avail or not np.isfinite(v):
                ax.bar(i, 0, color="none", edgecolor="#888888",
                       linewidth=1.2, hatch="//", label=lbl if i == 0 else "")
                continue
            # Panel 1 (Δ time): positive = more time = bad → red for positive.
            # Panels 0 & 2 (Δ risk/m, efficiency): negative = better → green for negative.
            if panel_idx == 1:
                color = "#d9534f" if v > 0 else "#5cb85c"
            else:
                color = "#5cb85c" if v <= 0 else "#d9534f"
            ax.bar(i, v, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(all_prefixes, rotation=30, ha="right", fontsize=9)
        ax.set_title(panel_title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, facecolor="white")
    return fig


__all__ = [
    "RUN_PREFIXES",
    "RUN_GRID_LAYOUT",
    "TEST_VAL_MAP_NAMES",
    "RunData",
    "aggregate_by_prefix",
    "compute_run_metrics",
    "compute_trial_metrics_table",
    "discover_trial_runs",
    "find_latest_run_by_prefix",
    "is_test_or_val_trial",
    "load_run_data",
    "render_comparison_video",
    "render_efficiency_barplot",
    "render_route_grid",
]
