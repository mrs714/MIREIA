import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont

from matplotlib.widgets import Slider, RadioButtons

from MIREIA.core.physics import RiskOracle
from MIREIA.data_collection.dataset_utils import load_jsonl_records
from MIREIA.analysis.plotter import Grid, RiskGrid
from MIREIA.simulation.bridge import EgoKinematics, DynamicObstacleKinematics, EnvironmentState, SimulationBridge

from carla import World

class RiskGridVisualizer:
    """
    Takes a RiskGrid and a SimulationBridge, and renders a top-down heatmap
    image with the ego vehicle and dynamic obstacles drawn as rotated boxes.
    Supports single-frame rendering and multi-frame video recording.
    """

    def __init__(self, risk_grid: RiskGrid, world: World, bridge: SimulationBridge, oracle: RiskOracle = None, vmax: float = None):
        self.risk_grid = risk_grid
        self.world = world
        self.bridge = bridge
        self.oracle = oracle or RiskOracle()
        self.vmax = vmax  # Fixed color scale max. None = auto-scale per frame.

    def _render_frame(self, ax, fig, risk_grid: RiskGrid = None):
        """
        Internal method that draws a single frame onto the given axes.
        If risk_grid is None, uses self.risk_grid.
        """
        ax.clear()

        rg = risk_grid or self.risk_grid
        ego = self.bridge.get_ego_kinematics()
        dynamic_obstacles = self.bridge.get_dynamic_obstacles()
        static_obstacles = self.bridge.get_static_obstacles()
        pedestrians = self.bridge.get_pedestrians()
        env = self.bridge.get_environment_state()

        # --- Reconstruct 2D risk array from the flat grid list ---
        n = int(rg.size / rg.resolution) + 1
        risk_array = np.array([p['risk_value'] for p in rg.grid]).reshape(n, n)
        half = rg.size / 2.0
        extent = [rg.center_x - half, rg.center_x + half,
                  rg.center_y - half, rg.center_y + half]

        # Heatmap
        vmin = 0.0 if self.vmax is not None else rg.lowest_risk
        vmax = self.vmax if self.vmax is not None else rg.highest_risk
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        ax.imshow(risk_array, origin='lower', extent=extent,
                  cmap='jet', norm=norm, aspect='equal', interpolation='bilinear')

        # Ego vehicle (green box with heading arrow)
        self._draw_vehicle_box(ax, ego.x, ego.y, ego.length, ego.width, ego.heading,
                               edgecolor='#00FF00', facecolor='black', linewidth=2, label='Ego')
        # Ego safety radius (red circle outline)
        ego_radius_m = 10
        ego_circle = patches.Circle((ego.x, ego.y), radius=ego_radius_m,
                        fill=False, edgecolor='red', linewidth=2, alpha=0.9)
        ax.add_patch(ego_circle)
        ego_heading_rad = math.radians(ego.heading)
        arrow_len = max(ego.v * 0.5, ego.length * 0.6)
        ax.arrow(ego.x, ego.y,
                 arrow_len * math.cos(ego_heading_rad),
                 arrow_len * math.sin(ego_heading_rad),
                 head_width=0.6, head_length=0.4, fc='#00FF00', ec='#00FF00', linewidth=1.5)

        # Dynamic obstacles (red/gray boxes with velocity arrows)
        for obj in dynamic_obstacles:
            self._draw_vehicle_box(ax, obj.x, obj.y, obj.length, obj.width, obj.heading,
                                   edgecolor='red', facecolor='#555555', linewidth=1, alpha=0.9)
            if abs(obj.vx) > 0.1 or abs(obj.vy) > 0.1:
                ax.arrow(obj.x, obj.y, obj.vx * 0.5, obj.vy * 0.5,
                         head_width=0.4, head_length=0.3, fc='white', ec='white')
                
        # Static obstacles (black boxes)
        for obs in static_obstacles:
            self._draw_static_obstacle(ax, obs.x, obs.y, obs.length, obs.width, obs.type, heading_deg=obs.heading)

        # Pedestrians (purple boxes)
        for ped in pedestrians:
            self._draw_vehicle_box(ax, ped.x, ped.y, ped.length, ped.width, ped.heading,
                                   edgecolor='#AA00FF', facecolor='#7700AA', linewidth=1, alpha=0.9)
            if abs(ped.vx) > 0.01 or abs(ped.vy) > 0.01:
                ax.arrow(ped.x, ped.y, ped.vx * 5, ped.vy * 5,
                         head_width=0.3, head_length=0.2, fc='#AA00FF', ec='#AA00FF')

        # Title
        title = (f"Risk Heatmap  |  Ego speed: {ego.v:.1f} m/s  |  "
                 f"mu: {env.mu:.2f}  |  vis: {env.visibility:.0f}m  |  "
                 f"obstacles: {len(dynamic_obstacles)}")
        ax.set_title(title, fontsize=11, color='white', pad=10)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=0.15)

    def render(self, save_path: str = None, dpi: int = 150):
        """
        Generates and optionally saves a single-frame top-down risk heatmap image.

        :param save_path: If provided, saves the figure to this path (e.g. 'output/risk.png').
        :param dpi: Resolution of the saved image.
        :return: The matplotlib Figure object.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 10))

        self._render_frame(ax, fig)

        # Add colorbar once for single frame
        rg = self.risk_grid
        n = int(rg.size / rg.resolution) + 1
        risk_array = np.array([p['risk_value'] for p in rg.grid]).reshape(n, n)
        vmin = 0.0 if self.vmax is not None else rg.lowest_risk
        vmax = self.vmax if self.vmax is not None else rg.highest_risk
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)

        fig.tight_layout()
        _add_colorbar(fig, ax, sm)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

        return fig

    def render_video(self, save_path: str, n_frames: int = 100,
                     grid_size: float = 100.0, grid_resolution: float = 1.0,
                     fps: int = None, dpi: int = 100,
                     baked_static_risk: RiskGrid = None):
        """
        Records a video of n_frames. Each frame updates the simulation bridge,
        recomputes the risk grid centered on the ego, and renders.

        :param save_path: Output video path (e.g. 'output/risk_video.mp4').
        :param n_frames: Number of frames to record.
        :param grid_size: Size of the risk grid in meters.
        :param grid_resolution: Resolution of the risk grid in meters.
        :param fps: Frames per second of the output video. If None (default),
            derived from the world's fixed_delta_seconds so the video plays
            back in real time.
        :param dpi: Resolution of each frame.
        :param baked_static_risk: Optional pre-baked static risk grid. If provided,
            road and static obstacle risk is sampled via bilinear interpolation
            instead of being recomputed every frame (much faster).
        """
        # Derive FPS from simulation tick step for real-time playback
        if fps is None:
            delta = self.world.get_settings().fixed_delta_seconds
            fps = int(round(1.0 / delta)) if delta and delta > 0 else 10
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 10))

        # Add a persistent colorbar
        if self.vmax is not None:
            init_norm = mcolors.Normalize(vmin=0.0, vmax=self.vmax)
        else:
            init_norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='jet', norm=init_norm)
        fig.tight_layout()
        cbar = _add_colorbar(fig, ax, sm)

        def update(frame_idx):
            # 1. Advance simulation state
            self.world.tick()  # Advance the simulation by one tick (e.g. 0.05s)
            self.bridge.update()

            # 2. Recompute grid centered on ego
            ego = self.bridge.get_ego_kinematics()
            grid = Grid(center_x=ego.x, center_y=ego.y,
                        size=grid_size, resolution=grid_resolution)
            risk_grid = self.oracle.calculate_risk_map(grid, self.bridge,
                                                       baked_static_risk=baked_static_risk)

            # 3. Render frame
            self._render_frame(ax, fig, risk_grid)

            # 4. Update colorbar norm (only if auto-scaling)
            if self.vmax is None:
                sm.set_norm(mcolors.Normalize(vmin=risk_grid.lowest_risk, vmax=risk_grid.highest_risk))

            print(f"\rFrame {frame_idx + 1}/{n_frames}", end="", flush=True)

        anim = FuncAnimation(fig, update, frames=n_frames, repeat=False)
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"\nVideo saved to {save_path}")

    @staticmethod
    def _draw_vehicle_box(ax, x, y, length, width, heading_deg,
                          edgecolor='white', facecolor='none', linewidth=1, alpha=1.0, label=None):
        """Draws a rotated rectangle representing a vehicle."""
        rect = patches.Rectangle((-length / 2, -width / 2), length, width,
                                 linewidth=linewidth, edgecolor=edgecolor,
                                 facecolor=facecolor, alpha=alpha, label=label)
        t = Affine2D().rotate_deg(heading_deg).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    def _draw_static_obstacle(self, ax, x, y, length, width, type, heading_deg=0):
        """Draws a static obstacle as a colored box."""
        if type == "Crosswalk":
            color = "#073800"
        elif type == "TrafficLight":
            color = '#FFAA00'
        elif type == "ParkedVehicle":
            color = "#500000"
        else:
            color = "#333333"
        rect = patches.Rectangle((-length / 2, -width / 2), length, width,
                                 linewidth=1, edgecolor='black', facecolor=color, alpha=0.9)
        t = Affine2D().rotate_deg(heading_deg).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)


def _add_colorbar(fig, ax, sm, height_ratio: float = 0.75, width: float = 0.02, pad: float = 0.02):
    """Add a shorter colorbar that doesn't overlap the title area."""
    cbar = fig.colorbar(sm, ax=ax, label='Risk')
    pos = ax.get_position()
    cbar_height = pos.height * height_ratio
    cbar_x = pos.x1 + pad
    cbar_y = pos.y0 + (pos.height - cbar_height) / 2.0
    cbar.ax.set_position([cbar_x, cbar_y, width, cbar_height])
    return cbar


def render_static_risk_map(risk_grid: RiskGrid, save_path: str,
                           resolution: tuple[int, int] = (1024, 1024),
                           dpi: int = 150,
                           vmax: float | None = None) -> str:
    """
    Render a static, full-map risk heatmap image to disk.

    :param risk_grid: Pre-baked RiskGrid covering the entire map.
    :param save_path: Output image path.
    :param resolution: (width, height) of the output image in pixels.
    :param dpi: Dots-per-inch for the saved image.
    :param vmax: Optional fixed max value for the color scale.
    :returns: The save_path for convenience.
    """
    plt.style.use('dark_background')
    width_px, height_px = resolution
    fig_w = max(width_px / dpi, 1.0)
    fig_h = max(height_px / dpi, 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n = int(risk_grid.size / risk_grid.resolution) + 1
    risk_array = np.array([p['risk_value'] for p in risk_grid.grid]).reshape(n, n)
    half = risk_grid.size / 2.0
    extent = [risk_grid.center_x - half, risk_grid.center_x + half,
              risk_grid.center_y - half, risk_grid.center_y + half]

    vmin = 0.0 if vmax is not None else risk_grid.lowest_risk
    vmax = vmax if vmax is not None else risk_grid.highest_risk
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(risk_array, origin='lower', extent=extent,
              cmap='jet', norm=norm, aspect='equal', interpolation='bilinear')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, alpha=0.15)

    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    fig.tight_layout()
    _add_colorbar(fig, ax, sm)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


def render_risk_map_with_actors(risk_grid: RiskGrid, world: World, bridge: SimulationBridge,
                                save_path: str,
                                resolution: tuple[int, int] = (1024, 1024),
                                dpi: int = 150,
                                vmax: float | None = None) -> str:
    """
    Render a full-map risk heatmap including ego and actors and save to disk.

    :param risk_grid: Pre-baked RiskGrid covering the entire map.
    :param world: CARLA world (used only for metadata in the visualizer).
    :param bridge: SimulationBridge providing ego and actor kinematics.
    :param save_path: Output image path.
    :param resolution: (width, height) of the output image in pixels.
    :param dpi: Dots-per-inch for the saved image.
    :param vmax: Optional fixed max value for the color scale.
    :returns: The save_path for convenience.
    """
    plt.style.use('dark_background')
    width_px, height_px = resolution
    fig_w = max(width_px / dpi, 1.0)
    fig_h = max(height_px / dpi, 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    visualizer = RiskGridVisualizer(risk_grid=risk_grid, world=world, bridge=bridge, vmax=vmax)
    visualizer._render_frame(ax, fig, risk_grid=risk_grid)

    n = int(risk_grid.size / risk_grid.resolution) + 1
    risk_array = np.array([p['risk_value'] for p in risk_grid.grid]).reshape(n, n)
    vmin = 0.0 if vmax is not None else risk_grid.lowest_risk
    vmax = vmax if vmax is not None else risk_grid.highest_risk
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    fig.tight_layout()
    _add_colorbar(fig, ax, sm)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


class DatasetVideoComposer:
    """
    Compose a 2x2 tiled video from dataset JSONL frames.

    Layout (clockwise):
    - top-left: dashcam
    - top-right: empty
    - bottom-right: aerial/top-down
    - bottom-left: risk map
    """

    def __init__(self, jsonl_path: str, fps: int = 10, background=(0, 0, 0)):
        self.jsonl_path = Path(jsonl_path)
        self.dataset_dir = self.jsonl_path.parent
        self.fps = fps
        self.background = background

    def build_video(self, output_name: str = "dataset_video.mp4") -> str:
        """
        Read the JSONL, compose frames, and write a video next to the JSONL.

        :param output_name: Output video filename.
        :returns: Path to the created video.
        """
        frames_dir = self.dataset_dir / "_video_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        records = self._load_records()
        if not records:
            raise RuntimeError("No frames found in JSONL.")

        tile_size = self._infer_tile_size(records)
        if tile_size is None:
            raise RuntimeError("No image paths found in JSONL.")

        for idx, record in enumerate(records):
            frame = self._compose_frame(record, tile_size)
            frame_path = frames_dir / f"frame_{idx:06d}.png"
            frame.save(frame_path)

        output_path = self.dataset_dir / output_name
        self._encode_video(frames_dir, output_path)
        shutil.rmtree(frames_dir, ignore_errors=True)
        return str(output_path)

    def _load_records(self) -> list[dict]:
        return load_jsonl_records(str(self.jsonl_path))

    def _infer_tile_size(self, records: list[dict]) -> tuple[int, int] | None:
        for record in records:
            for key in ("rgb_image_path", "risk_map_image_path", "topdown_image_path"):
                rel = record.get(key) or ""
                if rel:
                    img_path = self.dataset_dir / rel
                    if img_path.exists():
                        with Image.open(img_path) as img:
                            return img.size
        return None

    def _load_image(self, rel_path: str) -> Image.Image | None:
        if not rel_path:
            return None
        img_path = self.dataset_dir / rel_path
        if not img_path.exists():
            return None
        return Image.open(img_path)

    def _compose_frame(self, record: dict, tile_size: tuple[int, int]) -> Image.Image:
        tile_w, tile_h = tile_size
        canvas = Image.new("RGB", (tile_w * 2, tile_h * 2), self.background)

        dashcam = self._load_image(record.get("rgb_image_path", ""))
        risk = self._load_image(record.get("risk_map_image_path", ""))
        aerial = self._load_image(record.get("topdown_image_path", ""))
        risk_value = record.get("ground_truth_risk", None)

        if dashcam is not None:
            canvas.paste(dashcam.resize((tile_w, tile_h)), (0, 0))
            dashcam.close()
        if risk is not None:
            canvas.paste(risk.resize((tile_w, tile_h)), (0, tile_h))
            risk.close()
        if aerial is not None:
            canvas.paste(aerial.resize((tile_w, tile_h)), (tile_w, tile_h))
            aerial.close()

        if risk_value is not None:
            risk_tile = self._render_risk_tile(risk_value, tile_w, tile_h)
            if risk_tile is not None:
                canvas.paste(risk_tile, (tile_w, 0))

        return canvas

    def _render_risk_tile(self, risk_value: float, width: int, height: int) -> Image.Image | None:
        try:
            risk = float(risk_value)
        except (TypeError, ValueError):
            return None

        risk_display = max(0.0, min(10.0, risk))
        risk_norm = risk_display / 10.0
        r = int(255 * risk_norm)
        g = int(255 * (1.0 - risk_norm))
        b = 0
        tile = Image.new("RGB", (width, height), (r, g, b))

        draw = ImageDraw.Draw(tile)
        font_size = max(10, int(min(width, height) * 0.13))
        small_font_size = max(9, int(font_size * 0.5))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            small_font = ImageFont.truetype("arial.ttf", small_font_size)
        except OSError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        text = f"Ground truth:\n{risk_display:.2f}"
        text_w, text_h = self._measure_multiline(draw, text, font, spacing=4)
        draw.multiline_text(
            ((width - text_w) / 2, (height - text_h) / 2),
            text,
            fill=(0, 0, 0),
            font=font,
            align="center",
            spacing=4,
        )

        # Draw a simple vertical scale bar on the right
        bar_w = max(8, width // 30)
        bar_height = int(height * 0.65)
        bar_x0 = width - bar_w - 4
        bar_y0 = int((height - bar_height) / 2)
        bar_y1 = bar_y0 + bar_height
        for y in range(bar_y0, bar_y1):
            t = 1.0 - (y - bar_y0) / max(1, (bar_y1 - bar_y0))
            rr = int(255 * t)
            gg = int(255 * (1.0 - t))
            draw.line([(bar_x0, y), (bar_x0 + bar_w, y)], fill=(rr, gg, 0))

        top_label = "10"
        bot_label = "0"
        top_w, top_h = self._measure_multiline(draw, top_label, small_font)
        bot_w, bot_h = self._measure_multiline(draw, bot_label, small_font)
        draw.text((bar_x0 - top_w - 3, bar_y0 - top_h / 2), top_label, fill=(0, 0, 0), font=small_font)
        draw.text((bar_x0 - bot_w - 3, bar_y1 - bot_h / 2), bot_label, fill=(0, 0, 0), font=small_font)

        return tile

    def _measure_multiline(self, draw: ImageDraw.ImageDraw, text: str,
                           font: ImageFont.ImageFont, spacing: int = 2) -> tuple[int, int]:
        lines = text.split("\n")
        widths = []
        heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        text_w = max(widths) if widths else 0
        text_h = sum(heights) + spacing * max(0, len(lines) - 1)
        return text_w, text_h

    def _encode_video(self, frames_dir: Path, output_path: Path):
        if output_path.exists():
            output_path.unlink()

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or add it to PATH.")

        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate", str(self.fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


class DummyRiskVisualizer:
    def __init__(self):
        self.oracle = RiskOracle()
        # Initialize as correct types
        class DummyEgo(EgoKinematics):
            def __init__(self, x, y, v, vx, vy, yaw):
                self.x = x
                self.y = y
                self.v = v
                self.vx = vx
                self.vy = vy
                self.heading = yaw
        class DummyObstacle(DynamicObstacleKinematics):
            def __init__(self, x, y, vx, vy, yaw, l, w):
                self.x = x
                self.y = y
                self.vx = vx
                self.vy = vy
                self.heading = yaw
                self.length = l
                self.width = w
        class DummyEnv(EnvironmentState):
            def __init__(self, mu, visibility):
                self.mu = mu
                self.visibility = visibility

        self.DummyEgo = DummyEgo
        self.DummyObstacle = DummyObstacle
        self.DummyEnv = DummyEnv

        self.ego = DummyEgo(0, 0, 20, 20, 0, 0)
        self.env = DummyEnv(0.8, 300.0)
        self.obstacles = []
        self.scenario_name = 'Following'
        self.load_scenario('Following')

        # Setup Plot Style
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('Thesis Risk Field Visualizer')

        # --- Layout Definitions ---
        self.ax_plot = self.fig.add_axes([0.25, 0.30, 0.60, 0.65]) # Main Plot
        self.ax_cbar = self.fig.add_axes([0.87, 0.30, 0.02, 0.65]) # Color Bar (Thin strip on right)
        self.norm = mcolors.Normalize(vmin=0, vmax=1.5)
        self.setup_widgets()
        self.render()
        plt.show()

    def load_scenario(self, name):
        self.scenario_name = name
        O = self.DummyObstacle
        if name == 'Following':
            self.obstacles = [O(40, 0, 10, 0, 0, 4.5, 1.8)]
        elif name == 'Cut-In':
            self.obstacles = [O(25, 3.5, 18, -2, -0.2, 4.5, 1.8)]
        elif name == 'Oncoming':
            self.obstacles = [O(60, 3.5, -20, 0, 3.14, 4.5, 1.8)]
        elif name == 'Traffic Jam':
            self.obstacles = [
                O(20, 0, 0, 0, 0, 4.5, 1.8),
                O(32, 0, 0, 0, 0, 4.5, 1.8),
                O(44, 0, 0, 0, 0, 4.5, 1.8),
            ]

    def render(self, val=None):
        self.ax_plot.clear()
        self.ax_cbar.clear()

        # Update Title with Stats
        title_str = (f"SCENARIO: {self.scenario_name}  |  "
                     f"Friction (mu): {self.env.mu:.2f}  |  "
                     f"Visibility: {self.env.visibility:.0f}m  |  "
                     f"Ego Speed: {self.ego.v:.0f} m/s")
        self.ax_plot.set_title(title_str, fontsize=12, color='white', pad=10)

        # 1. Generate Grid
        x = np.linspace(-10, 80, 150) # Resolution: Long
        y = np.linspace(-10, 10, 80)  # Resolution: Lat
        X, Y = np.meshgrid(x, y)

        # 2. Compute Risk Field
        Z = self.oracle.calculate_risk_map(X, Y, self.ego, self.obstacles, self.env)

        # 3. Draw Heatmap (Contourf)
        levels = np.linspace(0, 1.5, 50)
        cf = self.ax_plot.contourf(X, Y, Z, levels=levels, cmap='jet', norm=self.norm, extend='max')

        # 4. Draw Road Markings
        self.ax_plot.plot([-10, 80], [1.75, 1.75], 'w--', linewidth=2, alpha=0.5) # Left Ln
        self.ax_plot.plot([-10, 80], [-1.75, -1.75], 'w--', linewidth=2, alpha=0.5) # Right Ln

        # 5. Draw Ego Vehicle (Green Box with black fill)
        ego_rect = patches.Rectangle((self.ego.x-2.2, self.ego.y-0.9), 4.5, 1.8, 
                                     linewidth=2, edgecolor='#00FF00', facecolor='#000000', label='Ego')
        self.ax_plot.add_patch(ego_rect) 

        # 6. Draw Obstacles (Gray Boxes with Velocity Arrows)
        for obj in self.obstacles:
            obs_rect = patches.Rectangle((obj.x-obj.length/2, obj.y-obj.width/2), obj.length, obj.width, 
                                         linewidth=1, edgecolor='red', facecolor='#555555', alpha=0.9)
            self.ax_plot.add_patch(obs_rect)
            # Velocity Arrow
            self.ax_plot.arrow(obj.x, obj.y, obj.vx*0.5, obj.vy*0.5, 
                               head_width=0.5, head_length=0.8, fc='white', ec='white')

        # 7. Formatting
        self.ax_plot.set_xlim(-10, 80)
        self.ax_plot.set_ylim(-8, 8)
        self.ax_plot.set_aspect('equal')
        self.ax_plot.set_xlabel("Longitudinal Distance (m)")
        self.ax_plot.grid(True, alpha=0.2)
        self.fig.canvas.draw_idle()

    def setup_widgets(self):
        # --- A. Scenario Selector (Top Left Panel) ---
        # Position: [left, bottom, width, height]
        ax_radio = self.fig.add_axes([0.02, 0.65, 0.15, 0.20], facecolor='#222222')
        ax_radio.text(0.5, 1.05, 'SCENARIOS', ha='center', transform=ax_radio.transAxes, 
                      weight='bold', color='white')
        
        self.radio = RadioButtons(ax_radio, ('Following', 'Cut-In', 'Oncoming', 'Traffic Jam'),
                                  activecolor='#00FF00')
        
        # Fix text colors for radio buttons
        for label in self.radio.labels:
            label.set_color('white')

        def change_scenario(label):
            self.load_scenario(label)
            self.render()
        self.radio.on_clicked(change_scenario)
        
        # --- B. Parameter Sliders (Bottom Panel) ---
        # 1. Friction Slider
        ax_mu = self.fig.add_axes([0.25, 0.15, 0.60, 0.03])
        self.s_mu = Slider(ax_mu, 'Friction (Dry->Wet)', 0.1, 1.0, valinit=0.8, 
                           color='#00AAFF')
        
        # 2. Visibility Slider
        ax_vis = self.fig.add_axes([0.25, 0.10, 0.60, 0.03])
        self.s_vis = Slider(ax_vis, 'Visibility (m)', 10, 500, valinit=300, 
                            color='#00FF00')
        
        # 3. Speed Slider
        ax_speed = self.fig.add_axes([0.25, 0.05, 0.60, 0.03])
        self.s_speed = Slider(ax_speed, 'Ego Speed (m/s)', 0, 45, valinit=20, 
                              color='#FF5500')

        # Slider Styling
        for s in [self.s_mu, self.s_vis, self.s_speed]:
            s.label.set_color('white')
            s.valtext.set_color('white')

        def update(val):
            self.env.mu = self.s_mu.val
            self.env.visibility = self.s_vis.val
            self.ego.v = self.s_speed.val
            self.ego.vx = self.s_speed.val
            self.render()
        self.s_mu.on_changed(update)
        self.s_vis.on_changed(update)
        self.s_speed.on_changed(update)

if __name__ == "__main__":
    
    print("Launching Risk Field Visualizer...")
    viz = DummyRiskVisualizer()