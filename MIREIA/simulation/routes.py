# Interactive waypoint snapping helpers for route start/end selection.
import argparse
import sys
from pathlib import Path

import carla
import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly: .\MIREIA\simulation\routes.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MIREIA.config import Config
from MIREIA.simulation.bridge import SimulationBridge, WaypointState, WaypointStateCollection


def select_closest_waypoints_interactive(
    waypoint_collection: WaypointStateCollection,
    max_clicks: int = 2,
) -> list[WaypointState]:
    """
    Show a waypoint map and snap each click to the nearest waypoint.

    Controls:
        - Left click: select nearest waypoint
        - Right click: undo last selected waypoint
        - Enter/close window: finish selection

    Output:
        Prints copy-paste-ready coordinate lines such as:
            START_CARLA = carla.Location(x=..., y=..., z=0.000)
            END_CARLA = carla.Location(x=..., y=..., z=0.000)

    Notes:
        - WaypointState stores map coordinates in y-up convention.
        - CARLA coordinates are y-down, so printed CARLA y is negated.
        - z is set to 0.000 because WaypointState is 2D in this project.
    """
    if max_clicks <= 0:
        raise ValueError("max_clicks must be > 0")

    wps = waypoint_collection.waypoints
    if not wps:
        raise ValueError("WaypointStateCollection is empty, cannot select waypoints.")

    wp_x = np.array([wp.x for wp in wps])
    wp_y = np.array([wp.y for wp in wps])

    selected_indices: list[int] = []

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(
        f"Click up to {max_clicks} waypoint(s)  |  Left=select nearest  Right=undo  Enter/Close=finish",
        fontsize=12,
        color='white',
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')

    ax.scatter(wp_x, wp_y, s=2, c='#555555', zorder=1)

    # Keep labels tiny to avoid visual clutter on dense maps.
    for i, wp in enumerate(wps):
        ax.text(wp.x, wp.y, str(i), fontsize=2, color='white', alpha=0.7)

    highlight_scatter = ax.scatter([], [], s=90, c=[], zorder=3, edgecolors='white', linewidths=0.6)
    info_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        color='white',
        bbox=dict(boxstyle='round', facecolor='#222222', alpha=0.8),
    )

    def _print_pick(pick_number: int, wp: WaypointState) -> None:
        carla_x = float(wp.x)
        carla_y = float(-wp.y)
        print(f"CLICK_{pick_number}_WAYPOINT_ID = {wp.id}")
        print(f"CLICK_{pick_number}_MAP = ({wp.x:.3f}, {wp.y:.3f})")
        print(f"CLICK_{pick_number}_CARLA = carla.Location(x={carla_x:.3f}, y={carla_y:.3f}, z=0.000)")

    def _print_start_end_if_available() -> None:
        if len(selected_indices) < 2:
            return
        start_wp = wps[selected_indices[0]]
        end_wp = wps[selected_indices[1]]
        print(f"START_WAYPOINT_ID = {start_wp.id}")
        print(f"END_WAYPOINT_ID = {end_wp.id}")
        print(f"START_CARLA = carla.Location(x={start_wp.x:.3f}, y={-start_wp.y:.3f}, z=0.000)")
        print(f"END_CARLA = carla.Location(x={end_wp.x:.3f}, y={-end_wp.y:.3f}, z=0.000)")

    def _redraw() -> None:
        if not selected_indices:
            highlight_scatter.set_offsets(np.empty((0, 2)))
            info_text.set_text("No waypoints selected")
            fig.canvas.draw_idle()
            return

        xs = [wp_x[i] for i in selected_indices]
        ys = [wp_y[i] for i in selected_indices]
        highlight_scatter.set_offsets(np.column_stack([xs, ys]))

        colors = []
        for k in range(len(selected_indices)):
            if k == 0:
                colors.append('#00FF00')
            elif k == 1:
                colors.append('#FF0000')
            else:
                colors.append('#FFFF00')
        highlight_scatter.set_facecolors(colors)

        info_text.set_text(f"Selected: {len(selected_indices)}/{max_clicks}")
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        if event.inaxes != ax:
            return

        if event.button == 1:
            if len(selected_indices) >= max_clicks:
                return
            dist_sq = (wp_x - event.xdata) ** 2 + (wp_y - event.ydata) ** 2
            idx = int(np.argmin(dist_sq))
            selected_indices.append(idx)
            _print_pick(len(selected_indices), wps[idx])
            _print_start_end_if_available()
            _redraw()
            if len(selected_indices) >= max_clicks:
                plt.close(fig)
        elif event.button == 3:
            if selected_indices:
                selected_indices.pop()
            _redraw()

    def _on_key(event) -> None:
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    _redraw()
    plt.tight_layout()
    plt.show()

    if len(selected_indices) < max_clicks:
        raise RuntimeError(
            f"Selection finished with {len(selected_indices)} click(s), expected {max_clicks}."
        )

    return [wps[i] for i in selected_indices]


def select_start_end_waypoints_interactive(
    waypoint_collection: WaypointStateCollection,
) -> tuple[WaypointState, WaypointState]:
    """Convenience wrapper that selects exactly two snapped waypoints: start then end."""
    selected = select_closest_waypoints_interactive(waypoint_collection, max_clicks=2)
    return selected[0], selected[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open map waypoint picker, snap clicks to nearest waypoints, and print START/END coordinates."
    )
    parser.add_argument("--host", default=Config.CARLA_HOST, help="CARLA host")
    parser.add_argument("--port", type=int, default=Config.CARLA_PORT, help="CARLA port")
    parser.add_argument("--timeout", type=float, default=15.0, help="CARLA client timeout seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    bridge = SimulationBridge(world)
    waypoints = bridge.get_waypoints()
    if waypoints is None or not waypoints.waypoints:
        raise RuntimeError("No waypoints available from current CARLA map.")

    print(f"Loaded {len(waypoints.waypoints)} waypoints from map: {world.get_map().name}")
    print("Click two points: START then END.")

    select_start_end_waypoints_interactive(waypoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


