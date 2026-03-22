# Defines Start/End points for standard scenarios
import json
import matplotlib.pyplot as plt
import numpy as np
from MIREIA.simulation.bridge import WaypointState, WaypointStateCollection

class Route:
    """
    A route on a map.
    """
    
    def __init__(self, route_id: str):
        self.route_id: str = route_id
        self.start: WaypointState = None
        self.end: WaypointState = None
        self.waypoints: list[WaypointState] = []

def create_route_from_waypoints(waypoint_collection: WaypointStateCollection) -> Route:
    """
    Given the waypoints extracted by the Bridge on a Map, opens an interactive interface to create routes by selecting waypoints. 
    The first selected waypoint will be the start, the last one the end, and the ones in between will be the waypoints of the route, in the order they were selected.
    
    Controls:
        - Left click: select the nearest waypoint (green=start, yellow=intermediate, red=end updates live)
        - Right click: undo the last selection
        - Enter/close window: confirm the route
    """
    wps = waypoint_collection.waypoints
    if not wps:
        raise ValueError("WaypointStateCollection is empty, cannot create a route.")

    wp_x = np.array([wp.x for wp in wps])
    wp_y = np.array([wp.y for wp in wps])

    selected_indices: list[int] = []

    # --- Set up the plot ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title("Click waypoints to build a route  |  Left=add  Right=undo  Enter/Close=confirm",
                 fontsize=12, color='white')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')

    # Draw all waypoints as small grey dots
    ax.scatter(wp_x, wp_y, s=2, c='#555555', zorder=1)

    # Add waypoint IDs as small text next to each point
    for i, wp in enumerate(wps):
        ax.text(wp.x, wp.y, str(i), fontsize=2, color='white', alpha=0.7)

    # Containers for dynamic artists
    highlight_scatter = ax.scatter([], [], s=80, c=[], zorder=3, edgecolors='white', linewidths=0.5)
    route_line, = ax.plot([], [], '-', color='#00FF88', linewidth=2, alpha=0.7, zorder=2)
    info_text = ax.text(0.01, 0.99, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='white',
                        bbox=dict(boxstyle='round', facecolor='#222222', alpha=0.8))

    def _redraw():
        """Update the highlight markers, route line, and info text."""
        if not selected_indices:
            highlight_scatter.set_offsets(np.empty((0, 2)))
            route_line.set_data([], [])
            info_text.set_text("No waypoints selected")
            fig.canvas.draw_idle()
            return

        xs = [wp_x[i] for i in selected_indices]
        ys = [wp_y[i] for i in selected_indices]
        offsets = np.column_stack([xs, ys])
        highlight_scatter.set_offsets(offsets)

        # Color: green for start, red for last (end), yellow for intermediate
        n = len(selected_indices)
        colors = []
        for k in range(n):
            if k == 0:
                colors.append('#00FF00')   # start = green
            elif k == n - 1 and n > 1:
                colors.append('#FF0000')   # end = red
            else:
                colors.append('#FFFF00')   # intermediate = yellow
        highlight_scatter.set_facecolors(colors)

        route_line.set_data(xs, ys)

        info_text.set_text(f"Selected: {n} waypoint(s)  |  Start → ... → End")
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click — add nearest waypoint
            dist_sq = (wp_x - event.xdata)**2 + (wp_y - event.ydata)**2
            idx = int(np.argmin(dist_sq))
            if idx not in selected_indices:
                selected_indices.append(idx)
            _redraw()
        elif event.button == 3:  # Right click — undo last
            if selected_indices:
                selected_indices.pop()
            _redraw()

    def _on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    _redraw()
    plt.tight_layout()
    plt.show()

    # --- Build the Route from the selection ---
    route = Route(route_id="interactive")
    if selected_indices:
        route.start = wps[selected_indices[0]]
        route.waypoints = [wps[i] for i in selected_indices]
        if len(selected_indices) > 1:
            route.end = wps[selected_indices[-1]]

    print(f"Route created with {len(route.waypoints)} waypoint(s).")
    return route


def route_to_dict(route: Route) -> dict:
    waypoint_ids = [wp.id for wp in route.waypoints]
    return {
        "route_id": route.route_id,
        "waypoint_ids": waypoint_ids,
    }


def save_route_json(route: Route, output_path: str) -> None:
    data = route_to_dict(route)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_route_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_route_from_waypoint_ids(
    waypoint_collection: WaypointStateCollection,
    waypoint_ids: list[int],
    route_id: str = "loaded",
) -> Route:
    waypoint_map = {wp.id: wp for wp in waypoint_collection.waypoints}
    route = Route(route_id=route_id)
    route.waypoints = [waypoint_map[wp_id] for wp_id in waypoint_ids if wp_id in waypoint_map]
    if route.waypoints:
        route.start = route.waypoints[0]
        route.end = route.waypoints[-1]
    return route


def load_route_from_json(
    path: str,
    waypoint_collection: WaypointStateCollection,
) -> Route:
    data = load_route_json(path)
    waypoint_ids = data.get("waypoint_ids", [])
    route_id = data.get("route_id", "loaded")
    return build_route_from_waypoint_ids(waypoint_collection, waypoint_ids, route_id=route_id)


