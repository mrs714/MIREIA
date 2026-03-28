from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

import carla


def _load_agents_modules():
    """Load CARLA agents modules with a local-workspace fallback path."""
    try:
        local_planner_mod = importlib.import_module("agents.navigation.local_planner")
        global_route_planner_mod = importlib.import_module("agents.navigation.global_route_planner")
        try:
            behavior_agent_mod = importlib.import_module("agents.navigation.behavior_agent")
        except ImportError:
            behavior_agent_mod = None
        return local_planner_mod, global_route_planner_mod, behavior_agent_mod
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        carla_pythonapi = repo_root / "PythonAPI" / "carla"
        if str(carla_pythonapi) not in sys.path:
            sys.path.insert(0, str(carla_pythonapi))
        local_planner_mod = importlib.import_module("agents.navigation.local_planner")
        global_route_planner_mod = importlib.import_module("agents.navigation.global_route_planner")
        try:
            behavior_agent_mod = importlib.import_module("agents.navigation.behavior_agent")
        except ImportError:
            behavior_agent_mod = None
        return local_planner_mod, global_route_planner_mod, behavior_agent_mod


_LOCAL_PLANNER_MOD, _GLOBAL_ROUTE_PLANNER_MOD, _BEHAVIOR_AGENT_MOD = _load_agents_modules()
LocalPlanner = _LOCAL_PLANNER_MOD.LocalPlanner
RoadOption = _LOCAL_PLANNER_MOD.RoadOption
GlobalRoutePlanner = _GLOBAL_ROUTE_PLANNER_MOD.GlobalRoutePlanner
BehaviorAgent = _BEHAVIOR_AGENT_MOD.BehaviorAgent if _BEHAVIOR_AGENT_MOD is not None else None


class SimpleRouteController:
    """
    Small wrapper that uses GlobalRoutePlanner.trace_route(start, end)
    and executes it with either:
      - BehaviorAgent (default): traffic-light + vehicle + pedestrian aware
      - LocalPlanner: pure waypoint follower (legacy behavior)
    """

    def __init__(self, target_speed: float = 10.0, sampling_resolution: float = 2.0,
                 local_planner_options: dict | None = None,
                 mode: str = "behavior_agent",
                 behavior: str = "normal"):
        mode = str(mode).strip().lower()
        if mode not in {"behavior_agent", "local_planner"}:
            raise ValueError("mode must be 'behavior_agent' or 'local_planner'")
        if mode == "behavior_agent" and BehaviorAgent is None:
            raise RuntimeError(
                "BehaviorAgent is unavailable (missing CARLA agent dependencies, e.g. shapely). "
                "Install dependencies or use mode='local_planner'."
            )

        self._target_speed = target_speed
        self._speed_modifier: Callable[[float, int, "SimpleRouteController"], float] | None = None
        self._tick_count = 0
        self._last_applied_target_speed = float(target_speed)
        self._mode = mode
        self._behavior = behavior
        self._sampling_resolution = sampling_resolution
        self._local_planner_options = local_planner_options.copy() if local_planner_options else {}
        self._local_planner_options.setdefault("target_speed", target_speed)
        self._local_planner_options.setdefault("dt", 0.05)
        self._local_planner_options.setdefault(
            "lateral_control_dict",
            {"K_P": 2.2, "K_I": 0.03, "K_D": 0.3, "dt": self._local_planner_options["dt"]},
        )
        self._local_planner_options.setdefault(
            "longitudinal_control_dict",
            {"K_P": 1.0, "K_I": 0.03, "K_D": 0.0, "dt": self._local_planner_options["dt"]},
        )
        self._last_plan = []
        self._goal_location: carla.Location | None = None
        self._arrival_threshold_m: float = 3.0

        self._vehicle: carla.Actor | None = None
        self._map: carla.Map | None = None
        self._global_planner = None
        self._local_planner = None
        self._behavior_agent = None

    def bind_vehicle(self, vehicle: carla.Actor, map_inst: carla.Map | None = None):
        self._vehicle = vehicle
        self._map = map_inst if map_inst is not None else vehicle.get_world().get_map()
        self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        if self._mode == "behavior_agent":
            # BehaviorAgent wraps BasicAgent+LocalPlanner and adds obstacle/light-aware behaviors.
            self._behavior_agent = BehaviorAgent(
                vehicle,
                behavior=self._behavior,
                opt_dict=self._local_planner_options,
                map_inst=self._map,
                grp_inst=self._global_planner,
            )
            self._local_planner = self._behavior_agent.get_local_planner()
        else:
            self._local_planner = LocalPlanner(vehicle, opt_dict=self._local_planner_options, map_inst=self._map)

    def set_target_speed(self, target_speed: float):
        """Set base target speed in km/h used by the speed modifier callback."""
        self._target_speed = max(0.0, float(target_speed))

    def set_speed_modifier(self, fn: Callable[[float, int, "SimpleRouteController"], float] | None):
        """Set optional callback to transform target speed each tick.

        Callback signature: fn(base_speed_kmh, tick_index, controller) -> target_speed_kmh
        """
        self._speed_modifier = fn

    def get_last_applied_target_speed(self) -> float:
        """Return last target speed sent to CARLA LocalPlanner (km/h)."""
        return self._last_applied_target_speed

    def set_destination(self, start_location: carla.Location, end_location: carla.Location,
                        clean_queue: bool = True):
        planner = self._require_local_planner()
        if self._global_planner is None:
            raise RuntimeError("Global route planner is not initialized. Call bind_vehicle first.")
        if self._map is None:
            raise RuntimeError("Map is not initialized. Call bind_vehicle first.")

        # Ensure route endpoints are valid driving waypoints to avoid off-road traces.
        # Always anchor the route start to the current ego position when available.
        start_source = self._vehicle.get_location() if self._vehicle is not None else start_location
        start_wp = self._map.get_waypoint(start_source, project_to_road=True, lane_type=carla.LaneType.Driving)
        end_wp = self._map.get_waypoint(end_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if start_wp is None or end_wp is None:
            raise RuntimeError("Could not project start/end to driving lane waypoints.")

        self._goal_location = end_wp.transform.location

        plan = self._global_planner.trace_route(start_wp.transform.location, end_wp.transform.location)
        if not plan:
            raise RuntimeError("GlobalRoutePlanner returned an empty plan.")

        # If planner starts away from the ego's current waypoint, prepend the local start.
        first_wp = plan[0][0]
        if start_wp.transform.location.distance(first_wp.transform.location) > 1.5:
            plan = [(start_wp, RoadOption.LANEFOLLOW)] + plan

        self._last_plan = plan
        self._tick_count = 0
        planner.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=clean_queue)

    def set_global_plan(self, plan: list[tuple[carla.Waypoint, RoadOption]], clean_queue: bool = True):
        planner = self._require_local_planner()
        self._last_plan = list(plan)
        self._tick_count = 0
        planner.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=clean_queue)

    def get_plan_length(self) -> int:
        return len(self._last_plan)

    def draw_plan(self, world: carla.World, max_points: int = 40, life_time: float = 0.1):
        plan_items = self._remaining_plan(max_points=max_points)
        for item in plan_items:
            wp = item[0] if isinstance(item, tuple) else item
            loc = wp.transform.location
            world.debug.draw_point(
                location=carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3),
                size=0.08,
                color=carla.Color(0, 200, 255),
                life_time=life_time,
            )

    def run_step(self, debug: bool = False) -> carla.VehicleControl:
        if self.done():
            # Hard-stop at destination to avoid creeping once the route is complete.
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        target_speed = self._compute_target_speed()
        self._apply_target_speed(target_speed)

        if self._behavior_agent is not None:
            control = self._behavior_agent.run_step(debug=debug)
        else:
            planner = self._require_local_planner()
            control = planner.run_step(debug=debug)

        self._tick_count += 1
        return control

    def done(self) -> bool:
        planner = self._require_local_planner()
        if planner.done():
            return True
        if self._vehicle is not None and self._goal_location is not None:
            return self._vehicle.get_location().distance(self._goal_location) <= self._arrival_threshold_m
        return False

    def _remaining_plan(self, max_points: int):
        planner = self._local_planner
        if planner is not None:
            for attr in ("_waypoints_queue", "waypoints_queue"):
                queue = getattr(planner, attr, None)
                if queue is not None:
                    try:
                        return list(queue)[:max_points]
                    except TypeError:
                        break
        return self._last_plan[:max_points]

    def _require_local_planner(self):
        if self._local_planner is None:
            raise RuntimeError("Controller is not bound to a vehicle. Call bind_vehicle first.")
        return self._local_planner

    def _compute_target_speed(self) -> float:
        speed = float(self._target_speed)
        if self._speed_modifier is not None:
            try:
                modified = self._speed_modifier(speed, self._tick_count, self)
            except TypeError:
                modified = self._speed_modifier(speed)
            speed = float(speed if modified is None else modified)
        speed = max(0.0, speed)
        self._last_applied_target_speed = speed
        return speed

    def _apply_target_speed(self, target_speed: float) -> None:
        """Propagate target speed to the active backend."""
        if self._behavior_agent is not None:
            # Base target speed used by BasicAgent/LocalPlanner.
            self._behavior_agent.set_target_speed(target_speed)
            # BehaviorAgent's high-level policy also caps with behavior.max_speed.
            behavior = getattr(self._behavior_agent, "_behavior", None)
            if behavior is not None and hasattr(behavior, "max_speed"):
                behavior.max_speed = target_speed
        else:
            planner = self._require_local_planner()
            planner.set_speed(target_speed)
