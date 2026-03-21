import os
import json
import subprocess
import numpy as np
import carla

from MIREIA.simulation.bridge import SimulationBridge
from MIREIA.simulation.sensors import SensorManager
from MIREIA.simulation.traffic_handler import TrafficHandler
from MIREIA.simulation.scenarios import Scenario
from MIREIA.data_collection.recorder import DatasetLogger
from MIREIA.core.physics import RiskOracle
from MIREIA.analysis.plotter import Grid, RiskGrid
from MIREIA.analysis.visualizer import render_risk_map_with_actors
from MIREIA.config import Config


class WorldManager:
    """
    Orchestrates the full CARLA simulation lifecycle:

    1. Connects to a running CARLA server.
    2. Loads a :class:`Scenario` — sets the map, weather, spawns traffic and
       the ego vehicle deterministically.
    3. Creates the :class:`SimulationBridge` (state access),
       :class:`SensorManager` (cameras), and :class:`TrafficHandler` (actors).
    4. Supports hot-switching between scenarios via :meth:`load_scenario`.
    """

    def __init__(self, scenario: Scenario | None = None,
                 sync_mode: bool = True,
                 fixed_delta: float = 0.05,
                 verbose: bool = False):
        """
        :param scenario: Initial scenario to load. If None the manager will
            connect to CARLA but not set up any scenario until
            :meth:`load_scenario` is called.
        :param sync_mode: Enable synchronous mode on the CARLA world.
        :param fixed_delta: Fixed timestep in seconds (only in sync mode).
        :param verbose: Print progress messages during setup.
        """
        self.verbose = verbose
        self._sync_mode = sync_mode
        self._fixed_delta = fixed_delta

        # CARLA handles — populated by __connect_carla
        self.client: carla.Client = None
        self.world: carla.World = None

        # Scenario-lifetime objects — populated by __setup_scenario
        self.scenario: Scenario | None = None
        self.traffic_handler: TrafficHandler | None = None
        self.bridge: SimulationBridge | None = None
        self.sensor_manager: SensorManager | None = None
        self.ego_vehicle: carla.Actor | None = None
        self.dataset_logger: DatasetLogger | None = None
        self.risk_oracle = RiskOracle()
        self.baked_static_risk: RiskGrid | None = None
        self._static_risk_resolution: float = 2.0
        self._static_risk_margin: float = 20.0
        self._static_risk_image_path: str | None = None
        self._static_risk_image_resolution: tuple[int, int] = (1024, 1024)
        self._static_risk_image_dpi: int = 150
        self._record_topdown: bool = False
        self._record_static_risk_image: bool = False

        # Connect to CARLA
        self.__connect_carla()

        # Load the initial scenario if one was provided
        if scenario is not None:
            self.load_scenario(scenario)

    # ── CARLA connection ────────────────────────────────────────────
    def __connect_carla(self):
        """
        Connect to a running CARLA server using the host/port from Config.
        Applies synchronous mode and fixed timestep if requested.
        """
        if self.verbose:
            print(f"Connecting to CARLA at {Config.CARLA_HOST}:{Config.CARLA_PORT}...")

        self.client = carla.Client(Config.CARLA_HOST, Config.CARLA_PORT)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()

        if self.verbose:
            current_map = self.world.get_map().name
            print(f"Connected. Current map: '{current_map}'.")

    def __apply_world_settings(self):
        """Apply synchronous mode and fixed delta to the CARLA world."""
        settings = self.world.get_settings()
        settings.synchronous_mode = self._sync_mode
        if self._sync_mode:
            settings.fixed_delta_seconds = self._fixed_delta
        self.world.apply_settings(settings)

    # ── Scenario lifecycle ──────────────────────────────────────────
    def load_scenario(self, scenario: Scenario):
        """
        Tear down any active scenario and set up a new one.

        This is the main entry point for switching between scenarios.
        It will:
        1. Destroy all actors from the previous scenario.
        2. Load the correct map (if it differs from the current one).
        3. Apply weather conditions.
        4. Spawn the ego vehicle and traffic deterministically.
        5. Build a fresh SimulationBridge.

        :param scenario: The Scenario to load.
        """
        # Clean up previous scenario if one was active
        self.__teardown_scenario()

        self.scenario = scenario

        if self.verbose:
            print(f"Loading scenario '{scenario.name}' (map={scenario.map_name}, seed={scenario.seed})...")

        # Ensure the scenario folder exists and save its JSON
        scenario.save()

        self.__load_map()
        self.__apply_world_settings()
        self.__apply_weather()
        self.__spawn_traffic()
        self.__initialize_bridge()
        self.__bake_static_risk_map()
        self._static_risk_image_path = None

        if self.verbose:
            print(f"Scenario '{scenario.name}' is ready.")

    def __load_map(self):
        """Load the scenario's map if it differs from the one currently active."""
        current_map = self.world.get_map().name
        target_map = self.scenario.map_name

        # CARLA map names look like '/Game/Carla/Maps/Town03'; we compare the
        # tail so the user can pass just 'Town03'.
        if not current_map.endswith(target_map):
            if self.verbose:
                print(f"Switching map from '{current_map}' to '{target_map}'...")
            self.world = self.client.load_world(target_map)
        else:
            # Same map — reload to get a clean slate (no leftover actors)
            self.world = self.client.reload_world()

        # Let the world settle for one tick after (re)loading
        self.world.tick()

    def __apply_weather(self):
        """Set the weather defined in the current scenario."""
        weather_params = self.scenario.get_weather_parameters()
        self.world.set_weather(weather_params)
        if self.verbose:
            print(f"Weather set to '{self.scenario.weather}'.")

    def __spawn_traffic(self):
        """
        Spawn ego vehicle and NPC traffic using :class:`TrafficHandler`.
        All randomness is governed by the scenario's seed.
        """
        self.traffic_handler = TrafficHandler(
            self.client, self.world, seed=self.scenario.seed
        )

        self.ego_vehicle = self.traffic_handler.spawn_ego(
            blueprint_id=self.scenario.ego_blueprint,
            spawn_index=self.scenario.ego_spawn_index,
            autopilot=self.scenario.ego_autopilot,
        )

        if self.scenario.n_vehicles > 0:
            self.traffic_handler.spawn_vehicles(
                n=self.scenario.n_vehicles,
                safe=self.scenario.safe_vehicles,
            )

        if self.scenario.n_pedestrians > 0:
            self.traffic_handler.spawn_pedestrians(
                n=self.scenario.n_pedestrians,
                pct_running=self.scenario.pct_running,
                pct_crossing=self.scenario.pct_crossing,
            )

    def __initialize_bridge(self):
        """
        Create a :class:`SimulationBridge` that scans the world for all
        actors spawned by the traffic handler.
        """
        self.bridge = SimulationBridge(self.world)
        if self.verbose:
            print(f"SimulationBridge initialized: {self.bridge}")

    def __bake_static_risk_map(self):
        if self.bridge is None:
            raise RuntimeError("Cannot bake static risk map: SimulationBridge is not initialized.")
            return

        bounds = self.__compute_map_bounds()
        if bounds is None:
            raise RuntimeError("Cannot bake static risk map: failed to compute map bounds.")
            return
        
        if self.verbose:
            print(f"Computing static risk map with bounds: {bounds} and resolution: {self._static_risk_resolution}m...")

        center_x, center_y, size = bounds
        grid = Grid(
            center_x=center_x,
            center_y=center_y,
            size=size,
            resolution=self._static_risk_resolution,
        )
        self.baked_static_risk = self.risk_oracle.bake_static_risk(grid, self.bridge)

    def save_static_risk_map_image(self, save_path: str | None = None,
                                   resolution: tuple[int, int] | None = None,
                                   dpi: int | None = None,
                                   vmax: float | None = None) -> str:
        """
        Render and save a static risk heatmap for the entire map.

        :param save_path: Optional output file path. Defaults to the scenario folder.
        :param resolution: Output image resolution (width, height) in pixels.
        :param dpi: Dots-per-inch for the saved image.
        :param vmax: Optional fixed max value for the color scale.
        :returns: The saved image path.
        """
        if self.scenario is None:
            raise RuntimeError("Cannot save static risk map: no scenario loaded.")
        if self.baked_static_risk is None:
            raise RuntimeError("Cannot save static risk map: static risk map is not baked.")

        resolution = resolution or self._static_risk_image_resolution
        dpi = dpi or self._static_risk_image_dpi

        if save_path is None:
            save_path = os.path.join(
                self.scenario.folder_path,
                f"risk_map_{resolution[0]}x{resolution[1]}.png",
            )

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        render_risk_map_with_actors(
            self.baked_static_risk,
            world=self.world,
            bridge=self.bridge,
            save_path=save_path,
            resolution=resolution,
            dpi=dpi,
            vmax=vmax,
        )
        self._static_risk_image_path = save_path
        return save_path

    def save_risk_frame_image(self, save_path: str,
                              resolution: tuple[int, int] | None = None,
                              dpi: int | None = None,
                              vmax: float | None = None) -> str:
        """
        Render and save a per-frame risk map with live actor overlays.

        :param save_path: Output image path.
        :param resolution: Output image resolution (width, height) in pixels.
        :param dpi: Dots-per-inch for the saved image.
        :param vmax: Optional fixed max value for the color scale.
        :returns: The saved image path.
        """
        if self.scenario is None:
            raise RuntimeError("Cannot save risk frame image: no scenario loaded.")
        if self.baked_static_risk is None:
            raise RuntimeError("Cannot save risk frame image: static risk map is not baked.")

        resolution = resolution or self._static_risk_image_resolution
        dpi = dpi or self._static_risk_image_dpi

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        render_risk_map_with_actors(
            self.baked_static_risk,
            world=self.world,
            bridge=self.bridge,
            save_path=save_path,
            resolution=resolution,
            dpi=dpi,
            vmax=vmax,
        )
        return save_path

    def compose_dataset_video(self, fps: int = 10, dataset_jsonl_path: str | None = None) -> str:
        """
        Build a dataset video from the current scenario's dataset JSONL.

        :param fps: Frames per second for the output video.
        :param dataset_jsonl_path: Optional path override for the dataset JSONL.
        :returns: Path to the rendered video.
        """
        if self.scenario is None:
            raise RuntimeError("Cannot compose dataset video: no scenario loaded.")

        dataset_jsonl_path = dataset_jsonl_path or os.path.join(
            self.scenario.folder_path, "dataset.jsonl"
        )

        from MIREIA.analysis.visualizer import DatasetVideoComposer

        composer = DatasetVideoComposer(dataset_jsonl_path, fps=fps)
        return composer.build_video()

    def __compute_map_bounds(self) -> tuple[float, float, float] | None:
        waypoints = self.bridge.get_waypoints() if self.bridge else None
        if waypoints is None or not waypoints.waypoints:
            return None

        xs = np.array([wp.x for wp in waypoints.waypoints])
        ys = np.array([wp.y for wp in waypoints.waypoints])

        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        size = max(max_x - min_x, max_y - min_y) + (2.0 * self._static_risk_margin)

        return center_x, center_y, size

    def get_risk(self) -> float:
        if self.bridge is None:
            raise RuntimeError("Cannot compute risk: SimulationBridge is not initialized.")
        if self.baked_static_risk is None:
            raise RuntimeError("Cannot compute risk: static risk map is not baked.")

        ego = self.bridge.get_ego_kinematics()
        if ego is None:
            raise RuntimeError("Cannot compute risk: ego vehicle not available.")

        return self.risk_oracle.calculate_scene_risk(
            (ego.x, ego.y), self.bridge, self.baked_static_risk
        )

    # ── Dataset recording ───────────────────────────────────────────
    def enable_recording(self, append: bool = True,
                         include_topdown: bool = False,
                         include_static_risk_image: bool = False,
                         static_risk_image_resolution: tuple[int, int] = (1024, 1024),
                         static_risk_image_dpi: int = 150) -> DatasetLogger:
        """
        Create a :class:`DatasetLogger` that writes frame data to a JSONL
        file inside the scenario's own folder
        (``<PATH_TO_SCENARIOS>/<name>/dataset.jsonl``).
        Call after :meth:`load_scenario`.

        :param append: Append to an existing file (True) or overwrite (False).
        :param include_topdown: If True, log the top-down image path when provided.
        :param include_static_risk_image: If True, log the static risk map image path.
        :param static_risk_image_resolution: Output resolution for the static risk map image.
        :param static_risk_image_dpi: DPI used when saving the static risk map image.
        :returns: The created DatasetLogger instance.
        """
        if self.scenario is None:
            raise RuntimeError("No scenario loaded — call load_scenario first.")

        jsonl_path = os.path.join(self.scenario.folder_path, "dataset.jsonl")
        self.dataset_logger = DatasetLogger(jsonl_path, append=append)
        self._record_topdown = include_topdown
        self._record_static_risk_image = include_static_risk_image
        self._static_risk_image_resolution = static_risk_image_resolution
        self._static_risk_image_dpi = static_risk_image_dpi

        if self._record_static_risk_image:
            self.save_static_risk_map_image(
                resolution=self._static_risk_image_resolution,
                dpi=self._static_risk_image_dpi,
            )

        if self.verbose:
            print(f"Recording enabled → {jsonl_path}")

        return self.dataset_logger

    # ── Sensor helpers ──────────────────────────────────────────────
    def setup_sensors(self, save_dir: str = "output",
                      ego_resolution: tuple[int, int] = (512, 512),
                      map_resolution: tuple[int, int] = (100, 100),
                      enable_map_camera: bool = True,
                      ego_camera_fov: float = 110.0,
                      map_fov: float = 90.0,
                      ego_camera_position: tuple[float, float, float] | None = None,
                      align_risk_rotation: bool = True) -> SensorManager:
        """
        Attach cameras to the ego vehicle.  Call after :meth:`load_scenario`.

        :param save_dir: Directory to write captured frames to.
        :param ego_resolution: (width, height) for the front-mounted camera.
        :param map_resolution: (width, height) for the bird's-eye camera.
        :returns: The created SensorManager instance.
        """
        if self.ego_vehicle is None:
            raise RuntimeError("No ego vehicle — load a scenario first.")
        world_map = self.world.get_map()
        map_center = None
        map_size = None
        if enable_map_camera:
            bounds = self.__compute_map_bounds()
            if bounds is not None:
                center_x, center_y, size = bounds
                map_center = (center_x, -center_y)
                map_size = size

        map_rotation_yaw = -90.0 if align_risk_rotation else 0.0
        map_rotation_roll = 0.0

        if ego_camera_position is None and self.scenario is not None:
            ego_camera_position = self.scenario.ego_camera_position

        self.sensor_manager = SensorManager(
            self.world, world_map, self.ego_vehicle,
            save_dir=save_dir,
            ego_resolution=ego_resolution,
            map_resolution=map_resolution,
            enable_map_camera=enable_map_camera,
            ego_camera_position=ego_camera_position,
            ego_camera_fov=ego_camera_fov,
            map_center=map_center,
            map_size=map_size,
            map_fov=map_fov,
            map_rotation_yaw=map_rotation_yaw,
            map_rotation_roll=map_rotation_roll,
        )
        return self.sensor_manager

    # ── Simulation stepping ─────────────────────────────────────────
    def tick(self, ground_truth_risk: float | None = None,
             rgb_image_path: str = "",
             topdown_image_path: str = "",
             risk_map_image_path: str = "") -> dict | None:
        """
        Advance the simulation by one step, update the bridge state,
        and — if recording is enabled — log the frame.

        :param ground_truth_risk: Risk label for this frame.  Required when
            a DatasetLogger is active; ignored otherwise.
        :param rgb_image_path: Relative path to the saved RGB image.
        :param topdown_image_path: Relative path to the saved top-down image.
        :param risk_map_image_path: Relative path to the static risk map image.
        :returns: The logged record dict, or *None* if recording is off.
        """
        self.world.tick()
        if self.bridge is not None:
            self.bridge.update()

        record = None
        if self.dataset_logger is not None and self.bridge is not None:
            if not self._record_topdown:
                topdown_image_path = ""
            if self._record_static_risk_image and not risk_map_image_path:
                if self._static_risk_image_path is None:
                    self.save_static_risk_map_image(
                        resolution=self._static_risk_image_resolution,
                        dpi=self._static_risk_image_dpi,
                    )
                risk_map_image_path = self._static_risk_image_path or ""
            record = self.dataset_logger.log_frame(
                bridge=self.bridge,
                scenario=self.scenario,
                ego_vehicle=self.ego_vehicle,
                frame_id=self.dataset_logger.frame_count,
                ground_truth_risk=ground_truth_risk,
                rgb_image_path=rgb_image_path,
                topdown_image_path=topdown_image_path,
                risk_map_image_path=risk_map_image_path,
                risk_oracle=self.risk_oracle,
                baked_static_risk=self.baked_static_risk,
            )
        return record

    # ── Teardown ────────────────────────────────────────────────────
    def __teardown_scenario(self):
        """Destroy all actors spawned by the current scenario."""
        if self.dataset_logger is not None:
            self.dataset_logger.close()
            self.dataset_logger = None

        if self.traffic_handler is not None:
            self.traffic_handler.destroy_all()
            self.traffic_handler = None

        self.bridge = None
        self.sensor_manager = None
        self.ego_vehicle = None
        self._static_risk_image_path = None

    def destroy(self):
        """
        Public cleanup — tears down the scenario and restores the world
        to asynchronous mode so it is not left frozen.
        """
        self.__teardown_scenario()
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            self.world.apply_settings(settings)
        self.scenario = None

    def __repr__(self):
        scenario_str = self.scenario.name if self.scenario else "None"
        return f"WorldManager(scenario='{scenario_str}', sync={self._sync_mode})"