import os
import json
import subprocess
import carla

from MIREIA.simulation.bridge import SimulationBridge
from MIREIA.simulation.sensors import SensorManager
from MIREIA.simulation.traffic_handler import TrafficHandler
from MIREIA.config import Config


# ── Predefined CARLA weather presets ────────────────────────────────
_WEATHER_PRESETS: dict[str, carla.WeatherParameters] = {
    "ClearNoon":            carla.WeatherParameters.ClearNoon,
    "CloudyNoon":           carla.WeatherParameters.CloudyNoon,
    "WetNoon":              carla.WeatherParameters.WetNoon,
    "WetCloudyNoon":        carla.WeatherParameters.WetCloudyNoon,
    "SoftRainNoon":         carla.WeatherParameters.SoftRainNoon,
    "MidRainyNoon":         carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon":         carla.WeatherParameters.HardRainNoon,
    "ClearSunset":          carla.WeatherParameters.ClearSunset,
    "CloudySunset":         carla.WeatherParameters.CloudySunset,
    "WetSunset":            carla.WeatherParameters.WetSunset,
    "WetCloudySunset":      carla.WeatherParameters.WetCloudySunset,
    "SoftRainSunset":       carla.WeatherParameters.SoftRainSunset,
    "MidRainSunset":        carla.WeatherParameters.MidRainSunset,
    "HardRainSunset":       carla.WeatherParameters.HardRainSunset,
}


class Scenario:
    """
    A Scenario holds every parameter needed to reproduce an identical
    simulation: map, weather, ego vehicle config, traffic density, seed,
    and a human-readable description of the situation being tested.

    It does not contain any live CARLA objects — it is a pure data container
    that can be serialized to / deserialized from a JSON file.

    Each scenario owns a folder under ``Config.PATH_TO_SCENARIOS/<name>/``
    where derived artefacts are stored (pre-baked static risk map, routes, …).
    The folder (and its JSON file) are created on the first call to
    :meth:`save`.
    """

    def __init__(self, name: str,
                 map_name: str = 'Town03',
                 description: str = '',
                 # Weather — preset name OR dict of individual parameters
                 weather: str | dict = 'ClearNoon',
                 # Ego vehicle
                 ego_blueprint: str = 'vehicle.lincoln.mkz_2020',
                 ego_spawn_index: int | None = None,
                 ego_autopilot: bool = True,
                 # Traffic
                 n_vehicles: int = 30,
                 n_pedestrians: int = 20,
                 pct_running: float = 0.0,
                 pct_crossing: float = 0.0,
                 safe_vehicles: bool = True,
                 # Reproducibility
                 seed: int = 42):
        self.name = name
        self.description = description
        self.map_name = map_name
        # Weather
        self.weather = weather
        # Ego
        self.ego_blueprint = ego_blueprint
        self.ego_spawn_index = ego_spawn_index
        self.ego_autopilot = ego_autopilot
        # Traffic
        self.n_vehicles = n_vehicles
        self.n_pedestrians = n_pedestrians
        self.pct_running = pct_running
        self.pct_crossing = pct_crossing
        self.safe_vehicles = safe_vehicles
        # Seed
        self.seed = seed

    # ── Paths ───────────────────────────────────────────────────────
    @property
    def folder_path(self) -> str:
        """Absolute path to this scenario's dedicated folder."""
        return os.path.join(Config.PATH_TO_SCENARIOS, self.name)

    @property
    def json_path(self) -> str:
        """Path to the scenario definition JSON inside its folder."""
        return os.path.join(self.folder_path, "scenario.json")

    @property
    def baked_risk_path(self) -> str:
        """Path to the pre-baked static risk .npy file."""
        return os.path.join(self.folder_path, "baked_static_risk.npy")

    # ── Persistence ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of all scenario parameters."""
        return {
            "name":             self.name,
            "description":      self.description,
            "map_name":         self.map_name,
            "weather":          self.weather,
            "ego_blueprint":    self.ego_blueprint,
            "ego_spawn_index":  self.ego_spawn_index,
            "ego_autopilot":    self.ego_autopilot,
            "n_vehicles":       self.n_vehicles,
            "n_pedestrians":    self.n_pedestrians,
            "pct_running":      self.pct_running,
            "pct_crossing":     self.pct_crossing,
            "safe_vehicles":    self.safe_vehicles,
            "seed":             self.seed,
        }

    def save(self):
        """
        Persist the scenario to ``<folder>/scenario.json``.
        Creates the folder if it does not exist yet.
        """
        os.makedirs(self.folder_path, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, name: str) -> "Scenario":
        """
        Load a scenario from its JSON file.

        :param name: Scenario name (must match a folder under PATH_TO_SCENARIOS).
        :returns: A fully-initialized Scenario instance.
        """
        json_path = os.path.join(Config.PATH_TO_SCENARIOS, name, "scenario.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def get_weather_parameters(self) -> carla.WeatherParameters:
        """
        Convert the stored weather specification into a ``carla.WeatherParameters`` object.

        If ``self.weather`` is a string it is looked up in the built-in preset
        table.  If it is a dict, each key is passed to the ``WeatherParameters``
        constructor (e.g. ``{"cloudiness": 80, "precipitation": 50}``).
        """
        if isinstance(self.weather, str):
            if self.weather not in _WEATHER_PRESETS:
                raise ValueError(
                    f"Unknown weather preset '{self.weather}'. "
                    f"Available: {list(_WEATHER_PRESETS.keys())}"
                )
            return _WEATHER_PRESETS[self.weather]
        elif isinstance(self.weather, dict):
            return carla.WeatherParameters(**self.weather)
        else:
            raise TypeError(f"weather must be str or dict, got {type(self.weather)}")

    def __repr__(self):
        return (
            f"Scenario('{self.name}', map='{self.map_name}', desc='{self.description[:50] + '...' if len(self.description) > 50 else self.description}', "
            f"weather='{self.weather}', vehicles={self.n_vehicles}, "
            f"pedestrians={self.n_pedestrians}, seed={self.seed})"
        )


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
        self.client.set_timeout(20.0)
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

    # ── Sensor helpers ──────────────────────────────────────────────
    def setup_sensors(self, save_dir: str = "output",
                      ego_resolution: tuple[int, int] = (800, 600),
                      map_resolution: tuple[int, int] = (2000, 2000)) -> SensorManager:
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
        self.sensor_manager = SensorManager(
            self.world, world_map, self.ego_vehicle,
            save_dir=save_dir,
            ego_resolution=ego_resolution,
            map_resolution=map_resolution,
        )
        return self.sensor_manager

    # ── Simulation stepping ─────────────────────────────────────────
    def tick(self):
        """
        Advance the simulation by one step and update the bridge state.
        Only meaningful in synchronous mode.
        """
        self.world.tick()
        if self.bridge is not None:
            self.bridge.update()

    # ── Teardown ────────────────────────────────────────────────────
    def __teardown_scenario(self):
        """Destroy all actors spawned by the current scenario."""
        if self.traffic_handler is not None:
            self.traffic_handler.destroy_all()
            self.traffic_handler = None

        self.bridge = None
        self.sensor_manager = None
        self.ego_vehicle = None

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