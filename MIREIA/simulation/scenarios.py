import os
import json
import carla

from MIREIA.config import Config


# Predefined CARLA weather presets
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

    It does not contain any live CARLA objects - it is a pure data container
    that can be serialized to / deserialized from a JSON file.

    Each scenario owns a folder under ``Config.PATH_TO_SCENARIOS/<name>/``
    where derived artifacts are stored (pre-baked static risk map, routes, ...).
    The folder (and its JSON file) are created on the first call to
    :meth:`save`.
    """

    def __init__(self, name: str,
                 map_name: str = 'Town03',
                 description: str = '',
                 # Weather - preset name OR dict of individual parameters
                 weather: str | dict = 'ClearNoon',
                 # Ego vehicle
                 ego_blueprint: str = 'vehicle.lincoln.mkz_2020',
                 ego_camera_position: tuple[float, float, float] | list[float] | None = None,
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
        if isinstance(ego_camera_position, list):
            ego_camera_position = tuple(ego_camera_position)
        self.ego_camera_position = ego_camera_position
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

    # Paths
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

    # Persistence
    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of all scenario parameters."""
        return {
            "name":             self.name,
            "description":      self.description,
            "map_name":         self.map_name,
            "weather":          self.weather,
            "ego_blueprint":    self.ego_blueprint,
            "ego_camera_position": list(self.ego_camera_position) if self.ego_camera_position is not None else None,
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
        table. If it is a dict, each key is passed to the ``WeatherParameters``
        constructor (e.g. ``{"cloudiness": 80, "precipitation": 50}``).
        """
        if isinstance(self.weather, str):
            if self.weather not in _WEATHER_PRESETS:
                raise ValueError(
                    f"Unknown weather preset '{self.weather}'. "
                    f"Available: {list(_WEATHER_PRESETS.keys())}"
                )
            return _WEATHER_PRESETS[self.weather]
        if isinstance(self.weather, dict):
            return carla.WeatherParameters(**self.weather)
        raise TypeError(f"weather must be str or dict, got {type(self.weather)}")

    def __repr__(self):
        return (
            f"Scenario('{self.name}', map='{self.map_name}', desc='{self.description[:50] + '...' if len(self.description) > 50 else self.description}', "
            f"weather='{self.weather}', vehicles={self.n_vehicles}, "
            f"pedestrians={self.n_pedestrians}, seed={self.seed})"
        )


def generate_mireia_dataset(target_count: int = 54) -> list[Scenario]:
    """Procedural generation of MIREIA scenarios."""
    weathers = list(_WEATHER_PRESETS.keys())

    # 4 distinct ego vehicles to learn different camera heights and physics
    egos = [
        'vehicle.lincoln.mkz_2020',
        'vehicle.tesla.model3',
        'vehicle.audi.etron',
        'vehicle.carlamotors.carlacola',
    ]

    ego_camera_positions = {
        'vehicle.lincoln.mkz_2020': (0.0, 0.0, 1.5),
        'vehicle.tesla.model3': (0.0, 0.0, 1.5),
        'vehicle.audi.etron': (0.0, 0.0, 1.5),
        'vehicle.carlamotors.carlacola': (0.0, 0.0, 1.5),
    }

    # Towns 01 through 07
    towns = [f'Town0{i}' for i in range(1, 8)]

    scenarios: list[Scenario] = []

    for i, weather in enumerate(weathers):
        # Cycle through towns to ensure all 7 maps get equal representation
        # across the weathers
        map_1 = towns[(i * 2) % len(towns)]
        map_2 = towns[(i * 2 + 1) % len(towns)]

        # Map 1 configurations
        scenarios.append(Scenario(
            name=f"{i+1:02d}A_{weather}_{map_1}_HighVol",
            map_name=map_1,
            description=f"High density traffic, aggressive AI, driving {egos[0]}.",
            weather=weather,
            ego_blueprint=egos[0],
            ego_camera_position=ego_camera_positions.get(egos[0]),
            n_vehicles=80,
            n_pedestrians=50,
            pct_running=15.0,
            pct_crossing=10.0,
            safe_vehicles=False,
        ))

        scenarios.append(Scenario(
            name=f"{i+1:02d}B_{weather}_{map_1}_LowVol",
            map_name=map_1,
            description=f"Low density traffic, safe AI, driving {egos[1]}.",
            weather=weather,
            ego_blueprint=egos[1],
            ego_camera_position=ego_camera_positions.get(egos[1]),
            n_vehicles=40,
            n_pedestrians=20,
            pct_running=0.0,
            pct_crossing=5.0,
            safe_vehicles=True,
        ))

        # Map 2 configurations
        scenarios.append(Scenario(
            name=f"{i+1:02d}C_{weather}_{map_2}_HighVol",
            map_name=map_2,
            description=f"High density traffic, aggressive AI, driving {egos[2]}.",
            weather=weather,
            ego_blueprint=egos[2],
            ego_camera_position=ego_camera_positions.get(egos[2]),
            n_vehicles=80,
            n_pedestrians=50,
            pct_running=15.0,
            pct_crossing=10.0,
            safe_vehicles=False,
        ))

        scenarios.append(Scenario(
            name=f"{i+1:02d}D_{weather}_{map_2}_LowVol",
            map_name=map_2,
            description="Low density traffic, safe AI, driving a heavy truck.",
            weather=weather,
            ego_blueprint=egos[3],
            ego_camera_position=ego_camera_positions.get(egos[3]),
            n_vehicles=40,
            n_pedestrians=20,
            pct_running=0.0,
            pct_crossing=5.0,
            safe_vehicles=True,
        ))

        if len(scenarios) >= target_count:
            return scenarios[:target_count]

    return scenarios


__all__ = [
    "Scenario",
    "generate_mireia_dataset",
]
