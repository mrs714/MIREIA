import os
import json
from collections import Counter
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


def _weather_to_dict(weather: carla.WeatherParameters) -> dict:
    """Convert a WeatherParameters instance into a constructor-friendly dict."""
    keys = [
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "sun_azimuth_angle",
        "sun_altitude_angle",
        "fog_density",
        "fog_distance",
        "wetness",
        "fog_falloff",
        "scattering_intensity",
        "mie_scattering_scale",
        "rayleigh_scattering_scale",
        "dust_storm",
    ]
    return {k: getattr(weather, k) for k in keys if hasattr(weather, k)}


def _night_weather_from_preset(preset_name: str, add_fog: bool) -> dict:
    """Build weather from a preset with forced night sun altitude and optional fog."""
    weather = _weather_to_dict(_WEATHER_PRESETS[preset_name])
    weather["sun_altitude_angle"] = -90.0

    if add_fog:
        base_fog_density = float(weather.get("fog_density", 0.0))
        base_fog_distance = float(weather.get("fog_distance", 1000.0))
        base_fog_falloff = float(weather.get("fog_falloff", 0.0))

        # Guarantee meaningful fog in the fog-enabled scenarios.
        weather["fog_density"] = max(base_fog_density, 45.0)
        weather["fog_distance"] = min(base_fog_distance, 35.0)
        weather["fog_falloff"] = max(base_fog_falloff, 1.0)

    return weather


# Canonical ego camera offsets by blueprint (x, y, z) in vehicle coordinates.
EGO_CAMERA_POSITIONS: dict[str, tuple[float, float, float]] = {
    'vehicle.lincoln.mkz_2020': (0.8, 0.0, 1.3),
    'vehicle.tesla.model3': (0.8, 0.0, 1.3),
    'vehicle.audi.etron': (0.65, 0.0, 1.4),
    'vehicle.carlamotors.carlacola': (2.2, 0.0, 1.9),
}

_MIREIA_SLOT_LETTERS = ("A", "B", "C", "D")
_MIREIA_BASE_SLOT_MAX_SET = 16
_MIREIA_SPLIT_FILL_TOWNS = ("Town01", "Town02", "Town03", "Town04")
_MIREIA_SPLIT_HOLDOUT_TOWNS = ("Town05", "Town10HD")
_MIREIA_FORCED_BASE_SET_FILL_TOWNS: dict[int, str] = {
    9: "Town04",
    12: "Town04",
    15: "Town04",
}


def get_default_ego_camera_position(ego_blueprint: str) -> tuple[float, float, float] | None:
    """Return the default camera offset for a known ego blueprint."""
    return EGO_CAMERA_POSITIONS.get(ego_blueprint)


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
                 ego_spawn_point: tuple[float, float, float] | list[float] | None = None,
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
        if isinstance(ego_spawn_point, list):
            ego_spawn_point = tuple(ego_spawn_point)
        self.ego_spawn_point = ego_spawn_point
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
            "ego_spawn_point": list(self.ego_spawn_point) if self.ego_spawn_point is not None else None,
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


def _slot_key_from_name(scenario_name: str) -> tuple[int, str] | None:
    """Return (set_number, letter) from names like '03C_*'; otherwise None."""
    if len(scenario_name) < 3:
        return None
    set_token = scenario_name[:2]
    letter = scenario_name[2]
    if not set_token.isdigit() or letter not in _MIREIA_SLOT_LETTERS:
        return None
    return int(set_token), letter


def _is_base_slot(slot: tuple[int, str] | None) -> bool:
    if slot is None:
        return False
    set_number, _ = slot
    return set_number <= _MIREIA_BASE_SLOT_MAX_SET


def _replace_map_token_in_name(scenario_name: str, map_name: str) -> str:
    parts = scenario_name.split("_")
    if len(parts) < 3:
        return scenario_name
    parts[2] = map_name
    return "_".join(parts)


def _clone_scenario_with_map(scenario: Scenario, map_name: str) -> Scenario:
    payload = scenario.to_dict()
    payload["map_name"] = map_name
    payload["name"] = _replace_map_token_in_name(scenario.name, map_name)
    return Scenario(**payload)


def _pick_balanced_town(town_counts: Counter[str], fill_towns: tuple[str, ...]) -> str:
    min_count = min(int(town_counts.get(town, 0)) for town in fill_towns)
    for town in fill_towns:
        if int(town_counts.get(town, 0)) == min_count:
            return town
    return fill_towns[0]


def _pick_split_aware_fill_town(
    slot: tuple[int, str] | None,
    town_counts: Counter[str],
    fill_towns: tuple[str, ...],
    forced_set_fill_towns: dict[int, str],
) -> str:
    """
    Pick refill town for a base slot.

    Allows per-set hard overrides (e.g. 09/12/15 -> Town04) and falls back
    to balanced assignment across fill_towns.
    """
    if slot is not None:
        set_number, _ = slot
        forced = forced_set_fill_towns.get(set_number)
        if forced and forced in fill_towns:
            return forced
    return _pick_balanced_town(town_counts, fill_towns)


def _load_existing_slot_state(
    scenarios_root: str,
    fill_towns: tuple[str, ...],
) -> tuple[set[tuple[int, str]], Counter[str]]:
    """
    Scan existing scenario folders and return:
    - occupied slot keys (01A..16D)
    - current base-slot counts for fill_towns
    """
    occupied_slots: set[tuple[int, str]] = set()
    base_town_counts: Counter[str] = Counter()

    if not os.path.isdir(scenarios_root):
        return occupied_slots, base_town_counts

    for entry in sorted(os.listdir(scenarios_root)):
        scenario_dir = os.path.join(scenarios_root, entry)
        if not os.path.isdir(scenario_dir):
            continue
        if entry in {"videos", "__pycache__"}:
            continue

        slot = _slot_key_from_name(entry)
        if not _is_base_slot(slot):
            continue
        occupied_slots.add(slot)  # Any folder occupying this slot should be preserved.

        scenario_json_path = os.path.join(scenario_dir, "scenario.json")
        if not os.path.isfile(scenario_json_path):
            continue

        try:
            with open(scenario_json_path, "r", encoding="utf-8") as handle:
                scenario_meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        map_name = str(scenario_meta.get("map_name", "")).strip()
        if map_name in fill_towns:
            base_town_counts[map_name] += 1

    return occupied_slots, base_town_counts


def _build_legacy_mireia_dataset(target_count: int = 64) -> list[Scenario]:
    """Legacy procedural generation of MIREIA scenarios (preserved behavior)."""
    weathers = list(_WEATHER_PRESETS.keys())

    # 4 distinct ego vehicles to learn different camera heights and physics
    egos = [
        'vehicle.lincoln.mkz_2020',
        'vehicle.tesla.model3',
        'vehicle.audi.etron',
        'vehicle.carlamotors.carlacola',
    ]

    # Towns 01 through 07
    towns = [f'Town0{i}' for i in range(1, 6)] + ['Town10HD']

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
            ego_camera_position=get_default_ego_camera_position(egos[0]),
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
            ego_camera_position=get_default_ego_camera_position(egos[1]),
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
            ego_camera_position=get_default_ego_camera_position(egos[2]),
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
            ego_camera_position=get_default_ego_camera_position(egos[3]),
            n_vehicles=40,
            n_pedestrians=20,
            pct_running=0.0,
            pct_crossing=5.0,
            safe_vehicles=True,
        ))

        if len(scenarios) >= target_count:
            return scenarios[:target_count]

    # Additional nighttime sets 15A-D and 16A-D.
    # Each ego appears twice (once per set), with fog enabled on exactly one of those two.
    extra_specs = [
        (15, 'A', 'ClearNoon', 0, False),
        (15, 'B', 'CloudyNoon', 1, False),
        (15, 'C', 'WetNoon', 2, False),
        (15, 'D', 'HardRainNoon', 3, False),
        (16, 'A', 'ClearSunset', 0, True),
        (16, 'B', 'CloudySunset', 1, True),
        (16, 'C', 'SoftRainSunset', 2, True),
        (16, 'D', 'HardRainSunset', 3, True),
    ]

    # Keep map assignment pattern consistent with the base generator.
    set_maps: dict[int, tuple[str, str]] = {}
    for set_number in (15, 16):
        i = set_number - 1
        map_1 = towns[(i * 2) % len(towns)]
        map_2 = towns[(i * 2 + 1) % len(towns)]
        set_maps[set_number] = (map_1, map_2)

    for set_number, letter, preset_name, ego_idx, add_fog in extra_specs:
        map_1, map_2 = set_maps[set_number]
        map_name = map_1 if letter in ('A', 'B') else map_2
        high_volume = letter in ('A', 'C')
        n_vehicles = 80 if high_volume else 40
        n_pedestrians = 50 if high_volume else 20
        pct_running = 15.0 if high_volume else 0.0
        pct_crossing = 10.0 if high_volume else 5.0
        safe_vehicles = not high_volume

        weather = _night_weather_from_preset(preset_name, add_fog=add_fog)
        fog_tag = "Fog" if add_fog else "NoFog"
        density_tag = "HighVol" if high_volume else "LowVol"
        scenarios.append(Scenario(
            name=f"{set_number:02d}{letter}_{preset_name}_{map_name}_{density_tag}_{fog_tag}_Night",
            map_name=map_name,
            description=(
                f"Nighttime ({preset_name} baseline) with {'fog' if add_fog else 'no extra fog'}, "
                f"driving {egos[ego_idx]}."
            ),
            weather=weather,
            ego_blueprint=egos[ego_idx],
            ego_camera_position=get_default_ego_camera_position(egos[ego_idx]),
            n_vehicles=n_vehicles,
            n_pedestrians=n_pedestrians,
            pct_running=pct_running,
            pct_crossing=pct_crossing,
            safe_vehicles=safe_vehicles,
        ))

        if len(scenarios) >= target_count:
            return scenarios[:target_count]

    return scenarios


def generate_mireia_dataset(
    target_count: int = 64,
    *,
    split_aware: bool = False,
    fill_only_missing_slots: bool = False,
    scenarios_root: str | None = None,
    split_fill_towns: tuple[str, ...] = _MIREIA_SPLIT_FILL_TOWNS,
    split_holdout_towns: tuple[str, ...] = _MIREIA_SPLIT_HOLDOUT_TOWNS,
    forced_set_fill_towns: dict[int, str] | None = None,
) -> list[Scenario]:
    """
    Generate MIREIA scenarios.

    Defaults preserve legacy behavior.

    Optional split-aware mode can remap base-slot holdout towns to train-like towns
    and optionally return only missing base slots based on existing folders.

    Typical post-migration usage:
        generate_mireia_dataset(
            split_aware=True,
            fill_only_missing_slots=True,
            scenarios_root=Config.PATH_TO_SCENARIOS,
        )
    """
    scenarios = _build_legacy_mireia_dataset(target_count=target_count)

    # If we are filling only missing slots, use split-aware remapping by default
    # to avoid reintroducing Town04/Town10HD into vacated base slots.
    if fill_only_missing_slots:
        split_aware = True

    if not split_aware and not fill_only_missing_slots:
        return scenarios

    fill_towns = tuple(split_fill_towns)
    holdout_towns = set(split_holdout_towns)
    root = scenarios_root or Config.PATH_TO_SCENARIOS
    forced_by_set = dict(_MIREIA_FORCED_BASE_SET_FILL_TOWNS)
    if forced_set_fill_towns is not None:
        forced_by_set.update(forced_set_fill_towns)

    occupied_slots: set[tuple[int, str]] = set()
    town_counts: Counter[str] = Counter({town: 0 for town in fill_towns})

    if fill_only_missing_slots:
        occupied_slots, existing_counts = _load_existing_slot_state(root, fill_towns)
        town_counts.update(existing_counts)
    else:
        for scenario in scenarios:
            slot = _slot_key_from_name(scenario.name)
            if _is_base_slot(slot) and scenario.map_name in fill_towns:
                town_counts[scenario.map_name] += 1

    filtered: list[Scenario] = []
    for scenario in scenarios:
        slot = _slot_key_from_name(scenario.name)

        # Preserve all already-existing base slots and only create what is missing.
        if fill_only_missing_slots and _is_base_slot(slot) and slot in occupied_slots:
            continue

        current = scenario
        if split_aware and _is_base_slot(slot) and scenario.map_name in holdout_towns:
            next_town = _pick_split_aware_fill_town(
                slot=slot,
                town_counts=town_counts,
                fill_towns=fill_towns,
                forced_set_fill_towns=forced_by_set,
            )
            current = _clone_scenario_with_map(scenario, next_town)

        if _is_base_slot(slot) and current.map_name in fill_towns:
            town_counts[current.map_name] += 1

        filtered.append(current)

    return filtered


def save_mireia_scenarios(
    scenarios: list[Scenario],
    *,
    overwrite_existing: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Save scenarios to disk.

    Returns (saved_names, skipped_names).
    """
    saved: list[str] = []
    skipped: list[str] = []

    for scenario in scenarios:
        if (
            not overwrite_existing
            and os.path.isdir(scenario.folder_path)
            and os.path.isfile(scenario.json_path)
        ):
            skipped.append(scenario.name)
            continue

        scenario.save()
        saved.append(scenario.name)

    return saved, skipped


__all__ = [
    "Scenario",
    "EGO_CAMERA_POSITIONS",
    "get_default_ego_camera_position",
    "generate_mireia_dataset",
    "save_mireia_scenarios",
]
