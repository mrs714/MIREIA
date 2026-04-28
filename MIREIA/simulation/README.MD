# MIREIA Simulation

CARLA simulator integration — world management, scenario loading, traffic, sensors, and trial execution.

## Modules

| Module | Key Class | Description |
|--------|-----------|-------------|
| `world_manager.py` | `WorldManager` | Central orchestrator: connects to CARLA, loads scenarios, spawns sensors, ticks world, manages cleanup |
| `bridge.py` | `SimulationBridge` | State snapshot per tick: ego kinematics, dynamic obstacle list, static obstacle list, environment state |
| `scenarios.py` | `Scenario` | Scenario dataclass + factories; weather presets; night/fog variant generation; Town4↔Town10 migration |
| `sensors.py` | `SensorManager` | Attaches RGB ego camera, top-down map camera, and optional extras; handles resolution, FOV, positioning |
| `traffic_handler.py` | `TrafficHandler` | Deterministic spawner for ego, traffic vehicles, and pedestrians using seeded RNG; Traffic Manager sync |
| `simple_route_controller.py` | `SimpleRouteController` | Route following via CARLA agents (GlobalRoutePlanner / LocalPlanner / BehaviorAgent) with speed modifier hooks |
| `trials.py` | `TrialDefinition` / `TrialRunner` | Bundles map + weather + route + ego config into a reproducible trial; runs multiple ego sub-trials on a shared world |
| `routes.py` | — | Interactive waypoint selection tool (click → snap to nearest waypoint → output CARLA coordinates) |

## Key Concepts

### WorldManager lifecycle

```python
wm = WorldManager(verbose=True)
wm.load_scenario(scenario)   # connects, loads map, sets weather, spawns traffic
sensors = wm.setup_sensors() # attaches cameras
wm.enable_recording(...)     # starts JSONL logging
for _ in range(n):
    wm.tick(...)             # advances simulation + logs one frame
wm.destroy()                 # cleans up actors and sensors
```

### SimulationBridge

Updated once per tick via `bridge.update()`. Provides:
- `get_ego_kinematics()` → position, velocity, heading
- `get_dynamic_obstacles()` → list of nearby vehicles/pedestrians
- `get_static_obstacles()` → pre-baked static risk grid waypoints
- `get_environment_state()` → visibility, road friction

### Scenario naming convention

```
{NN}[A-D]_{Weather}_{Town}_{TrafficVolume}[_Modifiers]
```

- `NN` — set number (01–16 train, 17–21 Town05 val, 22+ Town10HD holdout)
- Letter — A (high vol), B (low vol), C (high vol alt weather), D (low vol alt weather)
- Modifiers — e.g. `_NoFog_Night`

### Trial system

`TrialDefinition` pins the world setup (map, weather, traffic density, seed, route).
`EgoTrialConfig` pins one ego agent (vehicle type, target speed, controller mode).
`TrialRunner.run_trial(trial, ego_configs, ...)` runs each `EgoTrialConfig` in sequence on the same loaded world, saving per-subtrial datasets to `trials/<trial_name>/runs/<timestamp>_<subtrial_name>/`.

## Configuration (from `Config`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CARLA_HOST` | `127.0.0.1` | CARLA server host |
| `CARLA_PORT` | `2000` | CARLA server port |
| `SIM_FIXED_DELTA_SECONDS` | `0.05` | Simulation timestep (20 Hz) |
| `RECORD_EVERY_N_TICKS` | `5` | Frame capture stride (→ 4 fps) |
| `PATH_TO_SCENARIOS` | (env var) | Root directory for scenario folders |
| `PATH_TO_TRIALS` | (env var) | Root directory for trial run output |
