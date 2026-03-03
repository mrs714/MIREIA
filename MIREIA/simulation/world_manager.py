import os, subprocess
import carla
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor
import logging
import random as stdlib_random

from MIREIA.simulation.routes import Route
from MIREIA.simulation.bridge import SimulationBridge
from MIREIA.simulation.sensors import SensorManager
from MIREIA.core.physics import RiskOracle
from MIREIA.analysis.plotter import draw_risk_heatmap_3d, Grid

from MIREIA.config import Config


class TrafficHandler:
    """
    Deterministic spawner for ego vehicles, traffic vehicles, and pedestrians.
    
    All randomness is seeded so that the same seed produces the exact same
    scenario every time. Keeps track of all spawned actor IDs for clean
    destruction via destroy_all().
    """

    def __init__(self, client: carla.Client, world: carla.World, seed: int = 42, tm_port: int = 8000):
        self.client = client
        self.world = world
        self.seed = seed
        self.tm_port = tm_port

        # Deterministic RNG (Python stdlib) for blueprint/spawn-point selection
        self._rng = stdlib_random.Random(seed)

        # Traffic Manager — deterministic seed
        self.traffic_manager = client.get_trafficmanager(tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_random_device_seed(seed)

        # Match generate_traffic.py: if the world is in synchronous mode,
        # the Traffic Manager must also be set to synchronous mode.
        settings = self.world.get_settings()
        self.synchronous_mode = settings.synchronous_mode
        if self.synchronous_mode:
            self.traffic_manager.set_synchronous_mode(True)

        # Bookkeeping for cleanup
        self.ego_vehicle: carla.Actor = None
        self._vehicle_ids: list[int] = []
        self._walker_ids: list[dict] = []    # [{"id": int, "con": int}, ...]
        self._all_walker_actor_ids: list[int] = []  # interleaved [controller, walker, ...]

        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # -----------------------------------------------------------------
    #  EGO VEHICLE
    # -----------------------------------------------------------------
    def spawn_ego(self, blueprint_id: str = 'vehicle.lincoln.mkz_2020',
                  spawn_index: int = None, autopilot: bool = False) -> carla.Actor:
        """
        Spawn a single ego vehicle marked with role_name='hero'.

        :param blueprint_id: Vehicle blueprint filter string.
        :param spawn_index: Deterministic spawn-point index. If None, chosen
            from the seeded RNG.
        :param autopilot: Whether to enable autopilot on the ego vehicle.
        :returns: The spawned ego carla.Actor.
        """
        bp = self.world.get_blueprint_library().find(blueprint_id)
        bp.set_attribute('role_name', 'hero')

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_index is not None:
            sp = spawn_points[spawn_index % len(spawn_points)]
        else:
            sp = self._rng.choice(spawn_points)

        self.ego_vehicle = self.world.spawn_actor(bp, sp)

        if autopilot:
            self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())

        logging.info("Spawned ego vehicle '%s' at index %s (autopilot=%s)",
                     blueprint_id, spawn_index, autopilot)
        return self.ego_vehicle

    # -----------------------------------------------------------------
    #  TRAFFIC VEHICLES
    # -----------------------------------------------------------------
    def spawn_vehicles(self, n: int = 30, safe: bool = True,
                       car_lights_on: bool = False) -> list[int]:
        """
        Spawn *n* NPC vehicles with autopilot, using batched commands.

        :param n: Number of vehicles to attempt to spawn.
        :param safe: If True, only spawn car-type blueprints (no bikes/trucks).
        :param car_lights_on: Enable automatic headlight management.
        :returns: List of successfully spawned actor IDs.
        """
        blueprints = sorted(
            self.world.get_blueprint_library().filter('vehicle.*'),
            key=lambda bp: bp.id
        )
        if safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        if not blueprints:
            raise ValueError("No vehicle blueprints found with the current filters.")

        spawn_points = list(self.world.get_map().get_spawn_points())
        self._rng.shuffle(spawn_points)

        n = min(n, len(spawn_points))

        batch = []
        for i in range(n):
            bp = self._rng.choice(blueprints)
            if bp.has_attribute('color'):
                bp.set_attribute('color', self._rng.choice(
                    bp.get_attribute('color').recommended_values))
            if bp.has_attribute('driver_id'):
                bp.set_attribute('driver_id', self._rng.choice(
                    bp.get_attribute('driver_id').recommended_values))
            bp.set_attribute('role_name', 'autopilot')
            batch.append(
                SpawnActor(bp, spawn_points[i])
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
            )

        new_ids = []
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                new_ids.append(response.actor_id)

        if car_lights_on:
            for actor in self.world.get_actors(new_ids):
                self.traffic_manager.update_vehicle_lights(actor, True)

        self._vehicle_ids.extend(new_ids)
        logging.info("Spawned %d / %d requested vehicles.", len(new_ids), n)
        return new_ids

    # -----------------------------------------------------------------
    #  PEDESTRIANS
    # -----------------------------------------------------------------
    def spawn_pedestrians(self, n: int = 20,
                          pct_running: float = 0.0,
                          pct_crossing: float = 0.0) -> list[int]:
        """
        Spawn *n* AI-controlled pedestrians on the navigation mesh.
        Closely follows the proven logic from PythonAPI/examples/generate_traffic.py.

        :param n: Number of walkers to attempt to spawn.
        :param pct_running: Fraction [0-1] of walkers that run.
        :param pct_crossing: Fraction [0-1] of walkers allowed to cross roads.
        :returns: List of spawned walker actor IDs (without controllers).
        """
        # --- settings (mirrors generate_traffic.py) ---
        percentagePedestriansRunning = pct_running
        percentagePedestriansCrossing = pct_crossing

        # Seed the pedestrian module for determinism
        self.world.set_pedestrians_seed(self.seed)
        self._rng.seed(self.seed)  # re-seed so walker choices are deterministic

        blueprintsWalkers = sorted(
            self.world.get_blueprint_library().filter('walker.pedestrian.*'),
            key=lambda bp: bp.id
        )
        if not blueprintsWalkers:
            raise ValueError("No pedestrian blueprints found.")

        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(n):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = self._rng.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if self._rng.random() > percentagePedestriansRunning:
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))

        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self._walker_ids.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self._walker_ids)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self._walker_ids[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self._walker_ids[i]["con"] = results[i].actor_id

        # 4. we put together the walkers and controllers id to get the objects from their id
        self._all_walker_actor_ids = []
        for i in range(len(self._walker_ids)):
            self._all_walker_actor_ids.append(self._walker_ids[i]["con"])
            self._all_walker_actor_ids.append(self._walker_ids[i]["id"])
        all_actors = self.world.get_actors(self._all_walker_actor_ids)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not self.synchronous_mode:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self._all_walker_actor_ids), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        walker_only_ids = [e["id"] for e in self._walker_ids]
        logging.info("Spawned %d / %d requested pedestrians.", len(walker_only_ids), n)
        return walker_only_ids

    # -----------------------------------------------------------------
    #  CLEANUP
    # -----------------------------------------------------------------
    def destroy_all(self):
        """Destroy all actors created by this Spawner."""
        # Stop walker controllers first
        if self._all_walker_actor_ids:
            all_actors = self.world.get_actors(self._all_walker_actor_ids)
            for i in range(0, len(self._all_walker_actor_ids), 2):
                try:
                    all_actors[i].stop()
                except Exception:
                    pass

        # Destroy walkers + controllers
        if self._all_walker_actor_ids:
            self.client.apply_batch([DestroyActor(x) for x in self._all_walker_actor_ids])
            logging.info("Destroyed %d walker actors.", len(self._all_walker_actor_ids))

        # Destroy vehicles
        if self._vehicle_ids:
            self.client.apply_batch([DestroyActor(x) for x in self._vehicle_ids])
            logging.info("Destroyed %d vehicles.", len(self._vehicle_ids))

        # Destroy ego
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            logging.info("Destroyed ego vehicle.")

        # Reset bookkeeping
        self.ego_vehicle = None
        self._vehicle_ids.clear()
        self._walker_ids.clear()
        self._all_walker_actor_ids.clear()

class Scenario:
    """
    A Scenario represents a situation in which the ego vehicle will be tested.
    This class does not contain any logic, objects or actors, but simply serves as a container for the information that defines a scenario, such as:
    - Name
    - Definition
    - Map name ...

    It also manages the folder where 
    """    

    def __init__(self, name: str):
        self.name = name
        # Name will be something like "scenario_1", we have a folder like .../scenarios, and we join them to create a new folder .../scenarios/scenario_1 where we will store all the information related to that scenario, such as the pre baked risk map, the routes, etc.
        self.folder_path = os.path.join(Config.PATH_TO_SCENARIOS, self.name)

    def create_scenario(self, name: str, definition: str, map_name: str):
        self.name = name
        self.definition = definition
        self.map_name = map_name

    def load_scenario_from_json(self, json_path: str):
        """
        Load scenario information from a JSON file.
        """
        pass

# A scenario needs, in this order: 
"""
BASIC FEATURES
A name
A definition
A map (town)
Weather conditions
An ego vehicle with a starting position
Traffic density and walkers 

ADVANCED FEATURES
A route
A data recorder 
An instruction of how to record things

"""

class WorldManager:
    """
    Sets up a world and instantiates:
        - ScenarioManager: For spawning traffic, setting weather, and defining routes
        - SimulationBridge: For interfacing with the CARLA simulator and retrieving actor/environment state

    """
    
    def __init__(self, quality_level='High', sync_mode=True, render_offscreen=False, verbose=False):
        self.verbose = verbose
        # Initialized once connected to CARLA
        self.world = None 
        self.blueprints = None
        self.map = None
        self.spawn_points = None
        self.risk_oracle = RiskOracle()
        # Initialize CARLA
        self.__initialize_carla(quality_level, sync_mode, render_offscreen)
        # Initialized once scenario is set up
        self.bridge = None
        self.sensor_manager = None

        
    def __initialize_carla(self, quality_level, sync_mode, render_offscreen):
        
        # Connect to CARLA and set up world
        command = "cd ../carla && ./CarlaUE4.sh -quality-level={} {} {}".format(
            quality_level,
            "--sync" if sync_mode else "",
            "--render-offscreen" if render_offscreen else ""
        )
        subprocess.Popen(command, shell=True)
        if self.verbose:
            print(f"Connecting to CARLA with quality='{quality_level}', sync_mode={sync_mode}, render_offscreen={render_offscreen}...")
            print("Waiting up to 20 seconds for CARLA to initialize before trying to connect...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()

        if self.verbose:
            print("Connected to CARLA. Getting blueprints, map and spawnpoints...")

        self.blueprints = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        if self.verbose:
            print(f"CARLA initialized with map '{self.map.name}' and {len(self.spawn_points)} spawn points.")
            print("Setting up default scenario...")

    def __initialize_scenario(self):
        # if map.name != scenario.map_name => load new map
        # Set weather conditions
        # Spawn ego vehicle
        # Spawn traffic and pedestrians

        if self.verbose:
            print("Initializing SimulationBridge...")
        self.bridge = SimulationBridge(self.world)