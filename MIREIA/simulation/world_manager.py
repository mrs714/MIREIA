import os, subprocess
import carla

from MIREIA.simulation.routes import Route
from MIREIA.simulation.bridge import SimulationBridge
from MIREIA.simulation.sensors import SensorManager
from MIREIA.core.physics import RiskOracle
from MIREIA.analysis.plotter import draw_risk_heatmap_3d, Grid

from MIREIA.config import Config




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