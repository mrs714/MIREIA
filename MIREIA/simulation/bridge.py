import carla 
import numpy as np

class ActorKinematics:
    """
    Given an actor, this class provides methods to retrieve its kinematic state (position, velocity, acceleration) in a structured way.
    """
    def __init__(self, actor: carla.Actor):
        self.actor = actor
        # Bounding box is static — cache once
        bounding_box = actor.bounding_box
        self.length = bounding_box.extent.x * 2
        self.width = bounding_box.extent.y * 2
        self.update()
    
    def update(self):
        transform = self.actor.get_transform()
        self.x = transform.location.x
        self.y = transform.location.y
        self.heading = transform.rotation.yaw
        velocity = self.actor.get_velocity()
        self.v = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
        self.vx = velocity.x
        self.vy = velocity.y

    def __repr__(self):
        return f"ID: {self.actor.id}, ActorKinematics(x={self.x:.2f}, y={self.y:.2f}, v={self.v:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f}, heading={self.heading:.1f})"

class EgoKinematics(ActorKinematics):
    def __init__(self, actor):
        super().__init__(actor)

class DynamicObstacleKinematics(ActorKinematics):
    def __init__(self, actor):
        super().__init__(actor)

class PedestrianKinematics(ActorKinematics):
    def __init__(self, actor):
        super().__init__(actor)

class EnvironmentState:
    """
    Represents the environmental state in the simulation, including visibility and friction.
    """
    def __init__(self, world: carla.World):
        self.world = world
        self.visibility = 300.0  # Default visibility in meters
        self.mu = 0.8  # Default friction coefficient (dry asphalt)
        self._frame_counter = 0
        self.update()
    
    def update(self):
        # Weather changes rarely — only query every 30 frames
        self._frame_counter += 1
        if self._frame_counter % 30 != 1:
            return

        weather: carla.WeatherParameters = self.world.get_weather()

        # All values 0 to 100 
        cloudiness = weather.cloudiness
        precipitation = weather.precipitation
        fog_density = weather.fog_density
        wetness = weather.wetness

        self.visibility = max(10.0, 300.0 - (cloudiness * 2.0 + precipitation * 3.0 + fog_density * 4.0))
        self.mu = max(0.1, 0.8 - (precipitation * 0.003 + wetness * 0.002))

    def __repr__(self):
        return f"EnvironmentState(Visibility: {self.visibility:.1f}m, Friction: {self.mu:.2f})"
    
class WaypointState:
    """
    Represents the state of the road, including lane information and proximity to lane centers.
    Extracted from a waypoint.  Road geometry is static so values are cached at init.
    """
    def __init__(self, waypoint: carla.Waypoint):
        self.width = waypoint.lane_width
        transform = waypoint.transform
        self.x = transform.location.x
        self.y = transform.location.y
        self.heading = transform.rotation.yaw

class WaypointStateCollection:
    """
    All the waypoints in the map, with additional context, like closest waypoint to ego.
    Road geometry is static; numpy arrays are pre-built once for fast lookups.
    """
    def __init__(self, ego: EgoKinematics):
        self.ego = ego
        self.waypoints: list[WaypointState] = []
        self.closest_waypoint: WaypointState = None
        # Pre-built numpy arrays (populated by _build_arrays)
        self._wp_x: np.ndarray = None
        self._wp_y: np.ndarray = None
        self._wp_half_w: np.ndarray = None
        self._kd_tree = None

    def _build_arrays(self):
        """Call once after all waypoints have been added."""
        from scipy.spatial import cKDTree
        self._wp_x = np.array([wp.x for wp in self.waypoints])
        self._wp_y = np.array([wp.y for wp in self.waypoints])
        self._wp_half_w = np.array([wp.width / 2.0 for wp in self.waypoints])
        coords = np.column_stack((self._wp_x, self._wp_y))
        self._kd_tree = cKDTree(coords)

    def update(self):
        """
        Only recomputes the closest waypoint to the ego (fast numpy op).
        """
        if self._wp_x is None or len(self.waypoints) == 0:
            return
        dx = self._wp_x - self.ego.x
        dy = self._wp_y - self.ego.y
        idx = np.argmin(dx*dx + dy*dy)
        self.closest_waypoint = self.waypoints[idx]

    def __repr__(self):
        return f"WaypointStateCollection(Total Waypoints: {len(self.waypoints)}, Closest Waypoint: (x={self.closest_waypoint.x:.2f}, y={self.closest_waypoint.y:.2f}, heading={self.closest_waypoint.heading:.1f}))"

class SimulationBridge:
    """
    A bridge class to interface with the CARLA simulator, providing methods to retrieve actor kinematics and environmental state.
    """
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()
        self.ego: EgoKinematics = None
        self.dynamic_obstacles: list[DynamicObstacleKinematics] = []
        self.pedestrians: list[PedestrianKinematics] = []
        self.static_obstacles: WaypointStateCollection = None
        self.env_state = EnvironmentState(world)

        # Initialize dynamic obstacles and ego
        all_actors = self.world.get_actors()
        for actor in all_actors:
            if 'vehicle' in actor.type_id:
                if actor.attributes.get('role_name') == 'hero':
                    self._set_ego(actor)
                else:
                    self.dynamic_obstacles.append(DynamicObstacleKinematics(actor))
            elif 'walker.pedestrian' in actor.type_id:
                self.pedestrians.append(PedestrianKinematics(actor))

        # Initialize static obstacles (roads)
        self.static_obstacles = WaypointStateCollection(self.ego)

        for waypoint in self.map.generate_waypoints(2.0): # Generates waypoints in each lane every 2 meters and returns all of them
            self.static_obstacles.waypoints.append(WaypointState(waypoint))

        self.static_obstacles._build_arrays()  # Pre-build numpy arrays + KD-tree
        self.update()

    def _set_ego(self, ego_actor: carla.Actor):
        self.ego = EgoKinematics(ego_actor)

    # Getters
    def get_ego_kinematics(self) -> EgoKinematics:
        return self.ego
    
    def get_obstacles(self) -> list[DynamicObstacleKinematics]:
        return self.dynamic_obstacles

    def get_pedestrians(self) -> list[PedestrianKinematics]:
        return self.pedestrians

    def get_environment_state(self) -> EnvironmentState:
        return self.env_state

    def get_static_obstacles(self) -> WaypointStateCollection:
        return self.static_obstacles

    def update(self, update_ego=True, update_dynamic_obstacles=True, update_pedestrians=True, update_static_obstacles=True, update_weather=True):
        if self.ego and update_ego:
            self.ego.update()
        if update_dynamic_obstacles:
            for obstacle in self.dynamic_obstacles:
                obstacle.update()
        if update_pedestrians:
            for pedestrian in self.pedestrians:
                pedestrian.update()
        if self.static_obstacles and update_static_obstacles:
            self.static_obstacles.update()
        if update_weather:
            self.env_state.update()

    def __repr__(self):
        return f"SimulationBridge(Ego: {self.ego}, Dynamic Obstacles: {len(self.dynamic_obstacles)}, Static Obstacles: {len(self.static_obstacles.waypoints) if self.static_obstacles else 0}, EnvState: (Visibility: {self.env_state.visibility:.1f}m, Friction: {self.env_state.mu:.2f}))"