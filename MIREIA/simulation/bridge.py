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
    
class StaticObstacleState:
    """
    Represents static obstacles in the environment, which while inherently not dangerous, signal possible danger by obscuring visibility, or representing dangerous
    places, such as parked cars, or semaphores. For simplicity, we represent all static obstacles as rectangles with a position and width/length (e.g., lane boundaries, parked cars). 
    """
    def __init__(self, x: float, y: float, width: float, length: float):
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        
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

class WaypointStateCollection:
    """
    All the waypoints in the map, with additional context, like closest waypoint to ego.
    Road geometry is static; numpy arrays are pre-built once for fast lookups.
    """
    def __init__(self):
        self.waypoints: list[WaypointState] = []

    def add_waypoint(self, waypoint: WaypointState):
        self.waypoints.append(waypoint)

    def get_closest_waypoint(self, x: float, y: float) -> WaypointState:
        """Get closest waypoint to a given (x,y) point."""
        if not self.waypoints:
            return None
        closest_wp = min(self.waypoints, key=lambda wp: (wp.x - x) ** 2 + (wp.y - y) ** 2)
        return closest_wp

    def __repr__(self):
        closest_wp = self.get_closest_waypoint(0, 0)  # Replace (0, 0) with actual ego position if available
        return f"WaypointStateCollection(Total Waypoints: {len(self.waypoints)}, Closest Waypoint: (x={closest_wp.x:.2f}, y={closest_wp.y:.2f}, heading={closest_wp.heading:.1f}))"

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
        self.static_obstacles: list[StaticObstacleState] = []
        self.waypoints: WaypointStateCollection = None
        self.env_state = EnvironmentState(world)

        # Initialize dynamic obstacles and ego
        all_actors = self.world.get_actors()
        for actor in all_actors:
            # The following are dynamic and will need to be updated:
            if 'vehicle' in actor.type_id:
                if actor.attributes.get('role_name') == 'hero':
                    self._set_ego(actor)
                else:
                    self.dynamic_obstacles.append(DynamicObstacleKinematics(actor))
            elif 'walker.pedestrian' in actor.type_id:
                self.pedestrians.append(PedestrianKinematics(actor))

        # The following are static and only need to be intialized once:
        traffic_signs = self.world.get_environment_objects(object_type=carla.CityObjectLabel.TrafficLight)
        # Traffic light is a subtype of traffic sign, so it should be included.
        # However, traffic signs includes light poles and such which we don't want. 
        for sign in traffic_signs:
            # For simplicity, treat traffic signs as static obstacles with a fixed size
            transform = sign.transform
            self.static_obstacles.append(StaticObstacleState(transform.location.x, transform.location.y, width=1.0, length=1.0))
        


        # Initialize roads and waypoints
        self.waypoints = WaypointStateCollection()
        for waypoint in self.map.generate_waypoints(2.0): # Generates waypoints in each lane every 2 meters and returns all of them
            self.waypoints.add_waypoint(WaypointState(waypoint))

        self.update()

    def _set_ego(self, ego_actor: carla.Actor):
        self.ego = EgoKinematics(ego_actor)

    # Getters
    def get_ego_kinematics(self) -> EgoKinematics:
        return self.ego
    
    def get_dynamic_obstacles(self) -> list[DynamicObstacleKinematics]:
        return self.dynamic_obstacles

    def get_pedestrians(self) -> list[PedestrianKinematics]:
        return self.pedestrians

    def get_environment_state(self) -> EnvironmentState:
        return self.env_state
    
    def get_static_obstacles(self) -> list[StaticObstacleState]:
        return self.static_obstacles

    def get_waypoints(self) -> WaypointStateCollection:
        return self.waypoints

    def update(self, update_ego=True, update_dynamic_obstacles=True, update_pedestrians=True, update_weather=True):
        if self.ego and update_ego:
            self.ego.update()
        if update_dynamic_obstacles:
            for obstacle in self.dynamic_obstacles:
                obstacle.update()
        if update_pedestrians:
            for pedestrian in self.pedestrians:
                pedestrian.update()
        if update_weather:
            self.env_state.update()

    def __repr__(self):
        return f"SimulationBridge(Ego: {self.ego}, Dynamic Obstacles: {len(self.dynamic_obstacles)}, Static Obstacles: {len(self.waypoints.waypoints) if self.waypoints else 0}, EnvState: (Visibility: {self.env_state.visibility:.1f}m, Friction: {self.env_state.mu:.2f}))"