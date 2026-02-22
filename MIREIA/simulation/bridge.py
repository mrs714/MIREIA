import carla 

class ActorKinematics:
    """
    Given an actor, this class provides methods to retrieve its kinematic state (position, velocity, acceleration) in a structured way.
    """
    def __init__(self, actor: carla.Actor):
        self.actor = actor
        self.update()
    
    def update(self):
        location = self.actor.get_location()
        self.x = location.x
        self.y = location.y
        velocity = self.actor.get_velocity()
        self.v = (velocity.x ** 2 + velocity.y ** 2) ** 0.5
        self.vx = velocity.x
        self.vy = velocity.y
        transform = self.actor.get_transform()
        self.heading = transform.rotation.yaw
        bounding_box = self.actor.bounding_box
        self.length = bounding_box.extent.x * 2
        self.width = bounding_box.extent.y * 2

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
        self.update()
        self.visibility = 300.0  # Default visibility in meters
        self.mu = 0.8  # Default friction coefficient (dry asphalt)
    
    def update(self):
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
    Extracted from a waypoint.
    """
    def __init__(self, waypoint: carla.Waypoint):
        self.waypoint = waypoint
        self.update()
    
    def update(self):
        self.width = self.waypoint.lane_width
        transform = self.waypoint.transform
        self.x = transform.location.x
        self.y = transform.location.y
        self.heading = transform.rotation.yaw

class WaypointStateCollection:
    """
    All the waypoints in the map, with additional context, like closest waypoint to ego.
    """
    def __init__(self, ego: EgoKinematics):
        self.ego = ego
        self.waypoints: list[WaypointState] = []
        self.closest_waypoint: WaypointState = None
        self.update()

    def update(self):
        """
        Updatse all the waypoints stored, and finds the closest one to the ego.
        """
        for waypoint in self.waypoints:
            waypoint.update()

        if self.waypoints:
            self.closest_waypoint = min(self.waypoints, key=lambda wp: ((wp.x - self.ego.x) ** 2 + (wp.y - self.ego.y) ** 2) ** 0.5)

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

    def update(self):
        if self.ego:
            self.ego.update()
        for obstacle in self.dynamic_obstacles:
            obstacle.update()
        for pedestrian in self.pedestrians:
            pedestrian.update()
        if self.static_obstacles:
            self.static_obstacles.update()
        self.env_state.update()

    def __repr__(self):
        return f"SimulationBridge(Ego: {self.ego}, Dynamic Obstacles: {len(self.dynamic_obstacles)}, Static Obstacles: {len(self.static_obstacles.waypoints) if self.static_obstacles else 0}, EnvState: (Visibility: {self.env_state.visibility:.1f}m, Friction: {self.env_state.mu:.2f}))"