import numpy as np
import math
from MIREIA.core.constants import *
# PEDESTRIAN_AMPLITUDE_GAIN imported via wildcard
from MIREIA.simulation.bridge import SimulationBridge, EgoKinematics, DynamicObstacleKinematics, EnvironmentState, WaypointStateCollection, StaticObstacleState
from MIREIA.analysis.plotter import Grid, RiskGrid

class RiskOracle:
    """
    The Mathematical Core of the Driving Risk Field (DRF).
    Handles both single-point queries (for planning/labeling) 
    and vectorized grid queries (for visualization).
    """

    def __init__(self, config=None):
        # --- PHYSICAL CONSTANTS ---
        self.G = GRAVITY  
        
        # --- CALIBRATION PARAMETERS ---
        self.params = {
            # Environmental Severity
            'beta_vis': BETA_VIS,        # Penalty for overdriving visibility
            'min_friction': MIN_FRICTION,    # Clamp for mu
            
            # Dynamic Risk (Gaussian Shape)
            'reaction_time': REACTION_TIME,   # Human perception-reaction time (seconds)
            'amplitude_gain': AMPLITUDE_GAIN,  # Mass/Lethality multiplier
            'min_sigma_x': MIN_SIGMA_X,     # Minimum longitudinal spread (meters)
            'min_sigma_y': MIN_SIGMA_Y,     # Minimum lateral spread (meters)
            'max_distance': MAX_DISTANCE,   # Max distance for risk influence (meters)

            # Static Risk (Static Obstacles)
            'static_obstacle_dict': STATIC_OBSTACLE_DICT,  # Parameters for different static obstacle types
            
            # Pedestrian Risk
            'pedestrian_amplitude_gain': PEDESTRIAN_AMPLITUDE_GAIN,  # Higher consequence for vulnerable road users

            # Road Risk (Road Boundaries)
            'lane_width_std': LANE_WIDTH_STD,  # Assumed standard lane width
            'road_repulsion': ROAD_REPULSION,  # Max risk at lane boundary
            'road_exp': ROAD_EXP,        # "Wall" steepness (higher = harder wall)

            # Base Uncertainties (Added to Gaussian Spread)
            'base_longitudinal_uncertainty': BASE_LONGITUDINAL_UNCERTAINTY,  # Meters
            'base_lateral_uncertainty': BASE_LATERAL_UNCERTAINTY,  # Meters
        }
        
        if config:
            self.params.update(config)

    # =========================================================
    #  A. SINGLE POINT METHODS (For Data Gen / Path Planning)
    # =========================================================

    def calculate_scene_risk(self, query_point: tuple[float, float], bridge: SimulationBridge, baked_static_risk: RiskGrid) -> float:
        """Calculates risk at a specific (x,y) point.
        
        :param baked_static_risk: Optional pre-baked static risk grid. If provided,
            road and static obstacle risk is sampled via bilinear interpolation instead
            of being computed from scratch.
        """
        ego_kinematics: EgoKinematics = bridge.get_ego_kinematics()
        dynamic_obstacles: list[DynamicObstacleKinematics] = bridge.get_dynamic_obstacles()
        env_state: EnvironmentState = bridge.get_environment_state()

        mu = max(env_state.mu, self.params['min_friction'])
        vis = max(env_state.visibility, MIN_VISIBILITY)
        v_ego = ego_kinematics.v
        
        # 1. Environmental Scalar
        phi_env = (1.0 / mu) * (1.0 + self.params['beta_vis'] * (v_ego / vis))

        # 2. Dynamic Risk (vehicles)
        risk_dynamic = 0.0
        for obj in dynamic_obstacles:
            risk_dynamic += self._compute_gaussian_at_point(query_point, obj, ego_kinematics, mu)

        # 2b. Pedestrian Risk (vulnerable road users)
        pedestrians: list[DynamicObstacleKinematics] = bridge.get_pedestrians()
        for ped in pedestrians:
            risk_dynamic += self._compute_gaussian_at_point(
                query_point, ped, ego_kinematics, mu,
                amplitude_override=self.params['pedestrian_amplitude_gain']
            )

        # 3. Static Risk (road + static obstacles)
        risk_static_total = self._sample_baked_point(baked_static_risk, query_point[0], query_point[1])
        
        return phi_env * (risk_static_total + risk_dynamic)

    def _compute_gaussian_at_point(self, point: tuple[float, float], obj, ego_kinematics: EgoKinematics, mu: float, amplitude_override: float = None):
        """Helper for single-point Gaussian math."""
        px, py = point
        ox, oy = obj.x, obj.y
        dx, dy = px - ox, py - oy
        
        if (dx*dx + dy*dy) > self.params["max_distance"] ** 2: return 0.0 # Optimization for points >N m away

        # Relative velocity
        vx_rel = obj.vx - ego_kinematics.vx
        vy_rel = obj.vy - ego_kinematics.vy
        v_rel_mag = math.sqrt(vx_rel**2 + vy_rel**2)

        # Rotation
        yaw_rad = math.radians(obj.heading)
        cos_yaw, sin_yaw = math.cos(-yaw_rad), math.sin(-yaw_rad)
        x_local = dx * cos_yaw - dy * sin_yaw
        y_local = dx * sin_yaw + dy * cos_yaw

        # Shape
        sigma_x = (obj.length/2.0) + (self.params['reaction_time'] * v_rel_mag) + ((v_rel_mag**2) / (2 * mu * self.G)) + self.params['base_longitudinal_uncertainty'] # Base longitudinal uncertainty
        sigma_x = max(sigma_x, self.params['min_sigma_x'])
        
        sigma_y = (obj.width/2.0) + self.params["base_lateral_uncertainty"] # Base lateral uncertainty
        sigma_y = max(sigma_y, self.params['min_sigma_y']) # Ensure a minimum spread even for small obstacles

        exponent = -0.5 * ((x_local/sigma_x)**2 + (y_local/sigma_y)**2) # Gaussian formula; higher exponent = closer to center = more risk
        if exponent < -20: return 0.0

        amp = amplitude_override if amplitude_override is not None else self.params['amplitude_gain']
        return amp * math.exp(exponent)

    # =========================================================
    #  B. BAKED STATIC RISK MAP
    # =========================================================

    def bake_static_risk(self, grid: Grid, bridge: SimulationBridge) -> RiskGrid:
        """
        Pre-computes the static (time-invariant) risk layer: road boundaries + static obstacles.
        This only needs to be called once for a given grid region, since roads and
        static obstacles don't move. The result can be cached and reused across frames.

        :param grid: A Grid defining the area and resolution to bake.
        :param bridge: SimulationBridge providing waypoint and static obstacle data.
        :returns: A RiskGrid containing only the static risk components.
        """
        X, Y = grid.X, grid.Y
        road_data: WaypointStateCollection = bridge.get_waypoints()
        static_obstacles: list[StaticObstacleState] = bridge.get_static_obstacles()

        # 1. Road Risk
        risk_road = np.zeros_like(X)
        if road_data and road_data.waypoints:
            wp_x = np.array([wp.x for wp in road_data.waypoints])
            wp_y = np.array([wp.y for wp in road_data.waypoints])
            wp_half_w = np.array([wp.width / 2.0 for wp in road_data.waypoints])

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    dx = wp_x - X[i, j]
                    dy = wp_y - Y[i, j]
                    dist_sq = dx*dx + dy*dy
                    idx = np.argmin(dist_sq)
                    nearest_dist = np.sqrt(dist_sq[idx])
                    norm_dist = min(nearest_dist / wp_half_w[idx], 1.2)
                    risk_road[i, j] = self.params['road_repulsion'] * (norm_dist ** self.params['road_exp'])

        # 2. Static Obstacles Risk (radial Gaussian falloff)
        risk_static = np.zeros_like(X)
        for obj in static_obstacles:
            dx = X - obj.x
            dy = Y - obj.y
            dist_sq = dx*dx + dy*dy

            params = self.params['static_obstacle_dict'][obj.type]

            radius = params['radius']
            sigma_static = radius / params['falloff']

            mask = dist_sq < radius ** 2
            risk_static[mask] += params['danger'] * np.exp(-0.5 * (dist_sq[mask] / sigma_static ** 2))

        risk_values = risk_road + risk_static
        return RiskGrid.from_grid_and_risk(grid, risk_values)

    @staticmethod
    def _sample_baked_point(baked: RiskGrid, px: float, py: float) -> float:
        """
        Bilinear interpolation of a single (px, py) point from a baked RiskGrid.
        Returns 0.0 if the point is outside the baked grid.
        """
        half = baked.size / 2.0
        # Convert world coords to continuous grid indices
        fx = (px - (baked.center_x - half)) / baked.resolution
        fy = (py - (baked.center_y - half)) / baked.resolution
        n = int(baked.size / baked.resolution) + 1

        if fx < 0 or fy < 0 or fx >= n - 1 or fy >= n - 1:
            return 0.0

        ix, iy = int(fx), int(fy)
        dx, dy = fx - ix, fy - iy

        # grid is stored row-major: index = row * n + col  (row=iy, col=ix)
        v00 = baked.grid[iy * n + ix]['risk_value']
        v10 = baked.grid[iy * n + ix + 1]['risk_value']
        v01 = baked.grid[(iy + 1) * n + ix]['risk_value']
        v11 = baked.grid[(iy + 1) * n + ix + 1]['risk_value']

        return (v00 * (1 - dx) * (1 - dy) +
                v10 * dx * (1 - dy) +
                v01 * (1 - dx) * dy +
                v11 * dx * dy)

    @staticmethod
    def _sample_baked_grid(baked: RiskGrid, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorized bilinear interpolation of an entire meshgrid from a baked RiskGrid.
        Points outside the baked region get 0.0.
        """
        half = baked.size / 2.0
        n = int(baked.size / baked.resolution) + 1

        # Reconstruct the 2D risk array from the flat grid list
        baked_array = np.array([p['risk_value'] for p in baked.grid]).reshape(n, n)

        # Convert world coords to continuous grid indices
        fx = (X - (baked.center_x - half)) / baked.resolution
        fy = (Y - (baked.center_y - half)) / baked.resolution

        # Clamp to valid interpolation range [0, n-2] for floor indices
        ix = np.clip(np.floor(fx).astype(int), 0, n - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, n - 2)
        dx = np.clip(fx - ix, 0.0, 1.0)
        dy = np.clip(fy - iy, 0.0, 1.0)

        # Bilinear interpolation
        result = (baked_array[iy, ix] * (1 - dx) * (1 - dy) +
                  baked_array[iy, ix + 1] * dx * (1 - dy) +
                  baked_array[iy + 1, ix] * (1 - dx) * dy +
                  baked_array[iy + 1, ix + 1] * dx * dy)

        # Zero out points outside the baked region
        out_of_bounds = (fx < 0) | (fy < 0) | (fx >= n - 1) | (fy >= n - 1)
        result[out_of_bounds] = 0.0

        return result

    # =========================================================
    #  C. VECTORIZED METHODS (For Visualization / Heatmaps)
    # =========================================================

    def calculate_risk_map(self, grid: Grid, bridge: SimulationBridge,
                           baked_static_risk: RiskGrid = None) -> RiskGrid:
        """
        Vectorized calculation for generating 2D heatmaps efficiently.
        Called by the Visualizer.
        
        :param grid: A Grid object containing numpy meshgrids of coordinates.
        :param bridge: SimulationBridge providing ego, obstacles, environment and road data.
        :param baked_static_risk: Optional pre-baked static risk grid. If provided,
            road and static obstacle risk is sampled via bilinear interpolation instead
            of being computed from scratch (much faster).
        """
        X, Y = grid.X, grid.Y
        ego_kinematics: EgoKinematics = bridge.get_ego_kinematics()
        dynamic_obstacles: list[DynamicObstacleKinematics] = bridge.get_dynamic_obstacles()
        env_state: EnvironmentState = bridge.get_environment_state()

        # 1. Environment Scalar
        mu = max(env_state.mu, self.params['min_friction'])
        vis = max(env_state.visibility, MIN_VISIBILITY)
        phi = (1.0 / mu) * (1.0 + self.params['beta_vis'] * (ego_kinematics.v / vis))
        
        # 2. Dynamic Risk (Vectorized over X, Y) — vehicles + pedestrians
        risk_dyn = np.zeros_like(X)
        half = grid.size / 2.0
        skip_margin = 70.0  # skip obstacles more than 70m from grid edge

        pedestrians = bridge.get_pedestrians()
        all_dynamic = [(obj, self.params['amplitude_gain']) for obj in dynamic_obstacles] + \
                      [(ped, self.params['pedestrian_amplitude_gain']) for ped in pedestrians]
        
        for obj, amp_gain in all_dynamic:
            # Skip obstacles far from grid bounds
            if (abs(obj.x - grid.center_x) > half + skip_margin or
                abs(obj.y - grid.center_y) > half + skip_margin):
                continue

            # Relative Pos
            dx = X - obj.x
            dy = Y - obj.y
            
            # Relative Vel
            ego_heading_rad = math.radians(ego_kinematics.heading)
            vx_rel = obj.vx - ego_kinematics.v * math.cos(ego_heading_rad)
            vy_rel = obj.vy - ego_kinematics.v * math.sin(ego_heading_rad)
            v_rel_mag = np.sqrt(vx_rel**2 + vy_rel**2)
            
            # Rotation
            angle = -math.radians(obj.heading)
            x_rot = dx * math.cos(angle) - dy * math.sin(angle)
            y_rot = dx * math.sin(angle) + dy * math.cos(angle)
            
            # Sigma Calculation (Same physics as single-point)
            sigma_x = (obj.length/2.0) + (self.params['reaction_time'] * v_rel_mag) + (v_rel_mag**2 / (2*mu*self.G))
            sigma_x = np.maximum(sigma_x, self.params['min_sigma_x'])
            
            sigma_y = (obj.width/2.0) + 0.5
            sigma_y = np.maximum(sigma_y, self.params['min_sigma_y'])
            
            # Gaussian
            exponent = -0.5 * ( (x_rot**2 / sigma_x**2) + (y_rot**2 / sigma_y**2) )
            risk_dyn += amp_gain * np.exp(exponent)
            
        # 3. Static Risk (road + static obstacles)
        risk_static_total = self._sample_baked_grid(baked_static_risk, X, Y)
        
        risk_values = phi * (risk_static_total + risk_dyn)
        return RiskGrid.from_grid_and_risk(grid, risk_values)