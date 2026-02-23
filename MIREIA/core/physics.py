import numpy as np
import math
from MIREIA.core.constants import *
from MIREIA.simulation.bridge import SimulationBridge, EgoKinematics, DynamicObstacleKinematics, EnvironmentState, WaypointStateCollection
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
            
            # Static Risk (Road Boundaries)
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

    def calculate_scene_risk(self, query_point, bridge: SimulationBridge) -> float:
        """Calculates risk at a specific (x,y) point."""
        ego_kinematics: EgoKinematics = bridge.get_ego_kinematics()
        obstacles: list[DynamicObstacleKinematics] = bridge.get_obstacles()
        env_state: EnvironmentState = bridge.get_environment_state()
        road_data: WaypointStateCollection = bridge.get_waypoints()

        mu = max(env_state.mu, self.params['min_friction'])
        vis = max(env_state.visibility, MIN_VISIBILITY)
        v_ego = ego_kinematics.v
        
        # 1. Environmental Scalar
        phi_env = (1.0 / mu) * (1.0 + self.params['beta_vis'] * (v_ego / vis))

        # 2. Dynamic Risk
        risk_dynamic = 0.0
        for obj in obstacles:
            risk_dynamic += self._compute_gaussian_at_point(query_point, obj, ego_kinematics, mu)

        # 3. Static Risk – find the nearest waypoint to the query point
        risk_static = 0.0
        if road_data and road_data.waypoints:
            px, py = query_point
            best_dist_sq = float('inf')
            best_half_w = 1.0
            for waypoint in road_data.waypoints:
                d_sq = (waypoint.x - px)**2 + (waypoint.y - py)**2
                if d_sq < best_dist_sq:
                    best_dist_sq = d_sq
                    best_half_w = waypoint.width / 2.0
            nearest_dist = math.sqrt(best_dist_sq)
            norm_dist = min(nearest_dist / best_half_w, 1.2)
            risk_static = self.params['road_repulsion'] * (norm_dist ** self.params['road_exp'])

        return phi_env * (risk_static + risk_dynamic)

    def _compute_gaussian_at_point(self, point: tuple[float, float], obj: DynamicObstacleKinematics, ego_kinematics: EgoKinematics, mu: float):
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
            
        return self.params['amplitude_gain'] * math.exp(exponent)

    # =========================================================
    #  B. VECTORIZED METHODS (For Visualization / Heatmaps)
    # =========================================================

    def calculate_risk_map(self, grid: Grid, bridge: SimulationBridge) -> RiskGrid:
        """
        Vectorized calculation for generating 2D heatmaps efficiently.
        Called by the Visualizer.
        
        :param grid: A Grid object containing numpy meshgrids of coordinates.
        :param bridge: SimulationBridge providing ego, obstacles, environment and road data.
        """
        X, Y = grid.X, grid.Y
        ego_kinematics: EgoKinematics = bridge.get_ego_kinematics()
        obstacles: list[DynamicObstacleKinematics] = bridge.get_obstacles()
        env_state: EnvironmentState = bridge.get_environment_state()
        road_data: WaypointStateCollection = bridge.get_waypoints()

        # 1. Environment Scalar
        mu = max(env_state.mu, self.params['min_friction'])
        vis = max(env_state.visibility, MIN_VISIBILITY)
        phi = (1.0 / mu) * (1.0 + self.params['beta_vis'] * (ego_kinematics.v / vis))
        
        # 2. Dynamic Risk (Vectorized over X, Y)
        risk_dyn = np.zeros_like(X)
        half = grid.size / 2.0
        skip_margin = 70.0  # skip obstacles more than 70m from grid edge
        
        for obj in obstacles:
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
            risk_dyn += self.params['amplitude_gain'] * np.exp(exponent)
            
        # 3. Static Risk (brute-force nearest-waypoint lookup)
        risk_stat = np.zeros_like(X)
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
                    risk_stat[i, j] = self.params['road_repulsion'] * (norm_dist ** self.params['road_exp'])

        risk_values = phi * (risk_stat + risk_dyn)
        return RiskGrid.from_grid_and_risk(grid, risk_values)