# Weights (Mass, Friction values)

GRAVITY = 9.81  # m/s^2
# Environmental Severity
BETA_VIS = 1.0        # Penalty for overdriving visibility
MIN_FRICTION = 0.1    # Clamp for mu
DEFAULT_MU = 0.8     # Default friction coefficient (dry asphalt)

# Dynamic Risk (Gaussian Shape)
REACTION_TIME = 1.5   # Human perception-reaction time (seconds)
AMPLITUDE_GAIN = 1.0  # Mass/Lethality multiplier
MIN_SIGMA_X = 5     # Minimum longitudinal spread (meters)
MIN_SIGMA_Y = 3     # Minimum lateral spread (meters)
DEFAULT_VISIBILITY = 300.0  # Default visibility in meters
MIN_VISIBILITY = 10.0  # Minimum visibility to avoid divide-by-zero
MAX_DISTANCE = 30.0  # Maximum distance for risk influence (meters)

# Static risk (Static Obstacles)
STATIC_OBSTACLE_DANGER = 10.0  # Base risk for being at the exact location of a static obstacle
STATIC_OBSTACLE_RADIUS = 5.0  # Radius of influence for static obstacles (meters)
STATIC_OBSTACLE_FALLOFF = 3.0  # Controls how quickly risk drops off (higher = sharper falloff)

# Road risk
LANE_WIDTH_STD = 2  # Assumed standard lane width
ROAD_REPULSION =2  # Max risk at lane boundary
ROAD_EXP = 2        # "Wall" steepness (higher = harder wall)

# Base Uncertainties (Added to Gaussian Spread)
BASE_LONGITUDINAL_UNCERTAINTY = 5  # Meters
BASE_LATERAL_UNCERTAINTY = 5  # Meters