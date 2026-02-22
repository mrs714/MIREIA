# Weights (Mass, Friction values)

GRAVITY = 9.81  # m/s^2
# Environmental Severity
BETA_VIS = 1.0        # Penalty for overdriving visibility
MIN_FRICTION = 0.1    # Clamp for mu
DEFAULT_MU = 0.8     # Default friction coefficient (dry asphalt)

# Dynamic Risk (Gaussian Shape)
REACTION_TIME = 1.5   # Human perception-reaction time (seconds)
AMPLITUDE_GAIN = 1.0  # Mass/Lethality multiplier
MIN_SIGMA_X = 2.0     # Minimum longitudinal spread (meters)
MIN_SIGMA_Y = 1.0     # Minimum lateral spread (meters)
DEFAULT_VISIBILITY = 300.0  # Default visibility in meters
MIN_VISIBILITY = 10.0  # Minimum visibility to avoid divide-by-zero

# Static Risk (Road Boundaries)
LANE_WIDTH_STD = 3.5  # Assumed standard lane width
ROAD_REPULSION = 0.8  # Max risk at lane boundary
ROAD_EXP = 6.0        # "Wall" steepness (higher = harder wall)