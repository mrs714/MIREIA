# Weights (Mass, Friction values)

GRAVITY = 9.81  # m/s^2
# Kinetic Severity
KINETIC_LOG_GAIN = 0.3  # kappa: logarithmic speed scaling for risk
# Environmental Severity
BETA_VIS = 1.0        # Penalty for overdriving visibility
MIN_FRICTION = 0.1    # Clamp for mu
DEFAULT_MU = 0.8     # Default friction coefficient (dry asphalt)

# Dynamic Risk (Gaussian Shape)
REACTION_TIME = 1.5   # Human perception-reaction time (seconds)
AMPLITUDE_GAIN = 1.0  # Mass/Lethality multiplier
PEDESTRIAN_AMPLITUDE_GAIN = 2.0  # Pedestrians are vulnerable road users — higher consequence
MIN_SIGMA_X = 5     # Minimum longitudinal spread (meters)
MIN_SIGMA_Y = 3     # Minimum lateral spread (meters)
DEFAULT_VISIBILITY = 300.0  # Default visibility in meters
MIN_VISIBILITY = 10.0  # Minimum visibility to avoid divide-by-zero
MAX_DISTANCE = 30.0  # Maximum distance for risk influence (meters)

# Static risk (Static Obstacles): this should match with the ones defined in bridge.py
STATIC_OBSTACLE_DICT = {
    'TrafficLight': {
        'danger': 0.25,
        'radius': 20.0,
        'falloff': 2
    },
    'ParkedVehicle': {
        'danger': 0.25,
        'radius': 20.0,
        'falloff': 2
    },
    'Crosswalk': {
        'danger': 0.25,
        'radius': 20.0,
        'falloff': 2
    },
}

# Road risk
LANE_WIDTH_STD = 3  # Assumed standard lane width
ROAD_REPULSION =2  # Max risk at lane boundary
ROAD_EXP = 2        # "Wall" steepness (higher = harder wall)

# Base Uncertainties (Added to Gaussian Spread)
BASE_LONGITUDINAL_UNCERTAINTY = 5  # Meters
BASE_LATERAL_UNCERTAINTY = 5  # Meters