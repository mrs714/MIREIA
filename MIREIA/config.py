import os
from dotenv import load_dotenv

# Load the .env file as soon as this module is imported
load_dotenv()

class Config:
    # CARLA Connection
    CARLA_HOST: str = os.getenv('CARLA_HOST', '127.0.0.1')
    CARLA_PORT: int = int(os.getenv('CARLA_PORT', 2000))
    
    # Paths
    _MIREIA_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH_TO_SCENARIOS: str = os.getenv(
        'PATH_TO_SCENARIOS',
        os.path.join(_MIREIA_DIR, 'scenarios'),
    )
    PATH_TO_TRIALS: str = os.getenv(
        'PATH_TO_TRIALS',
        os.path.join(_MIREIA_DIR, 'trials'),
    )
    PATH_TO_MODELS: str = os.getenv(
        'PATH_TO_MODELS',
        os.path.join(_MIREIA_DIR, 'models'),
    )

    # Reproducibility
    RANDOM_SEED: int = int(os.getenv('MIREIA_RANDOM_SEED', 42))

    # Simulation and recording cadence defaults
    SIM_FIXED_DELTA_SECONDS: float = float(os.getenv('MIREIA_SIM_FIXED_DELTA_SECONDS', 0.05))
    RECORD_EVERY_N_TICKS: int = int(os.getenv('MIREIA_RECORD_EVERY_N_TICKS', 5))
    RECORDING_FPS: int = int(os.getenv('MIREIA_RECORDING_FPS', 4))

    # Online temporal inference defaults
    INFERENCE_SEQUENCE_LENGTH: int = int(os.getenv('MIREIA_INFERENCE_SEQUENCE_LENGTH', 16))
    INFERENCE_BURN_IN_FRAMES: int = int(os.getenv('MIREIA_INFERENCE_BURN_IN_FRAMES', 12))
    INFERENCE_EVAL_FRAMES: int = int(os.getenv('MIREIA_INFERENCE_EVAL_FRAMES', 4))