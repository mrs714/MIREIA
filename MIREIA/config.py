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