import carla
import math
import numpy as np


class Grid:
    """A numpy-based coordinate grid centered on a point."""
    def __init__(self, center_x, center_y, size=40.0, resolution=2.0):
        self.center_x = center_x
        self.center_y = center_y
        self.size = size
        self.resolution = resolution
        half = size / 2.0
        n = int(size / resolution) + 1
        xs = np.linspace(center_x - half, center_x + half, n)
        ys = np.linspace(center_y - half, center_y + half, n)
        self.X, self.Y = np.meshgrid(xs, ys)


class RiskGrid: 
    def __init__(self, center_x, center_y, size=40.0, resolution=2.0):
        self.center_x = center_x
        self.center_y = center_y
        self.size = size
        self.resolution = resolution
        self.grid = self.generate_grid()
        self.lowest_risk = min(point['risk_value'] for point in self.grid)
        self.highest_risk = max(point['risk_value'] for point in self.grid)

    @classmethod
    def from_grid_and_risk(cls, grid: Grid, risk_values: np.ndarray, z_height=0.5):
        """
        Construct a RiskGrid from a Grid and a 2D numpy array of risk values
        (as returned by RiskOracle.calculate_risk_map).
        """
        instance = cls.__new__(cls)
        instance.center_x = grid.center_x
        instance.center_y = grid.center_y
        instance.size = grid.size
        instance.resolution = grid.resolution
        instance.grid = []
        rows, cols = grid.X.shape
        for i in range(rows):
            for j in range(cols):
                instance.grid.append({
                    'x': float(grid.X[i, j]),
                    'y': float(grid.Y[i, j]),
                    'z': z_height,
                    'risk_value': float(risk_values[i, j]),
                })
        instance.lowest_risk = float(risk_values.min())
        instance.highest_risk = float(risk_values.max())
        return instance

    def generate_grid(self):
        """
        Generates a grid of points centered around (self.center_x, self.center_y).
        size: The total width and length of the grid in meters.
        resolution: The distance between each point in meters.
        """
        grid = []
        half_size = self.size / 2.0
        
        # Calculate the start and end points for our bounds
        start_x = self.center_x - half_size
        end_x = self.center_x + half_size
        start_y = self.center_y - half_size
        end_y = self.center_y + half_size
        
        x = start_x
        while x <= end_x:
            y = start_y
            while y <= end_y:
                # --- Dummy Risk Logic for Testing ---
                # Calculate distance to the center to create a gradient effect
                distance_to_center = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
                
                # Normalize risk from 0.0 to 1.0 (closer to center = higher risk)
                risk = max(0.0, 1.0 - (distance_to_center / half_size))
                
                grid.append({
                    'x': x,
                    'y': y,
                    'z': 0.5, # Assuming relatively flat ground
                    'risk_value': risk
                })
                y += self.resolution
            x += self.resolution
            
        return grid
    
def draw_risk_heatmap_3d(world, risk_grid: RiskGrid, z_height=0.1, tile_size=1.0, life_time=0.1, thickness=0.2):
    """
    Draws a 3D risk heatmap in the CARLA world.
    
    :param world: The CARLA world object.
    :param risk_grid: A RiskGrid object containing the risk values and coordinates.
    :param z_height: The height offset for the heatmap tiles.
    :param tile_size: The size of each heatmap tile.
    :param life_time: The duration each tile is visible in seconds.
    """

    for point in risk_grid.grid:
        risk = point['risk_value']
        # Normalize risk to [0, 1] using min and max
        if risk_grid.highest_risk != risk_grid.lowest_risk:
            norm_risk = (risk - risk_grid.lowest_risk) / (risk_grid.highest_risk - risk_grid.lowest_risk)
        else:
            norm_risk = 1.0  # If all risks are the same, set to max color
        color = carla.Color(r=int(255 * norm_risk), g=int(255 * (1.0 - norm_risk)), b=0)
        
        # Create a Bounding Box (Extent is HALF the total width/length)
        # We make the Z-extent extremely small (0.01) so it acts like a flat 2D plane
        extent = carla.Vector3D(x=tile_size/2, y=tile_size/2, z=0.01)
        
        # Center the box at your coordinates
        center = carla.Location(x=point['x'], y=point['y'], z=point['z'] + z_height)
        box = carla.BoundingBox(center, extent)
        
        # Rotation must be flat to the world
        rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
        
        world.debug.draw_box(
            box=box,
            rotation=rotation,
            thickness=thickness,      # Thickness of the painted lines
            color=color,
            life_time=life_time
        )

