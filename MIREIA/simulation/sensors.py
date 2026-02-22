# Spawns Cameras, Lidar, Collision sensors
import carla
import numpy as np

class SensorManager:
    """
    Manages the RGB camera attached to the vehicle, the one on top of the map, and any other ones set along the way.
    """

    def __init__(self, world: carla.World, map: carla.Map, ego_vehicle: carla.Actor, save_dir: str, ego_resolution=(800, 600), map_resolution=(2000, 2000)):
        self.__world = world
        self.__map = map
        self.__ego_vehicle = ego_vehicle
        self.__save_dir = save_dir
        # Initialize camera attributes
        self.__ego_camera = None
        self.__map_camera = None
        self.__other_cameras = []
        self.__map_center = self.__get_map_center()
        self.__setup_ego_camera(ego_resolution)
        self.__setup_map_camera(map_resolution)

    def clean_output_directory(self):
        import os
        import shutil

        if self.__save_dir == "output":
            if os.path.exists(self.__save_dir):
                shutil.rmtree(self.__save_dir)
            os.makedirs(self.__save_dir)
        else:
            print(f"Warning: Save directory is set to '{self.__save_dir}'. Skipping cleanup to avoid accidental data loss.")

    def __get_map_center(self):
        spawn_points = self.__map.get_spawn_points()

        # Extract all x and y coordinates from spawn points
        x_coords = [p.location.x for p in spawn_points]
        y_coords = [p.location.y for p in spawn_points]

        # Calculate the center of the bounding box
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords) + min(y_coords)) / 2
        return carla.Location(x=center_x, y=center_y, z=0)
    
    def __setup_ego_camera(self, ego_resolution):
        blueprint = self.__world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(ego_resolution[0]))
        blueprint.set_attribute('image_size_y', str(ego_resolution[1]))
        camera_init_trans = carla.Transform(carla.Location(z=1.5))
        self.__ego_camera = self.__world.spawn_actor(blueprint, camera_init_trans, attach_to=self.__ego_vehicle)

        self.__ego_camera.enable_postprocess_effects = True

    def __setup_map_camera(self, map_resolution):
            blueprint = self.__world.get_blueprint_library().find('sensor.camera.rgb')
            
            # --- SET ATTRIBUTES ON THE BLUEPRINT FIRST ---
            # Note: attributes are always strings in the blueprint library
            blueprint.set_attribute('image_size_x', str(map_resolution[0]))
            blueprint.set_attribute('image_size_y', str(map_resolution[1]))
            blueprint.set_attribute('bloom_intensity', '0.0') # Better way to kill the 'laser' glow
            blueprint.set_attribute('lens_flare_intensity', '0.0')

            world_center = self.__map_center
            map_location = carla.Location(x=world_center.x, y=world_center.y, z=150)
            map_rotation = carla.Rotation(pitch=-90)
            map_transform = carla.Transform(map_location, map_rotation)
            
            self.__map_camera = self.__world.spawn_actor(blueprint, map_transform)

    def __save_single_snapshot(self, camera: carla.Actor, camera_name: str):
        def callback(image):
            # Starts the camera callback function and then after saving a single image, it stops the camera again
            image_path = f"{self.__save_dir}/{camera_name}_{image.frame}.png"
            image.save_to_disk(image_path)
            camera.stop()
        camera.listen(callback)

    def __save_snapshots(self, camera: carla.Actor, camera_name: str, duration: float = None):
        # Starts the camera callback function and saves images until the simulation ends or the camera is stopped

        start_frame = self.__world.get_snapshot().frame
        
        def callback(image):
            if duration and start_frame + duration < image.frame:
                camera.stop()
                return
            image_path = f"{self.__save_dir}/{camera_name}_{image.frame}.png"
            image.save_to_disk(image_path)
        camera.listen(callback)
    
    def save_ego_frame(self):
        self.__save_single_snapshot(self.__ego_camera, "ego")

    def save_ego_frames(self, duration: float = None):
        self.__save_snapshots(self.__ego_camera, "ego", duration)

    def save_map_frame(self):
        self.__save_single_snapshot(self.__map_camera, "map")

    def save_map_frames(self, duration: float = None):
        self.__save_snapshots(self.__map_camera, "map", duration)