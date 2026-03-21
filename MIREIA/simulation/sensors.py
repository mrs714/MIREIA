# Spawns Cameras, Lidar, Collision sensors
import queue

import carla
import numpy as np

class SensorManager:
    """
    Manages the RGB camera attached to the vehicle, the one on top of the map, and any other ones set along the way.
    """

    def __init__(self, world: carla.World, map: carla.Map, ego_vehicle: carla.Actor,
                 save_dir: str, ego_resolution=(800, 600), map_resolution=(2000, 2000),
                 enable_map_camera: bool = True,
                 ego_camera_position: tuple[float, float, float] | None = None,
                 ego_camera_fov: float = 110.0,
                 map_center: tuple[float, float] | None = None,
                 map_size: float | None = None,
                 map_fov: float = 90.0,
                 map_rotation_yaw: float = 0.0,
                 map_rotation_roll: float = 0.0):
        self.__world = world
        self.__map = map
        self.__ego_vehicle = ego_vehicle
        self.__save_dir = save_dir
        # Initialize camera attributes
        self.__ego_camera = None
        self.__map_camera = None
        self.__other_cameras = []
        self.__map_center = None
        self.__map_size = map_size
        self.__map_fov = map_fov
        self.__map_rotation_yaw = map_rotation_yaw
        self.__map_rotation_roll = map_rotation_roll
        self.__map_camera_enabled = enable_map_camera
        self.__setup_ego_camera(ego_resolution, ego_camera_position, ego_camera_fov)
        if self.__map_camera_enabled:
            if map_center is None:
                self.__map_center = self.__get_map_center()
            else:
                self.__map_center = carla.Location(x=map_center[0], y=map_center[1], z=0)
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
    
    def __setup_ego_camera(self, ego_resolution, ego_camera_position: tuple[float, float, float] | None,
                           ego_camera_fov: float):
        blueprint = self.__world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(ego_resolution[0]))
        blueprint.set_attribute('image_size_y', str(ego_resolution[1]))
        blueprint.set_attribute('fov', str(ego_camera_fov))
        if ego_camera_position is None:
            ego_camera_position = (0.0, 0.0, 1.5)
        camera_init_trans = carla.Transform(
            carla.Location(x=ego_camera_position[0], y=ego_camera_position[1], z=ego_camera_position[2])
        )
        self.__ego_camera = self.__world.spawn_actor(blueprint, camera_init_trans, attach_to=self.__ego_vehicle)

        self.__ego_camera.enable_postprocess_effects = True

    def __setup_map_camera(self, map_resolution):
            blueprint = self.__world.get_blueprint_library().find('sensor.camera.rgb')
            
            # --- SET ATTRIBUTES ON THE BLUEPRINT FIRST ---
            # Note: attributes are always strings in the blueprint library
            blueprint.set_attribute('image_size_x', str(map_resolution[0]))
            blueprint.set_attribute('image_size_y', str(map_resolution[1]))
            blueprint.set_attribute('fov', str(self.__map_fov))
            blueprint.set_attribute('bloom_intensity', '0.0') # Better way to kill the 'laser' glow
            blueprint.set_attribute('lens_flare_intensity', '0.0')

            world_center = self.__map_center
            map_location = carla.Location(x=world_center.x, y=world_center.y, z=self.__compute_map_camera_height(map_resolution))
            map_rotation = carla.Rotation(pitch=-90,
                                          yaw=self.__map_rotation_yaw,
                                          roll=self.__map_rotation_roll)
            map_transform = carla.Transform(map_location, map_rotation)
            
            self.__map_camera = self.__world.spawn_actor(blueprint, map_transform)

    def __compute_map_camera_height(self, map_resolution: tuple[int, int]) -> float:
        if self.__map_size is None:
            return 150.0

        width_px, height_px = map_resolution
        aspect = width_px / max(1, height_px)
        target_view_height = max(self.__map_size, self.__map_size / max(1e-6, aspect))
        fov_rad = np.deg2rad(self.__map_fov)
        return (target_view_height / 2.0) / max(1e-6, np.tan(fov_rad / 2.0))

    def __capture_single(self, camera: carla.Actor, tick_fn, timeout: float):
        image_queue: queue.Queue = queue.Queue()

        def callback(image):
            image_queue.put(image)

        camera.listen(callback)
        try:
            if tick_fn is not None:
                tick_fn()
            image = image_queue.get(timeout=timeout)
        finally:
            camera.stop()
        return image

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
    
    def save_ego_frame(self, save_path: str | None = None,
                       tick_fn=None, timeout: float = 2.0) -> str:
        if tick_fn is None:
            tick_fn = self.__world.tick
        image = self.__capture_single(self.__ego_camera, tick_fn, timeout)
        if save_path is None:
            save_path = f"{self.__save_dir}/ego_{image.frame}.png"
        image.save_to_disk(save_path)
        return save_path

    def save_ego_frames(self, duration: float = None):
        self.__save_snapshots(self.__ego_camera, "ego", duration)

    def save_map_frame(self, save_path: str | None = None,
                       tick_fn=None, timeout: float = 2.0) -> str:
        if self.__map_camera is None:
            raise RuntimeError("Map camera is disabled or not initialized.")
        if tick_fn is None:
            tick_fn = self.__world.tick
        image = self.__capture_single(self.__map_camera, tick_fn, timeout)
        if save_path is None:
            save_path = f"{self.__save_dir}/map_{image.frame}.png"
        image.save_to_disk(save_path)
        return save_path

    def save_map_frames(self, duration: float = None):
        if self.__map_camera is None:
            raise RuntimeError("Map camera is disabled or not initialized.")
        self.__save_snapshots(self.__map_camera, "map", duration)