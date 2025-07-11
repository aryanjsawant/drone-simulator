import os
import random
import csv
import numpy as np
from dronesim.algorithms.a_star import a_star, read_occupancy_map, plot_path
from panda3d.core import (
    NodePath,
    PandaNode,
    Light,
    AmbientLight,
    DirectionalLight,
    Filename,
    LVecBase3f,
    LColor
)

from direct.showbase.Loader import Loader
from direct.task import Task, TaskManagerGlobal

from os import PathLike
from dronesim.types import PandaFilePath
from typing import Optional, Union, List


class Panda3DEnvironment(NodePath):
    # Default scene to load (adjusted path for your project structure)
    DEFAULT_SCENE = Filename("dronesim/assets/scenes/simple_loop.glb")
    DEFAULT_LOADER = Loader(None)
    ENV_DIMENSIONS = 320

    def __init__(self,
                 name: str,
                 scene_model: Optional[Union[PandaFilePath,
                                             NodePath]] = DEFAULT_SCENE,
                 attach_lights: List[Union[Light, NodePath]] = [],
                 enable_dull_ambient_light: bool = True,
                 loader: Loader = DEFAULT_LOADER,
                 task_mgr: Task.TaskManager = TaskManagerGlobal.taskMgr,
                 num_buildings: int = 9):
        super().__init__(name)

        self._attach_lights = attach_lights
        self._loader = loader
        self.building_positions = []
        self.occupancy_map = np.zeros((6, 8), dtype=int)
        self.a_star_path = None # Store the A* path here

        # Add ambient lighting (minimum scene light)
        if enable_dull_ambient_light:
            ambient = AmbientLight('ambient_light')
            ambient.set_color(LColor(0.6, 0.6, 0.6, 1))  # Dimmed ambient light
            ambient_np = self.attach_new_node(ambient)
            self.set_light(ambient_np)
            self._attach_lights.append(ambient)

        # Add directional light to simulate sunlight
        dir_light = DirectionalLight('directional_light')
        dir_light.set_color(LColor(1.2, 1.2, 1.0, 1))  # Brighter, warm white light
        dir_light_np = self.attach_new_node(dir_light)
        dir_light_np.set_hpr(45, -60, 0)  # angle the light
        self.set_light(dir_light_np)
        self._attach_lights.append(dir_light)

        self._setup_scene_lighting()
        task_mgr.add(self._load_random_buildings(num_buildings))

        # Load the given scene, if any
        if scene_model is not None:
            # Schedule async load of the scene model
            task_mgr.add(self.load_attach_scene(scene_model))

    async def load_attach_scene(self,
                                scene_path: Union[PandaFilePath, NodePath, PandaNode],
                                position: LVecBase3f = None,
                                rotation: LVecBase3f = None,
                                scale: LVecBase3f = None) -> NodePath:
        '''Attach a scene into this environment. The scene can be a NodePath, PandaNode
        or a File path to the model (physical or virtual file)'''
        if isinstance(scene_path, (str, PathLike, Filename)):
            # Load scene from given path
            scene_model = await self._loader.load_model(scene_path, blocking=False)
        elif isinstance(scene_path, NodePath):
            scene_model = scene_path
        elif isinstance(scene_path, PandaNode):
            scene_model = NodePath(scene_path)
        else:
            raise TypeError("Argument `scene_path` is not a valid type.")

        self.attach_scene(scene_model)

        # Optional transformation
        if position:
            scene_model.set_pos(position)
        if rotation:
            scene_model.set_hpr(rotation)
        if scale:
            scene_model.set_scale(scale)

        return scene_model

    def attach_scene(self, scene_model: NodePath):
        '''Attach the NodePath (model) instance to the environment'''
        scene_model.instance_to(self)

    def _setup_scene_lighting(self):
        '''
        Move all declared lights into the environment
        '''
        for light_node in self._attach_lights:
            if isinstance(light_node, Light):
                self.attach_new_node(light_node)
            elif isinstance(light_node, NodePath):
                light_node.reparent_to(self)
            else:
                raise TypeError(
                    "Received a light that has incorrect type (must be a subclass of p3d.Light).")

    async def _load_random_buildings(self, num_buildings: int):
        building_assets_dir = "dronesim/assets/buildings"
        building_models = [os.path.join(building_assets_dir, f) for f in os.listdir(building_assets_dir) if f.endswith(".glb")]
        safe_radius = 50.0
        min_spacing = 50.0

        if not building_models:
            return

        occupied_cells = set()

        for _ in range(num_buildings):
            model_path = random.choice(building_models)
            
            # Load the model to get its size
            model = await self._loader.load_model(model_path, blocking=False)
            min_point, max_point = model.getTightBounds()
            model_size = max_point - min_point

            # Set a random scale
            scale_x = (self.ENV_DIMENSIONS / 8) / model_size.x
            scale_y = (self.ENV_DIMENSIONS / 6) / model_size.y
            scale_val = min(scale_x, scale_y)
            scale = LVecBase3f(scale_val, scale_val, scale_val)
            scaled_size = LVecBase3f(model_size.x * scale.x, model_size.y * scale.y, model_size.z * scale.z)

            # Calculate placement bounds to keep the building inside the environment
            half_env = self.ENV_DIMENSIONS / 2
            max_x = half_env - (scaled_size.x / 2)
            max_y = half_env - (scaled_size.y / 2)
            
            # Set a random position within the calculated bounds, avoiding the safe zone and other buildings
            while True:
                position = LVecBase3f(random.uniform(-max_x, max_x), random.uniform(-max_y, max_y), 0)
                grid_x = int((position.x + half_env) / (self.ENV_DIMENSIONS / 8))
                grid_y = int((position.y + half_env) / (self.ENV_DIMENSIONS / 6))

                if position.length() < safe_radius:
                    continue

                if (grid_x, grid_y) in occupied_cells:
                    continue
                
                is_well_spaced = True
                for existing_pos in self.building_positions:
                    if (position - existing_pos).length() < min_spacing:
                        is_well_spaced = False
                        break
                
                if is_well_spaced:
                    occupied_cells.add((grid_x, grid_y))
                    break
            
            self.building_positions.append(position)

            # Attach and transform the model
            model.reparent_to(self)
            model.set_pos(position)
            model.set_scale(scale)
            model.set_color_scale(random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), 1)

            # Update occupancy map
            grid_rows, grid_cols = self.occupancy_map.shape
            cell_width = self.ENV_DIMENSIONS / grid_cols
            cell_height = self.ENV_DIMENSIONS / grid_rows

            # Calculate the min and max world coordinates of the building's bounding box
            building_min_x = position.x - (scaled_size.x / 2)
            building_max_x = position.x + (scaled_size.x / 2)
            building_min_y = position.y - (scaled_size.y / 2)
            building_max_y = position.y + (scaled_size.y / 2)

            # Convert world coordinates to grid coordinates
            grid_min_col = int((building_min_x + half_env) / cell_width)
            grid_max_col = int((building_max_x + half_env) / cell_width)
            grid_min_row = int((building_min_y + half_env) / cell_height)
            grid_max_row = int((building_max_y + half_env) / cell_height)

            # Ensure coordinates are within grid bounds
            grid_min_col = max(0, grid_min_col)
            grid_max_col = min(grid_cols - 1, grid_max_col)
            grid_min_row = max(0, grid_min_row)
            grid_max_row = min(grid_rows - 1, grid_max_row)

            # Mark all occupied cells in the occupancy map
            for r in range(grid_min_row, grid_max_row + 1):
                for c in range(grid_min_col, grid_max_col + 1):
                    self.occupancy_map[r, c] = 1

        self._save_and_print_map()

    def _save_and_print_map(self):
        # Print map to console
        print("Occupancy Map:")
        print(self.occupancy_map)

        # Save map to CSV
        with open("occupancy_map.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.occupancy_map)

        # Run A* algorithm
        grid = read_occupancy_map("occupancy_map.csv")
        start_node = (3, 4)  # Corresponds to drone's (0,0,0) world position
        goal_node = (5, 7)  # Example goal node
        path = a_star(grid, start_node, goal_node)

        if path:
            print("A* Path found:", path)
            plot_path(grid, path, "a_star_path.png")
            print("A* path image saved to a_star_path.png")
            self.a_star_path = path
        else:
            print("No A* path found.")
            self.a_star_path = None

    def get_a_star_path(self):
        return self.a_star_path