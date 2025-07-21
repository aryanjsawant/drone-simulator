import argparse
import logging
import math
import numpy as np
from panda3d.core import LVector3, Point3
from direct.interval.IntervalGlobal import Sequence, LerpPosInterval
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task

from dronesim import SimulatorApplication, Panda3DEnvironment, make_uav, DroneSimulator
from dronesim.interface import DroneAction, DroneState, IDroneControllable
from dronesim.sensor.panda3d.camera import Panda3DCameraSensor
from dronesim.actor.uav import UAVDroneModel # Import UAVDroneModel
from dronesim.algorithms.clustering import assign_buildings_to_drones # Import clustering algorithm

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["a_star", "dijkstra", "bfs", "dfs"], default="a_star",
                        help="Choose the pathfinding algorithm to use.")
    parser.add_argument("--energy-save", action="store_true", help="Enable energy-saving mode (5 drones, 2 buildings each).")
    args = parser.parse_args()

    # Create the environment first to get building data
    env = Panda3DEnvironment("basic_env", num_buildings=10, algorithm=args.algo)
    
    drones = []
    controllers = []
    num_drones = 5 if args.energy_save else 10
    
    for i in range(num_drones):
        sim, controller, drone_model = make_uav()
        drones.append(drone_model)
        controllers.append(controller)
        
        # Calculate scattered offset for each drone
        # Using a 2x5 grid for 10 drones
        row = i // 5
        col = i % 5
        offset_x = (col - 2) * 5.0  # Spread along X-axis
        offset_y = (row - 0.5) * 5.0 # Spread along Y-axis
        drone_model.set_offset(offset_x, offset_y, 0) 
        
        # Make drones smaller by half
        drone_model.set_scale(0.8) 

    app = SimulatorApplication(env, *drones) # Pass all drones to the app

    down_cam = Panda3DCameraSensor("downCameraRGB", size=(512, 512))
    # Attach camera to the first drone for now
    drones[0].controller.drone.add_sensor(down_camera_rgb=down_cam)
    down_cam.reparent_to(drones[0])
    down_cam.set_hpr(0, -90, 0)

    TAKEOFF_ALTITUDE = 100.0 # Drones will initially take off to this altitude

    def start_drone_mission():
        logging.info("Starting drone mission: Takeoff and hover over buildings.")
        building_hover_points = env.get_building_hover_points()
        
        if len(building_hover_points) < num_drones:
            logging.warning(f"Not enough buildings ({len(building_hover_points)}) for {num_drones} drones. Some drones will not have a target.")

        if args.energy_save:
            logging.info("Energy-saving mode activated. Assigning buildings using clustering algorithm.")
            drone_waypoints = assign_buildings_to_drones(building_hover_points, num_drones)
            for i, drone in enumerate(drones):
                if i < len(drone_waypoints):
                    waypoints = drone_waypoints[i]
                    if waypoints:
                        logging.info(f"Drone {i+1} taking off to {TAKEOFF_ALTITUDE} and then following waypoints: {waypoints}")
                        drone.controller.direct_action(DroneAction.TAKEOFF, altitude=TAKEOFF_ALTITUDE)
                        app.taskMgr.doMethodLater(6.0, drone.follow_waypoints, f'drone_waypoints_task_{i}', extraArgs=[waypoints])
                    else:
                        logging.info(f"Drone {i+1} has no waypoints assigned.")
                else:
                    logging.info(f"Drone {i+1} has no building target.")
        else:
            for i, drone in enumerate(drones):
                if i < len(building_hover_points):
                    target_pos = building_hover_points[i]
                    logging.info(f"Drone {i+1} taking off to {TAKEOFF_ALTITUDE} and then moving to {target_pos}")
                    drone.controller.direct_action(DroneAction.TAKEOFF, altitude=TAKEOFF_ALTITUDE)
                    # Schedule the horizontal movement after a short delay to allow for takeoff
                    app.taskMgr.doMethodLater(3.0, drone.go_to_and_hover, f'drone_move_task_{i}', extraArgs=[target_pos])
                else:
                    logging.info(f"Drone {i+1} has no building target.")

    app.accept("h", start_drone_mission) # Bind 'h' key to start the mission

    app.run()

if __name__ == "__main__":
    main()
