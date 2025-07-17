import argparse
import logging
from panda3d.core import LVector3
from direct.interval.IntervalGlobal import Sequence, LerpPosInterval
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task

from dronesim import SimulatorApplication, Panda3DEnvironment, make_uav
from dronesim.interface import DroneAction, DroneState
from dronesim.sensor.panda3d.camera import Panda3DCameraSensor
import math
import csv
import numpy as np

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    sim, controller1, drone1 = make_uav()
    _, controller2, drone2 = make_uav()

    env = Panda3DEnvironment("basic_env", num_buildings=10)
    app = SimulatorApplication(env, drone1, drone2)

    down_cam1 = Panda3DCameraSensor("downCameraRGB_1", size=(512, 512))
    sim.add_sensor(down_camera_rgb=down_cam1)
    down_cam1.reparent_to(drone1)
    down_cam1.set_hpr(0, -90, 0)

    # --- Key Handlers ---

    def start_circular_path():
        """
        Handles the 'z' key press.
        Takes off to a specified height and flies in a circular path.
        """
        if drone1.controller.drone.state.get('operation') != DroneState.IN_AIR:
            logging.info("Drone 1 is not in the air. Taking off first.")
            drone1.controller.direct_action(DroneAction.TAKEOFF, altitude=50.0)
            # It might take a moment to take off, so we'll start the path in a separate task
            taskMgr.doMethodLater(5, lambda task: start_circular_path_task(task, drone1), "start_circular_path_task")
        else:
            start_circular_path_task(None, drone1)

    def start_circular_path_task(task, drone):
        if not hasattr(drone, 'path_active') or not drone.path_active:
            drone.path_active = True
            drone.start_pos = drone.get_pos(base.render)
            drone.start_hpr = drone.get_hpr(base.render)
            drone.radius = 100.0
            drone.center = drone.start_pos - LVector3(drone.radius, 0, 0)
            drone.time = 0.0
            drone.speed = 1.5
            drone.max_time = (2 * math.pi) / drone.speed
            taskMgr.add(update_circular_path, "UpdateCircularPathTask", extraArgs=[drone], appendTask=True)
            logging.info(f"Drone 1 started circular path at center {drone.center}")
        return Task.done

    def update_circular_path(drone, task):
        if not hasattr(drone, 'path_active') or not drone.path_active:
            return Task.done

        dt = globalClock.get_dt()
        drone.time += dt

        if drone.time >= drone.max_time:
            drone.time -= drone.max_time

        angle = drone.time * drone.speed
        z_offset = math.sin(angle) * 0.5

        x = drone.center.x + drone.radius * math.cos(angle)
        y = drone.center.y + drone.radius * math.sin(angle)
        z = drone.start_pos.z + z_offset

        drone.set_pos(x, y, z)
        drone.look_at(
            drone.center.x + drone.radius * math.cos(angle + 0.01),
            drone.center.y + drone.radius * math.sin(angle + 0.01),
            z
        )
        return Task.cont

    def stop_circular_path():
        """
        Handles the 'x' key press.
        Stops the circular path and returns the drone to its starting position.
        """
        if hasattr(drone1, 'path_active') and drone1.path_active:
            drone1.path_active = False
            taskMgr.remove("UpdateCircularPathTask")
            # Return to start position
            start_pos = drone1.start_pos if hasattr(drone1, 'start_pos') else drone1.get_pos()
            drone1.controller.direct_action(DroneAction.STOP_IN_PLACE) # A bit of a hack, but it works
            logging.info("Drone 1 stopped circular path.")
            # Go back to original position
            seq = Sequence(LerpPosInterval(drone1, 3.0, start_pos, startPos=drone1.get_pos()))
            seq.start()


    def land_drone():
        """
        Handles the 'k' key press.
        Lands drone 2 at the A* goal position.
        """
        a_star_path = env.get_a_star_path()
        if not a_star_path:
            logging.info("No A* path available to determine landing goal. Generate the map first.")
            return

        if drone2.controller.drone.state.get('operation') == DroneState.LANDED:
            logging.info("Drone 2 is already landed.")
            return

        logging.info("Drone 2 landing at A* goal.")

        grid_rows, grid_cols = env.occupancy_map.shape
        grid_cell_width = env.ENV_DIMENSIONS / grid_cols
        grid_cell_height = env.ENV_DIMENSIONS / grid_rows

        # Get the A* goal node from the environment (which is the last node in the path)
        a_star_goal_node = a_star_path[-1]
        final_landing_x = (a_star_goal_node[1] * grid_cell_width) + (grid_cell_width / 2) - (env.ENV_DIMENSIONS / 2)
        final_landing_y = (a_star_goal_node[0] * grid_cell_height) + (grid_cell_height / 2) - (env.ENV_DIMENSIONS / 2)
        final_landing_pos = LVector3(final_landing_x, final_landing_y, 0)

        current_world_pos = drone2.get_pos(base.render)
        vertical_speed = 10.0

        seq = Sequence(LerpPosInterval(
            drone2,
            duration=(final_landing_pos - current_world_pos).length() / vertical_speed,
            pos=final_landing_pos,
            startPos=current_world_pos
        ), name="LandDrone2AtAStarGoal")
        seq.start()

        # After the movement, trigger the actual LAND action
        taskMgr.doMethodLater(seq.getDuration(), lambda task: drone2.controller.direct_action(DroneAction.LAND), "TriggerLandAction")

    def return_drone2_to_a_star_start():
        """
        Handles the 'x' key press.
        Returns drone 2 to the A* start position.
        """
        a_star_path = env.get_a_star_path()
        if not a_star_path:
            logging.info("No A* path available to determine start position. Generate the map first.")
            return

        logging.info("Drone 2 returning to A* start position.")

        grid_rows, grid_cols = env.occupancy_map.shape
        grid_cell_width = env.ENV_DIMENSIONS / grid_cols
        grid_cell_height = env.ENV_DIMENSIONS / grid_rows

        # Get the A* start node from the environment (which is the first node in the path)
        a_star_start_node = a_star_path[0]
        target_world_x = (a_star_start_node[1] * grid_cell_width) + (grid_cell_width / 2) - (env.ENV_DIMENSIONS / 2)
        target_world_y = (a_star_start_node[0] * grid_cell_height) + (grid_cell_height / 2) - (env.ENV_DIMENSIONS / 2)
        target_world_pos = LVector3(target_world_x, target_world_y, 50.0) # Return to takeoff height

        current_world_pos = drone2.get_pos(base.render)
        horizontal_speed = 25.0

        seq = Sequence(LerpPosInterval(
            drone2,
            duration=(target_world_pos - current_world_pos).length() / horizontal_speed,
            pos=target_world_pos,
            startPos=current_world_pos
        ), name="ReturnDrone2ToAStarStart")
        seq.start()

    # --- Key Handlers ---

    def takeoff_drone1():
        """
        Handles the 'i' key press.
        Takes off drone 1.
        """
        if drone1.controller.drone.state.get('operation') != DroneState.IN_AIR:
            logging.info("Drone 1 is not in the air. Taking off.")
            drone1.controller.direct_action(DroneAction.TAKEOFF, altitude=50.0)
        else:
            logging.info("Drone 1 is already in the air.")

    app.accept("i", takeoff_drone1)
    app.accept("x", return_drone2_to_a_star_start)
    app.accept("k", land_drone)

    def start_autonomous_flight():
        """
        Handles the 'c' key press.
        Starts an autonomous flight sequence for drone 2 using the A* path.
        """
        if drone2.controller.drone.state.get('operation') == DroneState.LANDED:
            logging.info("Drone 2 starting autonomous sequence.")
            a_star_path = env.get_a_star_path()
            if not a_star_path:
                logging.info("No A* path available to follow. Generate the map first.")
                return

            path_intervals = []
            current_world_pos = drone2.get_pos(base.render)
            takeoff_height = 50.0
            vertical_speed = 10.0
            horizontal_speed = 25.0

            grid_rows, grid_cols = env.occupancy_map.shape
            grid_cell_width = env.ENV_DIMENSIONS / grid_cols
            grid_cell_height = env.ENV_DIMENSIONS / grid_rows

            # 1. Takeoff
            takeoff_target_pos = LVector3(current_world_pos.x, current_world_pos.y, takeoff_height)
            path_intervals.append(LerpPosInterval(
                drone2,
                duration=(takeoff_target_pos - current_world_pos).length() / vertical_speed,
                pos=takeoff_target_pos,
                startPos=current_world_pos
            ))
            current_world_pos = takeoff_target_pos
            logging.info("Drone 2 takeoff complete.")

            # 2. Follow A* path
            for i, grid_node in enumerate(a_star_path):
                target_world_x = (grid_node[1] * grid_cell_width) + (grid_cell_width / 2) - (env.ENV_DIMENSIONS / 2)
                target_world_y = (grid_node[0] * grid_cell_height) + (grid_cell_height / 2) - (env.ENV_DIMENSIONS / 2)
                target_world_pos = LVector3(target_world_x, target_world_y, takeoff_height)

                if i == 0 and (current_world_pos - target_world_pos).length() < 1e-2:
                    continue

                path_intervals.append(LerpPosInterval(
                    drone2,
                    duration=(target_world_pos - current_world_pos).length() / horizontal_speed,
                    pos=target_world_pos,
                    startPos=current_world_pos
                ))
                current_world_pos = target_world_pos

            logging.info("Drone 2 finished following A* path.")

            # 3. Land at the end of the A* path
            if a_star_path:
                last_grid_node = a_star_path[-1]
                final_landing_x = (last_grid_node[1] * grid_cell_width) + (grid_cell_width / 2) - (env.ENV_DIMENSIONS / 2)
                final_landing_y = (last_grid_node[0] * grid_cell_height) + (grid_cell_height / 2) - (env.ENV_DIMENSIONS / 2)
                final_landing_pos = LVector3(final_landing_x, final_landing_y, 0)
            else:
                final_landing_pos = LVector3(current_world_pos.x, current_world_pos.y, 0) # Fallback to current position

            path_intervals.append(LerpPosInterval(
                drone2,
                duration=(final_landing_pos - current_world_pos).length() / vertical_speed,
                pos=final_landing_pos,
                startPos=current_world_pos
            ))
            logging.info("Drone 2 landing at A* goal.")

            seq = Sequence(*path_intervals, name="AutonomousFlightSequence")
            seq.start()
        else:
            logging.info("Drone 2 must be on the ground to start autonomous flight.")

    def follow_a_star_path():
        """
        Handles the 'p' key press.
        Makes drone 2 follow the A* path generated in the environment.
        """
        a_star_path = env.get_a_star_path()
        if not a_star_path:
            logging.info("No A* path available to follow. Generate the map first.")
            return

        if drone2.controller.drone.state.get('operation') != DroneState.LANDED:
            logging.info("Drone 2 must be on the ground to start A* path following.")
            return

        logging.info("Drone 2 starting to follow A* path.")

        path_intervals = []
        current_world_pos = drone2.get_pos(base.render)
        takeoff_height = 50.0
        vertical_speed = 10.0
        horizontal_speed = 25.0

        grid_rows, grid_cols = env.occupancy_map.shape
        grid_cell_width = env.ENV_DIMENSIONS / grid_cols
        grid_cell_height = env.ENV_DIMENSIONS / grid_rows

        # Ensure drone is at takeoff height before starting path traversal
        if current_world_pos.z < takeoff_height:
            takeoff_target_pos = LVector3(current_world_pos.x, current_world_pos.y, takeoff_height)
            path_intervals.append(LerpPosInterval(
                drone2,
                duration=(takeoff_target_pos - current_world_pos).length() / vertical_speed,
                pos=takeoff_target_pos,
                startPos=current_world_pos
            ))
            current_world_pos = takeoff_target_pos

        # Traverse the A* path
        for i, grid_node in enumerate(a_star_path):
            # Convert grid coordinates to world coordinates (center of the cell)
            target_world_x = (grid_node[1] * grid_cell_width) + (grid_cell_width / 2) - (env.ENV_DIMENSIONS / 2)
            target_world_y = (grid_node[0] * grid_cell_height) + (grid_cell_height / 2) - (env.ENV_DIMENSIONS / 2)
            target_world_pos = LVector3(target_world_x, target_world_y, takeoff_height)

            # Skip the first point if it's the current position after takeoff
            if i == 0 and current_world_pos.x == target_world_pos.x and current_world_pos.y == target_world_pos.y:
                continue

            path_intervals.append(LerpPosInterval(
                drone2,
                duration=(target_world_pos - current_world_pos).length() / horizontal_speed,
                pos=target_world_pos,
                startPos=current_world_pos
            ))
            current_world_pos = target_world_pos

        # Final landing at the end of the path
        if a_star_path:
            last_grid_node = a_star_path[-1]
            final_landing_x = (last_grid_node[1] * grid_cell_width) + (grid_cell_width / 2) - (env.ENV_DIMENSIONS / 2)
            final_landing_y = (last_grid_node[0] * grid_cell_height) + (grid_cell_height / 2) - (env.ENV_DIMENSIONS / 2)
            final_landing_pos = LVector3(final_landing_x, final_landing_y, 0)
        else:
            final_landing_pos = LVector3(current_world_pos.x, current_world_pos.y, 0) # Fallback to current position

        path_intervals.append(LerpPosInterval(
            drone2,
            duration=(final_landing_pos - current_world_pos).length() / vertical_speed,
            pos=final_landing_pos,
            startPos=current_world_pos
        ))

        seq = Sequence(*path_intervals, name="AStarFlightSequence")
        seq.start()


    app.accept("z", start_circular_path)
    app.accept("x", stop_circular_path)
    app.accept("k", land_drone)
    app.accept("c", start_autonomous_flight)
    app.accept("p", follow_a_star_path)

    app.run()

if __name__ == "__main__":
    main()
