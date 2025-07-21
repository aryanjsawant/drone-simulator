import argparse
import logging
import math
import numpy as np
from panda3d.core import LVector3
from direct.interval.IntervalGlobal import Sequence, LerpPosInterval
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task

from dronesim import SimulatorApplication, Panda3DEnvironment, make_uav
from dronesim.interface import DroneAction, DroneState
from dronesim.sensor.panda3d.camera import Panda3DCameraSensor

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["a_star", "dijkstra", "bfs", "dfs"], default="a_star",
                        help="Choose the pathfinding algorithm to use.")
    args = parser.parse_args()

    sim, controller, drone = make_uav()
    env = Panda3DEnvironment("basic_env", num_buildings=6, algorithm=args.algo)
    app = SimulatorApplication(env, drone)

    down_cam = Panda3DCameraSensor("downCameraRGB", size=(512, 512))
    sim.add_sensor(down_camera_rgb=down_cam)
    down_cam.reparent_to(drone)
    down_cam.set_hpr(0, -90, 0)

    def start_circular_path():
        if drone.controller.drone.state.get('operation') != DroneState.IN_AIR:
            logging.info("Drone is not in the air. Taking off first.")
            drone.controller.direct_action(DroneAction.TAKEOFF, altitude=50.0)
            taskMgr.doMethodLater(5, lambda task: start_circular_path_task(task), "start_circular_path_task")
        else:
            start_circular_path_task(None)

    def start_circular_path_task(task):
        if not hasattr(drone, 'path_active') or not drone.path_active:
            drone.path_active = True
            drone.start_pos = drone.get_pos(base.render)
            drone.start_hpr = drone.get_hpr(base.render)
            drone.radius = 100.0
            drone.center = drone.start_pos - LVector3(drone.radius, 0, 0)
            drone.time = 0.0
            drone.speed = 1.5
            drone.max_time = (2 * math.pi) / drone.speed
            taskMgr.add(update_circular_path, "UpdateCircularPathTask")
            logging.info(f"Drone started circular path at center {drone.center}")
        return Task.done

    def update_circular_path(task):
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
        if hasattr(drone, 'path_active') and drone.path_active:
            drone.path_active = False
            taskMgr.remove("UpdateCircularPathTask")
            start_pos = drone.start_pos if hasattr(drone, 'start_pos') else drone.get_pos()
            drone.controller.direct_action(DroneAction.STOP_IN_PLACE)
            logging.info("Drone stopped circular path.")
            seq = Sequence(LerpPosInterval(drone, 3.0, start_pos, startPos=drone.get_pos()))
            seq.start()

    def takeoff():
        if drone.controller.drone.state.get('operation') != DroneState.IN_AIR:
            logging.info("Drone is not in the air. Taking off.")
            drone.controller.direct_action(DroneAction.TAKEOFF, altitude=50.0)
        else:
            logging.info("Drone is already in the air.")

    def land():
        if drone.controller.drone.state.get('operation') == DroneState.LANDED:
            logging.info("Drone is already landed.")
            return

        logging.info("Drone landing (controller will handle descent).")
        drone.controller.direct_action(DroneAction.LAND)

    def start_autonomous_flight():
        current_pos = drone.get_pos(base.render)
        start_pos = LVector3(0, 0, 0)
        if drone.controller.drone.state.get('operation') == DroneState.LANDED and current_pos == start_pos:
            path = env.get_path()
            if not path:
                logging.info("No path available to follow. Generate the map first.")
                return

            logging.info("Drone starting autonomous flight.")

            path_intervals = []
            takeoff_height = 50.0
            vertical_speed = 10.0
            horizontal_speed = 25.0

            grid_rows, grid_cols = env.occupancy_map.shape
            cell_w = env.ENV_DIMENSIONS / grid_cols
            cell_h = env.ENV_DIMENSIONS / grid_rows

            takeoff_pos = LVector3(current_pos.x, current_pos.y, takeoff_height)
            path_intervals.append(LerpPosInterval(drone, (takeoff_pos - current_pos).length() / vertical_speed, takeoff_pos, startPos=current_pos))
            current_pos = takeoff_pos

            for i, node in enumerate(path):
                x = (node[1] * cell_w) + (cell_w / 2) - (env.ENV_DIMENSIONS / 2)
                y = (node[0] * cell_h) + (cell_h / 2) - (env.ENV_DIMENSIONS / 2)
                target_pos = LVector3(x, y, takeoff_height)

                if i == 0 and (current_pos - target_pos).length() < 1e-2:
                    continue

                path_intervals.append(LerpPosInterval(drone, (target_pos - current_pos).length() / horizontal_speed, target_pos, startPos=current_pos))
                current_pos = target_pos

            final_node = path[-1]
            final_x = (final_node[1] * cell_w) + (cell_w / 2) - (env.ENV_DIMENSIONS / 2)
            final_y = (final_node[0] * cell_h) + (cell_h / 2) - (env.ENV_DIMENSIONS / 2)
            final_pos = LVector3(final_x, final_y, 0)

            path_intervals.append(LerpPosInterval(drone, (final_pos - current_pos).length() / vertical_speed, final_pos, startPos=current_pos))

            logging.info(f"Drone finished following {args.algo} path and is landing.")
            seq = Sequence(*path_intervals, name="AutonomousFlight")
            seq.start()
        else:
            logging.info("Drone must be at the starting position (0, 0, 0) and landed to start autonomous flight.")

    def follow_path():
        path = env.get_path()
        if not path:
            logging.info("No A* path available to follow. Generate the map first.")
            return

        if drone.controller.drone.state.get('operation') != DroneState.LANDED:
            logging.info("Drone must be on the ground to follow path.")
            return

        logging.info("Drone following A* path.")

        path_intervals = []
        current_pos = drone.get_pos(base.render)
        takeoff_height = 50.0
        vertical_speed = 10.0
        horizontal_speed = 25.0

        if current_pos.z < takeoff_height:
            takeoff_pos = LVector3(current_pos.x, current_pos.y, takeoff_height)
            path_intervals.append(LerpPosInterval(drone, (takeoff_pos - current_pos).length() / vertical_speed, takeoff_pos, startPos=current_pos))
            current_pos = takeoff_pos

        grid_rows, grid_cols = env.occupancy_map.shape
        cell_w = env.ENV_DIMENSIONS / grid_cols
        cell_h = env.ENV_DIMENSIONS / grid_rows

        for i, node in enumerate(path):
            x = (node[1] * cell_w) + (cell_w / 2) - (env.ENV_DIMENSIONS / 2)
            y = (node[0] * cell_h) + (cell_h / 2) - (env.ENV_DIMENSIONS / 2)
            target_pos = LVector3(x, y, takeoff_height)

            if i == 0 and current_pos.x == target_pos.x and current_pos.y == target_pos.y:
                continue

            path_intervals.append(LerpPosInterval(drone, (target_pos - current_pos).length() / horizontal_speed, target_pos, startPos=current_pos))
            current_pos = target_pos

        final_node = path[-1]
        final_x = (final_node[1] * cell_w) + (cell_w / 2) - (env.ENV_DIMENSIONS / 2)
        final_y = (final_node[0] * cell_h) + (cell_h / 2) - (env.ENV_DIMENSIONS / 2)
        final_pos = LVector3(final_x, final_y, 0)

        path_intervals.append(LerpPosInterval(drone, (final_pos - current_pos).length() / vertical_speed, final_pos, startPos=current_pos))
        seq = Sequence(*path_intervals, name="AStarPathFlight")
        seq.start()

    app.accept("i", takeoff)
    app.accept("z", start_circular_path)
    app.accept("x", stop_circular_path)
    app.accept("k", land)
    app.accept("c", start_autonomous_flight)
    app.accept("p", follow_path)

    app.run()

if __name__ == "__main__":
    main()
