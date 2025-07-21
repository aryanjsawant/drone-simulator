from panda3d.core import LVecBase3f
from typing import List, Tuple
import math

def calculate_distance(p1: LVecBase3f, p2: LVecBase3f) -> float:
    return (p1 - p2).length()

def assign_buildings_to_drones(building_positions: List[LVecBase3f], num_drones: int) -> List[List[LVecBase3f]]:
    if len(building_positions) != 10 or num_drones != 5:
        raise ValueError("This clustering algorithm is designed for 10 buildings and 5 drones.")

    assigned_buildings = [False] * len(building_positions)
    drone_waypoints: List[List[LVecBase3f]] = [[] for _ in range(num_drones)]

    # Calculate all pairwise distances
    distances: List[Tuple[float, int, int]] = []
    for i in range(len(building_positions)):
        for j in range(i + 1, len(building_positions)):
            dist = calculate_distance(building_positions[i], building_positions[j])
            distances.append((dist, i, j))
    
    # Sort distances in ascending order
    distances.sort()

    drone_idx = 0
    for dist, i, j in distances:
        if not assigned_buildings[i] and not assigned_buildings[j]:
            if drone_idx < num_drones:
                drone_waypoints[drone_idx].append(building_positions[i])
                drone_waypoints[drone_idx].append(building_positions[j])
                assigned_buildings[i] = True
                assigned_buildings[j] = True
                drone_idx += 1
            else:
                # All drones assigned, break early
                break
    
    # Fallback for any unassigned buildings if the greedy approach didn't fill all drones
    # This should ideally not be hit if there are exactly 10 buildings and 5 drones
    unassigned_indices = [idx for idx, assigned in enumerate(assigned_buildings) if not assigned]
    if unassigned_indices:
        # Distribute remaining unassigned buildings to drones that have less than 2 waypoints
        current_unassigned_idx = 0
        for i in range(num_drones):
            while len(drone_waypoints[i]) < 2 and current_unassigned_idx < len(unassigned_indices):
                drone_waypoints[i].append(building_positions[unassigned_indices[current_unassigned_idx]])
                current_unassigned_idx += 1

    return drone_waypoints
