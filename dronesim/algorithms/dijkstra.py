import heapq
import os
from dronesim.algorithms.utils import plot_path, read_occupancy_map

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def dijkstra(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    cost_so_far = {start: 0}
    came_from = {}
    frontier = [(0, start)]

    while frontier:
        cost, current = heapq.heappop(frontier)
        if current == goal:
            break

        for d in DIRS:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
                new_cost = cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []
    path.append(start)
    path.reverse()

    if path == []:
        return path
    else:
        base_dir = ""
        image_output_path = os.path.join(base_dir, "path.png")
        plot_path(grid, path, image_output_path, title="Dijkstra Path")
        print(f"Image saved to {image_output_path}")

    return path

if __name__ == "__main__":
    base_dir = ""
    map_path = os.path.join(base_dir, "occupancy_map.csv")
    image_output_path = os.path.join(base_dir, "path.png")

    grid = read_occupancy_map(map_path)
    start = (3, 4)
    goal = (5, 7)

    path = dijkstra(grid, start, goal)

    if path:
        print("Dijkstra Path found:", path)
    else:
        print("No path found using Dijkstra.")

    plot_path(grid, path, image_output_path, title="Dijkstra Path")
    print(f"Image saved to {image_output_path}")
