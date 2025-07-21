import os
from dronesim.algorithms.utils import plot_path, read_occupancy_map

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def dfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    came_from = {start: None}
    visited = set()

    while stack:
        current = stack.pop()
        if current == goal:
            break

        visited.add(current)

        for d in DIRS:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if (
                0 <= neighbor[0] < rows and
                0 <= neighbor[1] < cols and
                grid[neighbor[0]][neighbor[1]] == 0 and
                neighbor not in visited and
                neighbor not in came_from
            ):
                stack.append(neighbor)
                came_from[neighbor] = current

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
        plot_path(grid, path, image_output_path, title="DFS Path")
        print(f"Image saved to {image_output_path}")

    return path

if __name__ == "__main__":
    base_dir = ""
    map_path = os.path.join(base_dir, "occupancy_map.csv")
    image_output_path = os.path.join(base_dir, "path.png")

    grid = read_occupancy_map(map_path)
    start = (3, 4)
    goal = (5, 7)

    path = dfs(grid, start, goal)

    if path:
        print("DFS Path found:", path)
    else:
        print("No path found using DFS.")

    plot_path(grid, path, image_output_path, title="DFS Path")
    print(f"Image saved to {image_output_path}")
