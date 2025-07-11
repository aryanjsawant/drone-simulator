import os
import csv
import heapq
import matplotlib.pyplot as plt
import numpy as np

# Define movement directions (4-connectivity)
DIRS = [(0,1), (1,0), (0,-1), (-1,0)]

def read_occupancy_map(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        grid = [[int(cell) for cell in row] for row in reader]
    return grid

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = [(0 + heuristic(start, goal), 0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            break

        for d in DIRS:
            neighbor = (current[0] + d[0], current[1] + d[1])

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue  # obstacle

                new_cost = cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []  # No path found
    path.append(start)
    path.reverse()
    return path

def plot_path(grid, path, output_path):
    grid = np.array(grid)
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot obstacles
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1:
                plt.text(c, r, 'X', color='red', ha='center', va='center', fontsize=16)

    # Plot path
    if path:
        x_coords = [p[1] for p in path]
        y_coords = [p[0] for p in path]
        plt.plot(x_coords, y_coords, color='green', linewidth=2, marker='o')

    plt.xlim(-0.5, grid.shape[1]-0.5)
    plt.ylim(-0.5, grid.shape[0]-0.5)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.title('A* Pathfinding')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    base_dir = "cal"
    map_path = os.path.join(base_dir, "occupancy_map.csv")
    image_output_path = os.path.join(base_dir, "a_star_path.png")

    grid = read_occupancy_map(map_path)
    start = (4, 3)
    goal = (5, 7)  # because grid has only 6 rows: index 0-5
    path = a_star(grid, start, goal)
    
    if not path:
        print("No path found.")
    else:
        print("Path found:", path)

    plot_path(grid, path, image_output_path)
    print(f"Image saved to {image_output_path}")
