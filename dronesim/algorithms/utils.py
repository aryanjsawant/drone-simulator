import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_path(grid, path, output_path, title="Pathfinding Result"):
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

    plt.xlim(-0.5, grid.shape[1] - 0.5)
    plt.ylim(-0.5, grid.shape[0] - 0.5)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def read_occupancy_map(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        grid = [[int(cell) for cell in row] for row in reader]
    return grid