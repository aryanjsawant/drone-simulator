# DroneSim: A Pathfinding Algorithm Comparison Platform

## Explore and Compare Pathfinding Algorithms in a Simulated Environment

DroneSim is a simulation platform built with Panda3D, designed to help you understand and compare various pathfinding algorithms in a dynamic, 3D environment. While capable of simulating different vehicle types, its core focus is on visualizing how algorithms like A*, Dijkstra, BFS, and DFS navigate complex spaces.

### Key Aspects of DroneSim:

- **Algorithm Comparison**: Easily switch between and observe the performance of different pathfinding algorithms.
- **Interactive Simulation**: Visualize drone movement and path generation in real-time.
- **Sensor Integration**: Basic sensor models (like cameras and IMU) are included to provide context for environmental interaction.
- **Extensible Design**: The modular structure allows for further development and integration of new algorithms or vehicle models.

---

## Features:

- **Multiple Pathfinding Algorithms**: Supports A*, Dijkstra, Breadth-First Search (BFS), and Depth-First Search (DFS).
- **3D Environment**: Navigate through simulated environments with obstacles and buildings.
- **Vehicle Simulation**: Includes a basic drone model for path execution.
- **Real-time Visualization**: See paths being generated and followed.

---

## Installation:

Install the package using pip:

```bash
pip install git+https://github.com/nb-programmer/dronesim.git
```

---

## Usage: Running Simulations and Comparing Algorithms

To run the default simulator, which launches a window with keyboard controls and a scene, use:

```bash
$ dronesim
```

Alternatively, execute the package using the Python interpreter:

```bash
$ py -m dronesim
```

### Comparing Pathfinding Algorithms:

The main purpose of DroneSim is to compare how different pathfinding algorithms perform. You can specify the algorithm to use when launching the simulator using the `--algo` argument:

```bash
$ dronesim --algo a_star
$ dronesim --algo dijkstra
$ dronesim --algo bfs
$ dronesim --algo dfs
```

After launching with a chosen algorithm, press `c` in the simulator window to trigger the autonomous flight based on the selected pathfinding algorithm. A visualization of the generated path will be saved as `path.png` in the current working directory.

Refer to the `examples/` folder (available after cloning this repository) for more advanced usage and custom controller implementations.

---

## Controls:

### Simulator Controls:

| Key       | Action                                     |
| :-------- | :----------------------------------------- |
| `Esc`     | Unlock/Lock and Show/Hide mouse            |
| `F1`      | Toggle visibility of HUD                   |
| `F3`      | Toggle visibility of debug view            |
| `Shift+F3`| Connect to Panda3D's PStats tool for profiling |
| `F5`      | Change camera mode (Free, First Person, Third Person) |
| `F6`      | Toggle control between camera and vehicle  |
| `F8`      | Toggle wireframe render                    |
| `F11`     | Toggle fullscreen                          |
| `v`       | Show/hide all buffers                      |
| ``\``    | Dump current simulator state to console    |

### Default Drone/Camera Controls:

| Key         | Action                                   |
| :---------- | :--------------------------------------- |
| `I`         | Take off                                 |
| `K`         | Land                                     |
| `W`         | Move forwards                            |
| `S`         | Move backwards                           |
| `A`         | Move left                                |
| `D`         | Move right                               |
| `Up arrow`  | Increase altitude                        |
| `Down arrow`| Decrease altitude                        |
| `Left arrow`| Heading counter-clockwise                |
| `Right arrow`| Heading clockwise                       |
| `Mouse wheel`| Free cam: change fly speed; Third Person cam: Change orbit radius |