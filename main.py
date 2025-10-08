from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

def show_png(grid):
    """Generate a simple image of the maze."""
    numerical_grid = np.array(grid, dtype=np.uint8)
    plt.figure(figsize=(10, 5))
    plt.imshow(numerical_grid, interpolation='nearest')
    plt.show()


maze = Maze(1122)
maze.generator = Prims(16, 16)
maze.generate()
maze.end = (1, 1)
maze.start = (31, 31)

gene_vector = random.randint(4, size=1000)


def execute_gene(maze, gene_vector):
    """Generate a simple image of the maze."""
    agent_position = list(maze.start)
    position_history = []

    for gene in gene_vector:
        tries_to_go_to = []
        if gene == 0:
            tries_to_go_to = [agent_position[0], agent_position[1] - 1]
        if gene == 1:
            tries_to_go_to = [agent_position[0], agent_position[1] + 1]
        if gene == 2:
            tries_to_go_to = [agent_position[0] - 1, agent_position[1]]
        if gene == 3:
            tries_to_go_to = [agent_position[0] + 1, agent_position[1]]

        maze_cell = maze.grid[tries_to_go_to[0]][tries_to_go_to[1]]
        if maze_cell == "#":
            continue
        else:
            if tries_to_go_to[0] > 0 \
            and tries_to_go_to[1] > 0 \
            and tries_to_go_to[0] < 32 \
            and tries_to_go_to[1] < 32:
                agent_position = tries_to_go_to
                position_history.append(agent_position)
            else:
                continue

    return position_history

agent_position_history = execute_gene(maze, gene_vector)
print(agent_position_history)
show_png(maze.grid)
