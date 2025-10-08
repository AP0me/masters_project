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
    """Simulate agent movement through a maze based on a gene vector.
    
    Args:
        maze: Maze object containing start position and grid
        gene_vector: List of movement commands (0: up, 1: down, 2: left, 3: right)
    
    Returns:
        List of visited positions
    """
    # Constants for movement directions
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    MOVEMENTS = {
        UP: (0, -1),
        DOWN: (0, 1),
        LEFT: (-1, 0),
        RIGHT: (1, 0)
    }
    GRID_SIZE = 32  # Assuming 32x32 grid from your original code
    
    agent_position = list(maze.start)
    position_history = []

    for gene in gene_vector:
        if gene not in MOVEMENTS:
            continue
            
        dx, dy = MOVEMENTS[gene]
        new_x, new_y = agent_position[0] + dx, agent_position[1] + dy
        
        if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
            continue
        
        if maze.grid[new_x][new_y] != "#":
            agent_position = [new_x, new_y]
            position_history.append(list(agent_position))  # Append a copy to avoid mutation

    return position_history
agent_position_history = execute_gene(maze, gene_vector)
print(agent_position_history)
show_png(maze.grid)
