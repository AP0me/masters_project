from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np

def showPNG(grid):
    """Generate a simple image of the maze."""
    numerical_grid = np.array(grid, dtype=np.uint8) 
    
    plt.figure(figsize=(10, 5))
    plt.imshow(numerical_grid, cmap=plt.cm.binary, interpolation='nearest') 
    plt.xticks([]), plt.yticks([])
    plt.show()


m = Maze()
m.generator = Prims(16, 16)
m.generate()
showPNG(m.grid)
