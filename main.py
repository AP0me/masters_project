from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt

def showPNG(grid):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()

m = Maze()
m.generator = Prims(27, 34)
m.generate()

print(m)
