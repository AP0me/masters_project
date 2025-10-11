import random
from copy import deepcopy
from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np

# Constants
POPULATION_SIZE = 1000
MIN_GENE_LENGTH = 50
MAX_GENE_LENGTH = 1000
GENERATIONS = 500
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 10
MAZE_SIZE = 161  # Change this value to adjust maze size

# Directions: 0=Up, 1=Down, 2=Left, 3=Right
DIRECTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1) 
}

def setup_maze(grid_size=MAZE_SIZE):
    """Initialize maze with proper dimensions."""
    maze = Maze()
    maze.generator = Prims(grid_size, grid_size)
    maze.generate()
    # Start position is always (1, 1) - one step into the maze from top-left corner
    maze.start = (1, 1)
    # End position is always (size-2, size-2) - one step away from bottom-right corner
    maze.end = (maze.grid.shape[0]-2, maze.grid.shape[1]-2)
    return maze

def execute_gene(maze, gene_vector):
    """Simulate agent movement with accurate wall collision."""
    x, y = maze.start
    path = [(x, y)]
    
    for gene in gene_vector:
        if gene not in DIRECTIONS:
            continue
            
        dx, dy = DIRECTIONS[gene]
        new_x, new_y = x + dx, y + dy
        
        # Check boundaries and walls (0=wall, 1=path)
        if (0 <= new_x < maze.grid.shape[0] and 
            0 <= new_y < maze.grid.shape[1] and 
            maze.grid[new_x][new_y] == 0):
            x, y = new_x, new_y
            path.append((x, y))
            
            if (x, y) == maze.end:
                break
                
    return path, (x, y)

def steps_to_exit(maze, position):
    """Calculate the minimum number of steps from position to maze exit using BFS.
    
    Args:
        maze: Maze object containing the grid
        position: Tuple (x, y) of the starting position
    
    Returns:
        int: Minimum steps to exit (or -1 if unreachable)
    """
    from collections import deque
    
    # Check if already at exit
    if position == maze.end:
        return 0
    
    visited = set()
    queue = deque()
    queue.append((position[0], position[1], 0))  # (x, y, steps)
    
    while queue:
        x, y, steps = queue.popleft()
        
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            
            # Check if new position is valid and not visited
            if (0 <= nx < maze.grid.shape[0] and 
                0 <= ny < maze.grid.shape[1] and 
                maze.grid[nx][ny] == 0 and  # 0 represents path in mazelib
                (nx, ny) not in visited):
                
                # Found the exit
                if (nx, ny) == maze.end:
                    return steps + 1
                
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
    
    # Exit not reachable from this position
    return -1


def evaluate_fitness(path, final_position, end_position, gene_length):
    """Calculate fitness score (higher is better)."""
    # Manhattan distance from final position to end
    distance = abs(final_position[0] - end_position[0]) + abs(final_position[1] - end_position[1])
    # Adjust the fitness calculation based on maze size
    # Penalize distance more heavily but reward shorter paths

    maze_size_factor = max(MAZE_SIZE, 10)  # Ensure we have a reasonable base
    return (maze_size_factor * 10 - distance) - (gene_length * 0.1)

def initialize_population(pop_size):
    """Create initial random population with variable gene lengths."""
    # Adjust gene length bounds based on maze size
    base_min_length = MIN_GENE_LENGTH
    base_max_length = MAX_GENE_LENGTH
    # Scale gene lengths with maze size (longer for larger mazes)
    scaled_min = min(base_min_length, base_max_length)
    scaled_max = max(base_min_length + MAZE_SIZE * 5, base_max_length)
    
    return [[random.randint(0, 3) for _ in range(random.randint(scaled_min, scaled_max))] 
            for _ in range(pop_size)]

def select_parents(population, fitness_scores):
    """Select two parents using tournament selection."""
    parents = []
    for _ in range(2):
        candidates = random.sample(list(zip(population, fitness_scores)), TOURNAMENT_SIZE)
        winner = max(candidates, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents[0], parents[1]

def crossover(parent1, parent2):
    """Perform one-point crossover with variable-length genes."""
    if random.random() > CROSSOVER_RATE or len(parent1) < 2 or len(parent2) < 2:
        return deepcopy(parent1), deepcopy(parent2)
    
    point1 = random.randint(1, len(parent1) - 1)
    point2 = random.randint(1, len(parent2) - 1)
    
    child1 = parent1[:point1] + parent2[point2:]
    child2 = parent2[:point2] + parent1[point1:]
    
    # Ensure children stay within length bounds
    scaled_max = MAX_GENE_LENGTH + MAZE_SIZE * 5
    child1 = child1[:scaled_max]
    child2 = child2[:scaled_max]
    
    # Ensure minimum length
    scaled_min = MIN_GENE_LENGTH
    if len(child1) < scaled_min:
        child1 += [random.randint(0, 3) for _ in range(scaled_min - len(child1))]
    if len(child2) < scaled_min:
        child2 += [random.randint(0, 3) for _ in range(scaled_min - len(child2))]
        
    return child1, child2

def mutation(individual):
    """Mutate genes with given probability, including normally distributed gene length changes."""
    mutated = deepcopy(individual)
    
    # Gene-wise mutation
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            mutated[i] = random.randint(0, 3)
    
    # Length mutation - Normally distributed gene length changes
    if random.random() < 0.1:
        gene_change = round(random.normalvariate(0, 5))
        new_length = len(mutated) + gene_change
        
        # Use maze-size-adjusted bounds
        scaled_min = MIN_GENE_LENGTH
        scaled_max = MAX_GENE_LENGTH + MAZE_SIZE * 5
        new_length = max(scaled_min, min(new_length, scaled_max))
        
        if new_length > len(mutated):
            for _ in range(new_length - len(mutated)):
                mutated.append(random.randint(0, 3))
        elif new_length < len(mutated):
            del mutated[new_length:]
    
    return mutated
    
def genetic_algorithm(maze):
    """Run genetic algorithm to solve maze."""
    population = initialize_population(POPULATION_SIZE)
    best_fitness = -float('inf')
    best_individual = None
    best_path = None
    
    for generation in range(GENERATIONS):
        fitness_scores = []
        paths = []
        
        for individual in population:
            path, final_pos = execute_gene(maze, individual)
            score = evaluate_fitness(path, final_pos, maze.end, len(individual))
            fitness_scores.append(score)
            
            if score > best_fitness:
                best_fitness = score
                best_individual = deepcopy(individual)
                best_path = path
        
        new_population = []
        
        # Elitism - keep the best individual
        if best_individual:
            new_population.append(best_individual)
        
        # Generate new population
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1)
            child2 = mutation(child2)
            new_population.extend([child1, child2])
        
        population = new_population[:POPULATION_SIZE]
        
        if generation % 10 == 0:
            final_pos = best_path[-1] if best_path else (0, 0)
            distance = abs(final_pos[0] - maze.end[0]) + abs(final_pos[1] - maze.end[1])
            print(f"Generation {generation}: Fitness={best_fitness:.1f}, "
                  f"Length={len(best_individual) if best_individual else 0}, "
                  f"Position={final_pos}, Distance={distance}")
    
    return best_individual, best_path

def visualize_maze(maze, path=None):
    """Proper visualization accounting for matrix coordinates."""
    plt.figure(figsize=(max(MAZE_SIZE/2, 8), max(MAZE_SIZE/2, 8)))
    
    plt.imshow(maze.grid, cmap='binary')
    
    if path:
        y_coords = [p[1] for p in path]
        x_coords = [p[0] for p in path]
        plt.plot(y_coords, x_coords, 'r-', linewidth=2)
    
    plt.plot(maze.start[1], maze.start[0], 'go', markersize=8)
    plt.plot(maze.end[1], maze.end[0], 'bs', markersize=8)
    
    plt.gca().invert_yaxis()
    plt.title(f"Maze Size: {maze.grid.shape[0]}x{maze.grid.shape[1]}")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Configurable maze size (must be odd number for maze generation)
    print(f"Configuring maze with size: {MAZE_SIZE}")
    
    if MAZE_SIZE % 2 == 0:
        print(f"Warning: Maze size should be odd for proper generation. Using {MAZE_SIZE + 1}")
        actual_size = MAZE_SIZE + 1
    else:
        actual_size = MAZE_SIZE
    
    maze = setup_maze(actual_size)
    
    print(f"Maze size: {maze.grid.shape}")
    print(f"Start: {maze.start}, End: {maze.end}")
    
    # Run the genetic algorithm
    best_solution, best_path = genetic_algorithm(maze)
    
    # Visualize the best solution
    print("\nBest solution:")
    print(f"Path length: {len(best_path)}")
    print(f"Gene length: {len(best_solution)}")
    print(f"Final position: {best_path[-1]}")
    print(f"Reached goal: {best_path[-1] == maze.end}")
    print(f"Distance to goal: {abs(best_path[-1][0] - maze.end[0]) + abs(best_path[-1][1] - maze.end[1])}")
    
    visualize_maze(maze, best_path)