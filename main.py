import random
from collections import deque
from copy import deepcopy
from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np

# Constants
MIN_GENE_LENGTH = 50
MAX_GENE_LENGTH = 1000
GENERATIONS = 200
CROSSOVER_RATE = 0.7
MAZE_SIZE = 129  # Change this value to adjust maze size
POPULATION_SIZE = (MAZE_SIZE * MAZE_SIZE) / 16
TOURNAMENT_SIZE = POPULATION_SIZE * 50//100
MUTATION_TIP_LENGTH = 30
NUM_SHADOWS = 5

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
    maze.start = (1, 1)
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
    """Calculate the minimum number of steps from position to maze exit using BFS."""
    if position == maze.end:
        return 0
    
    visited = set([position])  # Start with initial position visited
    queue = deque([(position[0], position[1], 0)])
    
    while queue:
        x, y, steps = queue.popleft()
        
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            
            # Early boundary check
            if not (0 <= nx < maze.grid.shape[0] and 0 <= ny < maze.grid.shape[1]):
                continue
            
            # Check if we found the exit
            if (nx, ny) == maze.end:
                return steps + 1
            
            # Check if valid path and not visited
            if maze.grid[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
    
    return -1

def evaluate_fitness(maze, final_position, gene_length, path):
    """Calculate fitness based on exact steps remaining to exit, with backtracking penalty."""
    x, y = final_position
    
    # Calculate backtracking penalty
    unique_positions = len(set(path))
    backtrack_penalty = (len(path) - unique_positions) / len(path)
    
    # Full path completion gives maximum score
    if (x, y) == maze.end:
        return 1000 - (gene_length * 0.1) - backtrack_penalty
    
    steps = steps_to_exit(maze, (x, y))
    
    if steps == -1:
        return -1000 - backtrack_penalty
    
    max_possible_steps = 3 * MAZE_SIZE  # Approximate worst-case scenario
    scaled_fitness = max_possible_steps - steps
    
    return scaled_fitness - (gene_length * 0.1) - backtrack_penalty

def initialize_population(pop_size):
    """Create initial random population with variable gene lengths."""
    scaled_min = min(MIN_GENE_LENGTH, MAX_GENE_LENGTH)
    scaled_max = max(MIN_GENE_LENGTH + MAZE_SIZE * 5, MAX_GENE_LENGTH)
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

def mutation(individual):
    """Mutate the last 30 genes by replacing them with new random values."""
    mutated = deepcopy(individual)
    # Only perform mutation if individual has at least 30 elements
    if len(mutated) >= 30:
        # Create new random array of size 30
        new_segment = np.random.randint(0, 4, size=30)  # 0 to 3 inclusive
        # Replace the last 30 elements
        mutated[-30:] = new_segment
    return mutated

def crossover(parent1, parent2):
    """Perform one-point crossover with variable-length genes."""
    if random.random() > CROSSOVER_RATE or len(parent1) < 2 or len(parent2) < 2:
        return deepcopy(parent1), deepcopy(parent2)
    
    # Ensure at least one element remains from each parent
    point1 = random.randint(len(parent1)//2, len(parent1) - 1)
    point2 = random.randint(len(parent2)//2, len(parent2) - 1)
    
    child1 = parent1[:point1] + parent2[point2:]
    child2 = parent2[:point2] + parent1[point1:]
    
    scaled_max = MAX_GENE_LENGTH + MAZE_SIZE * 5
    child1 = child1[:scaled_max]
    child2 = child2[:scaled_max]
    
    scaled_min = MIN_GENE_LENGTH
    if len(child1) < scaled_min:
        child1 += [random.randint(0, 3) for _ in range(scaled_min - len(child1))]
    if len(child2) < scaled_min:
        child2 += [random.randint(0, 3) for _ in range(scaled_min - len(child2))]
        
    return child1, child2

def genetic_algorithm(maze):
    """Run genetic algorithm to solve maze with visualization of other candidates."""
    population = initialize_population(POPULATION_SIZE)
    best_fitness = -float('inf')
    best_individual = None
    best_path = None
    
    # Set up plot
    plt.figure(figsize=(max(MAZE_SIZE/2, 8), max(MAZE_SIZE/2, 8)))
    
    for generation in range(GENERATIONS):
        fitness_scores = []
        paths = []
        all_individuals = []  # Store all individuals for visualization
        
        for individual in population:
            path, final_pos = execute_gene(maze, individual)
            score = evaluate_fitness(maze, final_pos, len(individual), path)
            fitness_scores.append(score)
            paths.append(path)
            all_individuals.append(individual)
            
            if score > best_fitness:
                best_fitness = score
                best_individual = deepcopy(individual)
                best_path = path
        
        new_population = []
        
        # Elitism
        if best_individual:
            new_population.append(best_individual)
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1)
            child2 = mutation(child2)
            new_population.extend([child1, child2])
        
        population = new_population[:POPULATION_SIZE]
        
        if generation % 10 == 0:
            final_pos = best_path[-1] if best_path else (0, 0)
            steps = steps_to_exit(maze, final_pos) if final_pos != maze.end else 0
            
            # Calculate backtracking for best solution
            unique_positions = len(set(best_path))
            backtrack_count = len(best_path) - unique_positions
            
            print(f"Gen {generation}: Fit={best_fitness:.1f}, Len={len(best_individual)}, "
                  f"Steps left={steps}, Backtrack={backtrack_count}")

            # Visualize current state
            plt.clf()  # Clear the current figure
            plt.imshow(maze.grid, cmap='binary')
            
            # Plot shadows of other candidates (random sample)
            shadow_indices = random.sample(range(len(paths)), NUM_SHADOWS)
            
            for i in shadow_indices:
                if paths[i] and len(paths[i]) > 1:  # Ensure valid path
                    y_coords_shadow = [p[1] for p in paths[i]]
                    x_coords_shadow = [p[0] for p in paths[i]]
                    plt.plot(y_coords_shadow, x_coords_shadow, 'gray', 
                             alpha=0.5, linewidth=1, linestyle='-')
            
            # Plot genetic algorithm's best path (red)
            if best_path:
                y_coords = [p[1] for p in best_path]
                x_coords = [p[0] for p in best_path]
                plt.plot(y_coords, x_coords, 'r-', linewidth=3, alpha=0.8, label='GA Best Solution')
            
            # Plot actual shortest path from final position to exit (blue)
            if best_path and best_path[-1] != maze.end:
                final_pos = best_path[-1]
                optimal_path = find_optimal_path(maze, final_pos)
                if optimal_path:
                    y_coords_opt = [p[1] for p in optimal_path]
                    x_coords_opt = [p[0] for p in optimal_path]
                    plt.plot(y_coords_opt, x_coords_opt, 'b-', linewidth=2, alpha=0.7,
                             label='Optimal Path Remaining')
            
            # Start and end markers
            plt.plot(maze.start[1], maze.start[0], 'go', markersize=10, label='Start')
            plt.plot(maze.end[1], maze.end[0], 'bs', markersize=10, label='End')
            
            plt.gca().invert_yaxis()
            plt.title(f"Gen {generation}: Maze Size {maze.grid.shape[0]}x{maze.grid.shape[1]}\n"
                     f"Showing {NUM_SHADOWS} candidate shadows")
            plt.legend()
            plt.draw()
            plt.pause(0.1)  # Small pause to update the plot
    
    return best_individual, best_path

def visualize_maze(maze, path=None):
    """Visualization with genetic solution (red) and actual optimal path from final position (blue)."""
    plt.figure(figsize=(max(MAZE_SIZE/2, 8), max(MAZE_SIZE/2, 8)))
    plt.imshow(maze.grid, cmap='binary')
    
    # Plot genetic algorithm's path (red)
    if path:
        y_coords = [p[1] for p in path]
        x_coords = [p[0] for p in path]
        plt.plot(y_coords, x_coords, 'r-', linewidth=2, label='GA Solution')
    
    # Plot actual shortest path from final position to exit (blue)
    if path and path[-1] != maze.end:
        final_pos = path[-1]
        optimal_path = find_optimal_path(maze, final_pos)
        if optimal_path:
            y_coords_opt = [p[1] for p in optimal_path]
            x_coords_opt = [p[0] for p in optimal_path]
            plt.plot(y_coords_opt, x_coords_opt, 'b-', linewidth=2, label='Optimal Path Remaining')
    
    # Start and end markers
    plt.plot(maze.start[1], maze.start[0], 'go', markersize=8, label='Start')
    plt.plot(maze.end[1], maze.end[0], 'bs', markersize=8, label='End')
    
    plt.gca().invert_yaxis()
    plt.title(f"Maze Size: {maze.grid.shape[0]}x{maze.grid.shape[1]}")
    plt.legend()
    plt.show()

def find_optimal_path(maze, start_pos):
    """Find the actual shortest path from a start position to the maze exit using BFS."""
    if start_pos == maze.end:
        return [start_pos]
    
    visited = {start_pos: None}  # Store parent information for path reconstruction
    queue = deque([start_pos])
    
    while queue:
        current = queue.popleft()
        
        if current == maze.end:
            # Reconstruct path from end to start
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            return path[::-1]  # Reverse to get start to end
        
        x, y = current
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            
            # Check boundaries and valid path
            if (0 <= nx < maze.grid.shape[0] and 
                0 <= ny < maze.grid.shape[1] and 
                maze.grid[nx][ny] == 0 and 
                (nx, ny) not in visited):
                
                visited[(nx, ny)] = current
                queue.append((nx, ny))
    
    return None  # No path found

if __name__ == "__main__":
    random.seed(10)
    np.random.seed(10)
    actual_size = MAZE_SIZE if MAZE_SIZE % 2 else MAZE_SIZE + 1
    print(f"Using maze size: {actual_size}")
    
    maze = setup_maze(actual_size)
    print(f"Maze size: {maze.grid.shape}")
    
    best_solution, best_path = genetic_algorithm(maze)
    
    print("\nBest solution:")
    print(f"Path length: {len(best_path)}")
    print(f"Gene length: {len(best_solution)}")
    print(f"Final position: {best_path[-1]}")
    print(f"Reached goal: {best_path[-1] == maze.end}")
    
    visualize_maze(maze, best_path)