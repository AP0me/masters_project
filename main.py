import os
import random
from collections import deque
from copy import deepcopy
from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np

# Constants
GENERATIONS = 500
CROSSOVER_RATE = 0.7
NUM_SHADOWS = 5
MAZE_SIZE = 128  # Change this value to adjust maze size
POPULATION_SIZE = (MAZE_SIZE * MAZE_SIZE) // 16
TOURNAMENT_SIZE = POPULATION_SIZE * 50//100
MIN_GENE_LENGTH = MAZE_SIZE
MAX_GENE_LENGTH = 3 * MAZE_SIZE
MUTATION_TIP_LENGTH = 10
IMAGE_SAVE_PATH = "/home/apome/Desktop/Anar_Computer/split_desktop/x/DSnAI/masters_project/pictures"  # Change this to your desired path

# Directions: 0=Up, 1=Down, 2=Left, 3=Right
DIRECTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1) 
}

# Create the directory if it doesn't exist
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

def setup_maze(grid_size=MAZE_SIZE):
    """Initialize maze with proper dimensions."""
    maze = Maze()
    maze.generator = Prims(grid_size, grid_size)
    maze.generate()
    maze.start = (1, 1)
    maze.end = (maze.grid.shape[0]-2, maze.grid.shape[1]-2)
    return maze

def find_optimal_path(maze, start_pos=None):
    """Find the actual shortest path from a start position to the maze exit using BFS.
    If no start_pos is provided, use maze.start."""
    if start_pos is None:
        start_pos = maze.start
    
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

def create_optimal_distance_map(maze):
    """Create a dictionary mapping each cell in the optimal path to steps remaining to exit."""
    optimal_path = find_optimal_path(maze, maze.start)
    if optimal_path is None:
        return {}
    
    distance_map = {}
    total_length = len(optimal_path)
    for i, cell in enumerate(optimal_path):
        distance_map[cell] = total_length - i - 1
    
    return distance_map

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

def steps_to_exit(maze, position, distance_map):
    """Calculate the minimum number of steps from position to maze exit using BFS.
    Uses precomputed distance map to speed up computation when on optimal path."""
    if position == maze.end:
        return 0
    
    # Check if position is in the precomputed optimal path
    if position in distance_map:
        return distance_map[position]
    
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
            
            # Check if cell is on optimal path (use precomputed map)
            if (nx, ny) in distance_map:
                return steps + 1 + distance_map[(nx, ny)]
            
            # Check if valid path and not visited
            if maze.grid[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
    
    return -1

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
    """Mutate the individual by adding, removing, or replacing genes."""
    mutated = deepcopy(individual)
    
    # Choose mutation type: 0=replace, 1=add, 2=remove
    mutation_type = random.randint(0, 2)
    
    if mutation_type == 0:  # Replace mutation (original behavior)
        # Only perform mutation if individual has at least MUTATION_TIP_LENGTH elements
        if len(mutated) >= MUTATION_TIP_LENGTH:
            # Create new random array of size MUTATION_TIP_LENGTH
            new_segment = np.random.randint(0, 4, size=MUTATION_TIP_LENGTH)  # 0 to 3 inclusive
            # Replace the last MUTATION_TIP_LENGTH elements
            mutated[-MUTATION_TIP_LENGTH:] = new_segment
    
    elif mutation_type == 1:  # Add mutation
        # Add random number of new genes (1 to MUTATION_TIP_LENGTH)
        num_to_add = random.randint(1, MUTATION_TIP_LENGTH)
        new_genes = np.random.randint(0, 4, size=num_to_add)
        # Add at random position
        position = random.randint(0, len(mutated))
        mutated = mutated[:position] + list(new_genes) + mutated[position:]
    
    elif mutation_type == 2:  # Remove mutation
        # Remove random number of genes (1 to MUTATION_TIP_LENGTH)
        if len(mutated) > MIN_GENE_LENGTH:  # Ensure we don't go below minimum length
            num_to_remove = random.randint(1, min(MUTATION_TIP_LENGTH, len(mutated) - MIN_GENE_LENGTH))
            # Remove from random position
            position = random.randint(0, len(mutated) - num_to_remove)
            mutated = mutated[:position] + mutated[position + num_to_remove:]
    
    # Ensure gene length stays within bounds
    scaled_max = MAX_GENE_LENGTH + MAZE_SIZE * 5
    if len(mutated) > scaled_max:
        mutated = mutated[:scaled_max]
    
    scaled_min = MIN_GENE_LENGTH
    if len(mutated) < scaled_min:
        mutated += [random.randint(0, 3) for _ in range(scaled_min - len(mutated))]
    
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

def evaluate_fitness_with_distance_map(maze, final_position, gene_length, path, distance_map):
    """Calculate fitness using the optimized steps_to_exit function."""
    x, y = final_position
    
    # Calculate backtracking penalty
    unique_positions = len(set(path))
    backtrack_penalty = (len(path) - unique_positions) / len(path)
    
    # Full path completion gives maximum score
    if (x, y) == maze.end:
        return 200 - (gene_length * 0.1) - backtrack_penalty
    
    steps = steps_to_exit(maze, (x, y), distance_map)
    
    if steps == -1:
        return -200 - backtrack_penalty
    
    max_possible_steps = 3 * MAZE_SIZE
    scaled_fitness = max_possible_steps - steps
    
    return scaled_fitness - (gene_length * 0.1) - backtrack_penalty

def genetic_algorithm(maze):
    """Run genetic algorithm to solve maze with visualization of other candidates and save images."""
    # Precompute the optimal distance map
    distance_map = create_optimal_distance_map(maze)
    
    population = initialize_population(POPULATION_SIZE)
    best_fitness = -float('inf')
    best_individual = None
    best_path = None
    
    # Track fitness statistics across generations
    best_fitness_history = []
    avg_fitness_history = []
    
    # Set up plot
    plt.figure(figsize=(max(MAZE_SIZE/2, 8), max(MAZE_SIZE/2, 8)))
    
    for generation in range(GENERATIONS):
        fitness_scores = []
        paths = []
        all_individuals = []
        
        for individual in population:
            path, final_pos = execute_gene(maze, individual)
            # Use the optimized steps_to_exit with distance_map
            score = evaluate_fitness_with_distance_map(maze, final_pos, len(individual), path, distance_map)
            fitness_scores.append(score)
            paths.append(path)
            all_individuals.append(individual)
            
            if score > best_fitness:
                best_fitness = score
                best_individual = deepcopy(individual)
                best_path = path
        
        # Track statistics for this generation
        current_best_fitness = max(fitness_scores)
        current_avg_fitness = sum(fitness_scores) / len(fitness_scores)
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(current_avg_fitness)
        
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
            steps = steps_to_exit(maze, final_pos, distance_map) if final_pos != maze.end else 0
            
            # Calculate backtracking for best solution
            unique_positions = len(set(best_path))
            backtrack_count = len(best_path) - unique_positions
            
            print(f"Gen {generation}: BestFit={current_best_fitness:.1f}, AvgFit={current_avg_fitness:.1f}, "
                  f"Len={len(best_individual)}, Steps left={steps}, Backtrack={backtrack_count}")

            # Visualize current state
            plt.clf()  # Clear the current figure
            plt.imshow(maze.grid, cmap='binary')
            
            # Plot shadows of other candidates (random sample)
            shadow_indices = random.sample(range(len(paths)), NUM_SHADOWS)
            
            for i in shadow_indices:
                if paths[i] and len(paths[i]) > 1:  # Ensure valid path
                    y_coords_shadow = [p[1] for p in paths[i]]
                    x_coords_shadow = [p[0] for p in paths[i]]
                    plt.plot(y_coords_shadow, x_coords_shadow, 'green', alpha=1, linewidth=3, linestyle='-')
            
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
            
            # Save the image
            filename = os.path.join(IMAGE_SAVE_PATH, f"generation_{generation:04d}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved image: {filename}")
            
            plt.draw()
            plt.pause(0.1)  # Small pause to update the plot
    
    # Create fitness progression plot at the end
    plt.figure(figsize=(10, 6))
    plt.plot(range(GENERATIONS), best_fitness_history, 'r-', label='Best Fitness', linewidth=2)
    plt.plot(range(GENERATIONS), avg_fitness_history, 'b-', label='Average Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression Over Generations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save fitness progression plot
    fitness_filename = os.path.join(IMAGE_SAVE_PATH, "fitness_progression.png")
    plt.savefig(fitness_filename, dpi=150, bbox_inches='tight')
    print(f"Saved fitness progression: {fitness_filename}")
    
    plt.show()
    
    return best_individual, best_path, best_fitness_history, avg_fitness_history

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

    visualize_maze(maze, [])
    
    # Precompute distance map once
    distance_map = create_optimal_distance_map(maze)
    
    best_solution, best_path, best_history, avg_history = genetic_algorithm(maze)
    
    print("\nBest solution:")
    print(f"Path length: {len(best_path)}")
    print(f"Gene length: {len(best_solution)}")
    print(f"Final position: {best_path[-1]}")
    print(f"Reached goal: {best_path[-1] == maze.end}")
    print(f"Final best fitness: {best_history[-1]:.2f}")
    print(f"Final average fitness: {avg_history[-1]:.2f}")
    
    visualize_maze(maze, best_path)