import random
from copy import deepcopy
from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np

# Constants
POPULATION_SIZE = 100
MIN_GENE_LENGTH = 50
MAX_GENE_LENGTH = 1000
GENERATIONS = 300
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 10

# Directions: 0=Up, 1=Down, 2=Left, 3=Right
DIRECTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1) 
}

def setup_maze():
    """Initialize maze with proper dimensions."""
    maze = Maze()
    maze.generator = Prims(16, 16)
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

def evaluate_fitness(path, final_position, end_position, gene_length):
    """Calculate fitness score (higher is better) using maze solver for distance."""
    # If agent reached the end, maximum fitness
    if final_position == end_position:
        return MAX_GENE_LENGTH  # Maximum possible score
    
    # Create a temporary maze copy with the agent's position as new start
    temp_maze = deepcopy(maze)
    temp_maze.start = final_position
    
    # Use maze solver to find shortest path from final position to end
    solver = BacktrackingSolver()
    try:
        solution = solver.solve(temp_maze.grid, temp_maze.start, temp_maze.end)
        shortest_distance = len(solution) if solution else float('inf')
    except:
        shortest_distance = float('inf')
    
    # Fitness is inversely proportional to remaining distance
    # Also consider gene length (shorter solutions are better)
    fitness = (MAX_GENE_LENGTH - shortest_distance) - (len(gene) / 100)
    
    return max(0, fitness)  # Ensure fitness is non-negative

def initialize_population(pop_size):
    """Create initial random population with variable gene lengths."""
    return [[random.randint(0, 3) for _ in range(random.randint(MIN_GENE_LENGTH, MAX_GENE_LENGTH))] 
            for _ in range(pop_size)]

def select_parents(population, fitness_scores):
    """Select two parents using tournament selection."""
    parents = []
    for _ in range(2):
        candidates = random.sample(list(zip(population, fitness_scores)), TOURNAMENT_SIZE)
        winner = max(candidates, key=lambda x: x[1])[0]  # Changed to max for higher=fitter
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
    child1 = child1[:MAX_GENE_LENGTH]
    child2 = child2[:MAX_GENE_LENGTH]
    
    # Ensure minimum length
    if len(child1) < MIN_GENE_LENGTH:
        child1 += [random.randint(0, 3) for _ in range(MIN_GENE_LENGTH - len(child1))]
    if len(child2) < MIN_GENE_LENGTH:
        child2 += [random.randint(0, 3) for _ in range(MIN_GENE_LENGTH - len(child2))]
        
    return child1, child2

def mutation(individual):
    """Mutate genes with given probability, including normally distributed gene length changes."""
    mutated = deepcopy(individual)
    
    # Gene-wise mutation (keeps length constant)
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            mutated[i] = random.randint(0, 3)
    
    # Length mutation - Normally distributed gene length changes
    if random.random() < 0.1:  # 10% chance to modify gene length
        # Generate normally distributed change (mean=0, std_dev=5)
        gene_change = round(random.normalvariate(0, 5))
        
        # Apply the change
        new_length = len(mutated) + gene_change
        
        # Ensure new length stays within bounds
        new_length = max(MIN_GENE_LENGTH, min(new_length, MAX_GENE_LENGTH))
        
        # Adjust the gene sequence
        if new_length > len(mutated):
            # Add genes (randomly)
            for _ in range(new_length - len(mutated)):
                mutated.append(random.randint(0, 3))
        elif new_length < len(mutated):
            # Remove genes (randomly)
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
            
            if score > best_fitness:  # Changed to > for maximization
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
            print(f"Generation {generation}: Best fitness={best_fitness:.1f}, "
                  f"Gene length={len(best_individual) if best_individual else 0}, "
                  f"Final position={best_path[-1] if best_path else None}")
    
    return best_individual, best_path

def visualize_maze(maze, path=None):
    """Proper visualization accounting for matrix coordinates."""
    plt.figure(figsize=(10, 10))
    
    plt.imshow(maze.grid, cmap='binary')
    
    if path:
        y_coords = [p[1] for p in path]
        x_coords = [p[0] for p in path]
        plt.plot(y_coords, x_coords, 'r-', linewidth=2)
    
    plt.plot(maze.start[1], maze.start[0], 'go', markersize=10)
    plt.plot(maze.end[1], maze.end[0], 'bs', markersize=10)
    
    plt.gca().invert_yaxis()
    plt.show()

# Main execution
if __name__ == "__main__":
    maze = setup_maze()
    
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
    
    visualize_maze(maze, best_path)