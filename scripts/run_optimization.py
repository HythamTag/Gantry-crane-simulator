import numpy as np
import yaml
from multiprocessing import Pool
from src.simulation.simulator import CraneSimulation
from src.optimization.genetic_algorithm import GeneticOptimizer
from src.optimization.particle_swarm import PSOOptimizer
import csv
import os


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def update_config_load_mass(config, load_mass):
    config['crane_system']['physical_params']['masses']['load'] = load_mass
    return config


def run_single_optimization(config_path, load_mass, optimizer_type, initial_solution=None):
    config = load_config(config_path)
    updated_config = update_config_load_mass(config, load_mass)

    simulation = CraneSimulation(updated_config, render=False)

    # Choose optimizer based on user selection
    if optimizer_type.lower() == 'genetic' or optimizer_type.lower() == 'ga':
        optimizer = GeneticOptimizer(simulation, updated_config, initial_solution=initial_solution)
    elif optimizer_type.lower() == 'pso':
        optimizer = PSOOptimizer(simulation, updated_config, initial_solution=initial_solution)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose 'genetic' or 'pso'.")

    optimizer.run()

    # Fix for unpacking issue
    result = optimizer.get_best_solution()

    # Determine the format of the result and extract values correctly
    if isinstance(result, tuple):
        if len(result) == 2:
            best_solution, best_fitness = result
        else:
            # If more than two values are returned, take the first two
            best_solution, best_fitness = result[0], result[1]
    elif isinstance(result, dict):
        best_solution = result.get('solution')
        best_fitness = result.get('fitness')
    elif hasattr(result, 'solution') and hasattr(result, 'fitness'):
        # Handle object with attributes
        best_solution = result.solution
        best_fitness = result.fitness
    else:
        # Default fallback
        best_solution = result
        best_fitness = None

    return load_mass, best_solution, best_fitness


def run_parallel_optimizations(config_path, load_masses, optimizer_type, initial_solution):
    with Pool() as pool:
        results = pool.starmap(run_single_optimization,
                               [(config_path, mass, optimizer_type, initial_solution) for mass in load_masses])
    return results


def run_sequential_optimizations(config_path, load_masses, optimizer_type, initial_solution, results_file):
    results = []
    best_solution = initial_solution

    # Load existing results if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip the header row
            results = [{'load_mass': float(row[0]),
                        'best_solution': [float(x) for x in row[1:-1]],
                        'best_fitness': float(row[-1])}
                       for row in reader]
        if results:
            best_solution = np.array(results[-1]['best_solution'])

    for i, mass in enumerate(load_masses):
        print(f"\n--- Starting optimization {i + 1}/{len(load_masses)} for load mass: {mass} ---")

        # Fixed function call
        load_mass, best_solution, best_fitness = run_single_optimization(
            config_path, mass, optimizer_type, best_solution)

        # Ensure best_solution is properly handled
        if best_solution is None:
            print(f"Warning: No valid solution found for mass {mass}")
            if i > 0:  # Use previous solution if available
                best_solution = np.array(results[-1]['best_solution'])
            else:
                best_solution = initial_solution
            best_fitness = float('inf')  # Assign a poor fitness

        # Convert to list for storage if it's a numpy array
        if isinstance(best_solution, np.ndarray):
            best_solution_list = best_solution.tolist()
        else:
            best_solution_list = best_solution

        result = {
            'load_mass': load_mass,
            'best_solution': best_solution_list,
            'best_fitness': best_fitness
        }
        results.append(result)

        # Save results after each optimization
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['load_mass'] + [f'solution_{i}' for i in range(len(best_solution_list))] + ['fitness'])
            for res in results:
                writer.writerow([res['load_mass']] + res['best_solution'] + [res['best_fitness']])

        print(f"Completed optimization for mass: {mass}")
        print(f"Best fitness: {best_fitness}")
        print(f"Best solution: {best_solution}")
        print("---")

    return results


def print_results(results):
    print("\n--- Results ---")
    for result in results:
        # Check if result is a tuple (from direct function return) or a dict (from loaded results)
        if isinstance(result, tuple):
            load_mass, best_solution, best_fitness = result
            print(f"Load Mass: {load_mass}")
            print(f"Best Solution: {best_solution}")
            print(f"Best Fitness: {best_fitness}")
        else:
            # Handle dictionary format
            print(f"Load Mass: {result['load_mass']}")
            print(f"Best Solution: {result['best_solution']}")
            print(f"Best Fitness: {result['best_fitness']}")
        print("---")


def main():
    config_path = '../config/simulation_params.yaml'
    initial_Q = [800000000, 10000000, 1300, 1300000,
                 2000000, 100000, 32400, 32400]
    initial_R = [1.0, 1.0]

    initial_solution = np.hstack((initial_Q, initial_R))

    while True:
        choice = input("Enter '1' for single optimization, '2' for parallel optimizations, '3' for sequential optimizations, or 'q' to quit: ")

        if choice == 'q':
            break
        elif choice in ['1', '2', '3']:
            # Ask user to select optimization algorithm
            optimizer_type = input("Select optimizer (genetic/pso): ").strip().lower()
            if optimizer_type not in ['genetic', 'ga', 'pso']:
                print("Invalid optimizer selection. Please choose 'genetic' or 'pso'.")
                continue

            # Create results filename based on optimizer type
            results_file = f'optimization_results_{optimizer_type}.csv'

            if choice == '1':
                load_mass = float(input("Enter the load mass for optimization: "))
                results = [run_single_optimization(config_path, load_mass, optimizer_type, initial_solution)]
                print_results(results)
            elif choice in ['2', '3']:
                min_mass = float(input("Enter the minimum load mass: "))
                max_mass = float(input("Enter the maximum load mass: "))
                num_points = int(input("Enter the number of points to optimize: "))
                load_masses = np.linspace(min_mass, max_mass, num_points)
                print(f"Optimizing for masses: {load_masses}")

                if choice == '2':
                    results = run_parallel_optimizations(config_path, load_masses, optimizer_type, initial_solution)

                    # Save results for parallel optimization
                    with open(results_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        # Get the length of the solution from the first result
                        if results and isinstance(results[0], tuple) and len(results[0]) >= 2:
                            solution_length = len(results[0][1]) if hasattr(results[0][1], '__len__') else 1
                        else:
                            solution_length = len(initial_solution)

                        writer.writerow(['load_mass'] + [f'solution_{i}' for i in range(solution_length)] + ['fitness'])
                        for res in results:
                            if isinstance(res, tuple) and len(res) >= 3:
                                load_mass, best_solution, best_fitness = res
                                if hasattr(best_solution, 'tolist'):
                                    solution_list = best_solution.tolist()
                                elif hasattr(best_solution, '__iter__'):
                                    solution_list = list(best_solution)
                                else:
                                    solution_list = [best_solution]
                                writer.writerow([load_mass] + solution_list + [best_fitness])
                else:  # choice == '3'
                    results = run_sequential_optimizations(config_path, load_masses, optimizer_type,
                                                           initial_solution, results_file)

                print_results(results)
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()