import csv
import os.path
import time
import pygad
import numpy as np
import yaml
import datetime
from typing import Tuple, Dict, Any, List


class GeneticOptimizer:
    def __init__(self, simulation: Any, config: str, initial_solution: np.ndarray = None, parallel_processing=None):
        self.parallel_processing = parallel_processing
        self.simulation = simulation
        self.config = config
        self.initial_solution = initial_solution
        self.filename_prefix = self._generate_filename_prefix()
        self.foldername = self._generate_filename_prefix()

        self.ga_instance = None
        self.ga_params = self._initialize_ga_params()

        self.best_fitnesses: List[float] = []
        self.best_Q_values: List[np.ndarray] = []
        self.best_R_values: List[np.ndarray] = []
        self.fitness_components: List[List[float]] = []
        self.max_thetas: List[List[float]] = []
        self.max_deviations: List[List[float]] = []
        self.times_to_target: List[float] = []
        self.max_accelerations: List[List[float]] = []
        self.max_jerks: List[List[float]] = []

        self.last_gen_time = time.time()
        self.gen_time = time.time()

        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)

        self.csv_filepath = os.path.join(self.foldername,
                                         f"optimization_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        # Global fitness components
        self.fitness_component_names = ['ISE_x', 'ISE_l', 'ISE_theta1', 'ISE_theta2',
                                        'MSE_theta1', 'MSE_theta2', 'q_ddot1_value',
                                        'q_ddot3_value', 'MSE_time']

        self._initialize_csv()

        # Save the config file
        self._save_config()

    def _save_config(self):
        config_save_path = os.path.join(self.foldername, f"config.yaml")
        with open(config_save_path, 'w') as file:
            yaml.dump(self.config, file)

    def _generate_filename_prefix(self):
        physical_params = self.config['crane_system']['physical_params']
        initial_conditions = self.config['crane_system']['initial_conditions']
        targets = self.config['crane_system']['target_positions']
        enable_trajectory_velocity_tracking = self.config['controller']['enable_trajectory_velocity_tracking']

        prefix = (
            f"[m_trolley{physical_params['masses']['trolley']}]_"
            f"[m_hook{physical_params['masses']['hook']}]_"
            f"[m_load{physical_params['masses']['load']}]_"
            f"[init_pos{initial_conditions['trolley_position']:.2f}]_"
            f"[init_len{initial_conditions['rope_length']:.2f}]_"
            f"[target_pos{targets['trolley']:.2f}]_"
            f"[target_len{targets['rope']:.2f}]"
            f"[traj_vel{enable_trajectory_velocity_tracking}]"
        )

        return prefix

    def _initialize_ga_params(self) -> Dict[str, Any]:
        ga_config = self.config['genetic_algorithm']
        solution_space = ga_config['solution_space']

        if self.config['controller']['use_integral']:
            gene_space = ([solution_space['Q_range']] * 8 +
                          [solution_space['I_range']] * 2 +
                          [solution_space['R_range']] * 2)
        else:
            gene_space = ([solution_space['Q_range']] * 8 +
                          [solution_space['R_range']] * 2)
        return {
            'num_generations': ga_config['num_generations'],
            'num_parents_mating': ga_config['num_parents_mating'],
            'fitness_func': self._fitness_wrapper,
            'sol_per_pop': ga_config['sol_per_pop'],
            'num_genes': ga_config['num_genes'],
            'gene_type': float,
            'initial_population': self._generate_initial_population(),
            'gene_space': gene_space,

            # 'mutation_type': "random",
            'mutation_probability': ga_config['mutation']['probability'],

            'mutation_type': "random",
            # 'mutation_probability': [ga_config['mutation']['probability'] * 3, ga_config['mutation']['probability']],

            'keep_parents': ga_config['mutation']['keep_parents'],
            'on_generation': self._on_generation,
            'parallel_processing': self.parallel_processing,
        }

    def _fitness_wrapper(self, ga_instance: Any, solution: np.ndarray, solution_idx: int) -> float:
        Q_values, R_values = self._unpack_solution(solution)
        self._set_controller_matrices(Q_values, R_values)

        if self.simulation.simulate() is False:
            return -np.inf

        sim_data = self._get_simulation_data()
        fitness, components = self._calculate_fitness(sim_data)

        return fitness

    def _unpack_solution(self, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.config['controller']['use_integral']:
            return solution[:10], solution[10:]
        else:
            return solution[:8], solution[8:]

    def _set_controller_matrices(self, Q_values: np.ndarray, R_values: np.ndarray):
        self.simulation.controller.Q_state = np.diag(Q_values[:8])
        if self.config['controller']['use_integral']:
            self.simulation.controller.Q_int = np.diag(Q_values[8:10])
        self.simulation.controller.R = np.diag(R_values)

    def _get_simulation_data(self) -> dict:
        return {
            'q_values': np.array(self.simulation.q_values).T,
            'q_dot_values': np.array(self.simulation.q_dot_values).T,
            'q_ddot_values': np.array(self.simulation.q_ddot_values).T,
            'q_dddot_values': np.array(self.simulation.q_dddot_values).T,
            'integral_error_values': self.simulation.integral_error_values.T,
            'target_x': self.simulation.target[0],
            'target_l': self.simulation.target[1],
            'trajectory_pos_vector': self.simulation.trajectory_pos_vector.T,
            'initial_state': self.simulation.initial_state
        }

    def _calculate_fitness(self, data: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        q_values = data['q_values']
        q_ddot_values = data['q_ddot_values']
        q_dddot_values = data['q_dddot_values']
        trajectory_pos_vector = data['trajectory_pos_vector']
        target_x, target_l = data['target_x'], data['target_l']

        tolerance = self.config['crane_system']['target_positions']['target_tolerance']
        tolerance_reached = lambda q, target: np.abs(q - target) <= tolerance

        trolley_reached = tolerance_reached(q_values[0], target_x)
        rope_reached = tolerance_reached(q_values[1], target_l)

        full_target_reached = trolley_reached & rope_reached

        # Calculate time to target - if target is never reached, use maximum simulation time
        if full_target_reached.any():
            full_target_reached_index = np.where(full_target_reached)[0][0]
            full_time_to_target = full_target_reached_index * self.config['simulation']['time_step']
        else:
            full_time_to_target = len(q_values[0]) * self.config['simulation']['time_step']
            # Apply a severe penalty for not reaching target at all
            target_failure_penalty = 1e6  # Very large penalty

        components = {}
        weights = self.config['genetic_algorithm']['fitness_function']['weights']

        components['ISE_x'] = np.sum((trajectory_pos_vector - q_values[0]) ** 2) * weights['w1']
        components['ISE_l'] = np.sum((target_l - q_values[1]) ** 2) * weights['w2']
        # components['ISE_theta1'] = np.sum(q_values[2] ** 2) * weights['w3']
        # components['ISE_theta2'] = np.sum(q_values[3] ** 2) * weights['w4']
        components['ISE_theta1'] = 0
        components['ISE_theta2'] = 0
        components['MSE_theta1'] = np.rad2deg(np.max(np.abs(q_values[2]))) ** 2 * weights['w3']
        components['MSE_theta2'] = np.rad2deg(np.max(np.abs(q_values[3]))) ** 2 * weights['w4']
        components['q_ddot1_value'] = np.max(np.abs(q_ddot_values[1])) ** 2 * weights['w5']
        components['q_ddot3_value'] = np.max(np.abs(q_ddot_values[3])) ** 2 * weights['w6']

        # Time-to-target component
        components['MSE_time'] = 0

        # Apply time penalty for exceeding threshold
        if full_time_to_target > 7.6:
            components['MSE_time'] = (((full_time_to_target - 6.5789) * 10) ** 5) * weights['w7']

        # Add severe penalty if target was never reached
        if not full_target_reached.any():
            # Calculate final distance from target to use as part of the penalty
            final_x_error = np.abs(q_values[0][-1] - target_x)
            final_l_error = np.abs(q_values[1][-1] - target_l)
            error_factor = (final_x_error + final_l_error) * 10  # Scale error for penalty

            # Apply target failure penalty proportional to how far we ended from target
            components['target_failure'] = target_failure_penalty * error_factor * weights['w7']

        # Calculate the fitness (negative because we're minimizing)
        base_fitness = -sum(components.values())

        # Record additional metrics
        max_theta1 = np.max(np.abs(q_values[2]))
        max_theta2 = np.max(np.abs(q_values[3]))
        max_x_deviation = np.abs(target_x - q_values[0][-1])
        max_l_deviation = np.abs(target_l - q_values[1][-1])
        max_trolley_acc = np.max(np.abs(q_ddot_values[0]))
        max_rope_acc = np.max(np.abs(q_ddot_values[1]))
        max_trolley_jerk = np.max(np.abs(q_dddot_values[0]))
        max_rope_jerk = np.max(np.abs(q_dddot_values[1]))

        self.max_thetas.append([max_theta1, max_theta2])
        self.max_deviations.append([max_x_deviation, max_l_deviation])
        self.times_to_target.append(full_time_to_target)
        self.max_accelerations.append([max_trolley_acc, max_rope_acc])
        self.max_jerks.append([max_trolley_jerk, max_rope_jerk])

        return base_fitness, components

    def _initialize_csv(self):
        headers = ['Generation', 'Best_Fitness', 'Time_to_Target',
                   'Max_Theta1', 'Max_Theta2', 'Max_X_Deviation', 'Max_L_Deviation',
                   'Max_X_Acceleration', 'Max_L_Acceleration',
                   'Max_X_Jerk', 'Max_L_Jerk']

        # Add Q and R headers dynamically
        if self.config['controller']['use_integral']:
            headers.extend([f'Q{i + 1}' for i in range(10)])
        else:
            headers.extend([f'Q{i + 1}' for i in range(8)])
        headers.extend(['R1', 'R2'])

        # Add Fitness component headers dynamically
        headers.extend(['Fitness_' + c for c in self.fitness_component_names])

        with open(self.csv_filepath, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

    def _on_generation(self, ga_instance: Any):
        gen = ga_instance.generations_completed

        best_solution, best_fitness, _ = ga_instance.best_solution()
        Q_values, R_values = self._unpack_solution(best_solution)

        start_time = time.time()

        self._set_controller_matrices(Q_values, R_values)
        res = self.simulation.simulate()

        end_time = time.time()
        trial_duration = end_time - start_time

        sim_data = self._get_simulation_data()
        fitness, components = self._calculate_fitness(sim_data)

        self._update_optimization_data(best_fitness, Q_values, R_values, components)

        data = ([gen, best_fitness] +
                [self.times_to_target[-1]] +
                list(map(np.rad2deg, self.max_thetas[-1])) +
                self.max_deviations[-1] +
                self.max_accelerations[-1] +
                self.max_jerks[-1] +
                list(Q_values) +
                list(R_values) +
                [components[name] for name in self.fitness_component_names])

        with open(self.csv_filepath, 'a', newline='') as f:
            csv.writer(f).writerow(data)

        gen_duration = time.time() - self.gen_time
        self.gen_time = time.time()

        print(f"\n########################### Generation {gen} #########################################")
        print(f"Simulation status: {res}")
        print(f'Generation duration: {gen_duration}')
        print(f'Trial duration: {trial_duration}')
        print(f"time_to_target : {self.times_to_target[-1]}")
        print(f"max_theta1: {np.rad2deg(self.max_thetas[-1][0])}")
        print(f"max_theta2: {np.rad2deg(self.max_thetas[-1][1])}")
        print(f"max_x_acceleration: {self.max_accelerations[-1][0]}")
        print(f"max_l_acceleration: {self.max_accelerations[-1][1]}")
        print(f"max_x_jerk: {self.max_jerks[-1][0]}")
        print(f"max_l_jerk: {self.max_jerks[-1][1]}")
        print(f"Best Fitness: {best_fitness}")
        print(f"Best Q values: {Q_values.tolist()}")
        print(f"Best R values: {R_values.tolist()}")
        print("=" * 50)

        # Reset lists for the next generation
        self.fitness_components = []
        self.max_thetas = []
        self.max_deviations = []
        self.times_to_target = []
        self.max_accelerations = []
        self.max_jerks = []

    def _generate_initial_population(self) -> np.ndarray:
        pop_size = self.config['genetic_algorithm']['sol_per_pop']
        num_genes = self.config['genetic_algorithm']['num_genes']
        solution_space = self.config['genetic_algorithm']['solution_space']
        population = np.zeros((pop_size, num_genes))

        if num_genes == 12:
            population[:, :8] = np.random.uniform(solution_space['Q_range']['low'], solution_space['Q_range']['high'], (pop_size, 8))
            population[:, 8:10] = np.random.uniform(solution_space['I_range']['low'], solution_space['I_range']['high'], (pop_size, 2))
            population[:, 10:] = np.random.uniform(solution_space['R_range']['low'], solution_space['R_range']['high'], (pop_size, 2))
        elif num_genes == 10:
            population[:, :8] = np.random.uniform(solution_space['Q_range']['low'], solution_space['Q_range']['high'], (pop_size, 8))
            population[:, 8:] = np.random.uniform(solution_space['R_range']['low'], solution_space['R_range']['high'], (pop_size, 2))

        if self.initial_solution is not None:
            population[0] = self.initial_solution

        return population

    def _update_optimization_data(self, best_fitness: float, Q_values: np.ndarray, R_values: np.ndarray, components: Dict[str, float]):
        self.best_fitnesses.append(best_fitness)
        self.best_Q_values.append(Q_values)
        self.best_R_values.append(R_values)
        self.fitness_components.append(list(components.values()))

    def run(self):
        print("Starting Genetic Algorithm Optimization")
        self.ga_instance = pygad.GA(**self.ga_params)
        self.ga_instance.run()
        print("Genetic Algorithm Optimization Completed")

    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        if self.ga_instance is None:
            raise RuntimeError("Optimization hasn't been run yet.")
        return self.ga_instance.best_solution()
