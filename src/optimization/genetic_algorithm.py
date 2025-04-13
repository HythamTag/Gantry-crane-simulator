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

        # Initialize wider parameter ranges for high-mass systems
        self.is_high_mass = self._check_if_high_mass()
        self.exploration_factor = 10.0 if self.is_high_mass else 1.0

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

        # Track generations without improvement
        self.generations_without_improvement = 0
        self.best_fitness_so_far = float('-inf')
        self.original_mutation_rate = self.config['genetic_algorithm']['mutation']['probability']

        # Track best overall solution across multiple restarts
        self.global_best_solution = None
        self.global_best_fitness = float('-inf')

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

    def _check_if_high_mass(self) -> bool:
        """Determine if we're dealing with a high-mass system that needs more exploration"""
        load_mass = self.config['crane_system']['physical_params']['masses']['load']
        hook_mass = self.config['crane_system']['physical_params']['masses']['hook']
        return load_mass > hook_mass

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

        # For high mass systems, use more aggressive mutation settings
        mutation_percent = 50 if self.is_high_mass else 30
        mutation_type = "random"  # Changed from adaptive to random for more exploration

        # Initialize a list of mutation probabilities
        num_genes = ga_config['num_genes']
        base_mutation_prob = ga_config['mutation']['probability']
        mutation_prob = base_mutation_prob * 1.5 if self.is_high_mass else base_mutation_prob

        return {
            'num_generations': ga_config['num_generations'],
            'num_parents_mating': ga_config['num_parents_mating'],
            'fitness_func': self._fitness_wrapper,
            'sol_per_pop': ga_config['sol_per_pop'],
            'num_genes': num_genes,
            'gene_type': float,
            'initial_population': self._generate_initial_population(),
            'gene_space': gene_space,

            # Updated mutation parameters for better exploration
            'mutation_type': mutation_type,
            'mutation_probability': mutation_prob,
            'mutation_percent_genes': mutation_percent,
            'mutation_by_replacement': True,

            # Use two-point crossover instead of default
            'crossover_type': "two_points",

            # Elite selection (keep parents)
            'keep_parents': min(ga_config['mutation']['keep_parents'], ga_config['sol_per_pop'] // 4),
            'parent_selection_type': "tournament",
            'K_tournament': 3,  # Tournament size

            # Callbacks
            'on_generation': self._on_generation,

            # Parallel processing
            'parallel_processing': self.parallel_processing,
        }

    def _fitness_wrapper(self, ga_instance: Any, solution: np.ndarray, solution_idx: int) -> float:
        # Apply scaling for high-mass systems
        if self.is_high_mass:
            # Scale solution parameters to explore a wider range for high mass
            scaled_solution = np.copy(solution)
            if self.config['controller']['use_integral']:
                # Scale Q values (state weights)
                scaled_solution[:8] = solution[:8] * self.exploration_factor
                # Scale I values (integral weights)
                scaled_solution[8:10] = solution[8:10] * self.exploration_factor
                # Scale R values (control weights) - inversely to encourage more aggressive control
                scaled_solution[10:] = solution[10:] / self.exploration_factor
            else:
                # Scale Q values (state weights)
                scaled_solution[:8] = solution[:8] * self.exploration_factor
                # Scale R values (control weights) - inversely to encourage more aggressive control
                scaled_solution[8:] = solution[8:] / self.exploration_factor

            Q_values, R_values = self._unpack_solution(scaled_solution)
        else:
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
        components['ISE_theta1'] = 0
        components['ISE_theta2'] = 0
        components['MSE_theta1'] = 0
        components['MSE_theta2'] = (np.rad2deg(np.max(np.abs(q_values[3]))) ** 2) * weights['w4']
        components['q_ddot1_value'] = 0
        components['q_ddot3_value'] = 0

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
                   'Max_X_Jerk', 'Max_L_Jerk',
                   'Current_Mutation_Rate', 'Current_Exploration_Factor']

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

    def _adapt_mutation_rate(self, ga_instance):
        """Adapt mutation rate based on improvement history and current stage"""
        current_best_fitness = ga_instance.best_solution()[1]
        gen = ga_instance.generations_completed
        max_gen = self.config['genetic_algorithm']['num_generations']

        # Calculate current stage of the optimization (early, middle, late)
        progress = gen / max_gen

        # Check if we're improving
        if current_best_fitness > self.best_fitness_so_far:
            self.best_fitness_so_far = current_best_fitness
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        # Base mutation strategies on progress and improvement
        if self.generations_without_improvement >= 5:
            # Significant stagnation - increase mutation dramatically
            if self.is_high_mass:
                new_rate = min(0.9, ga_instance.mutation_probability * 1.5)
            else:
                new_rate = min(0.8, ga_instance.mutation_probability * 1.3)
            ga_instance.mutation_probability = new_rate

            # Also increase exploration factor if high mass
            if self.is_high_mass:
                self.exploration_factor *= 1.2
                self.exploration_factor = min(self.exploration_factor, 50.0)

        elif self.generations_without_improvement == 0 and progress > 0.5:
            # Recently improved and in later half - gradually refine
            new_rate = max(self.original_mutation_rate, ga_instance.mutation_probability * 0.9)
            ga_instance.mutation_probability = new_rate

            # Decrease exploration factor slightly to refine
            if self.is_high_mass and progress > 0.7:
                self.exploration_factor *= 0.95
                self.exploration_factor = max(self.exploration_factor, 1.0)

        # Early stages - keep high mutation
        elif progress < 0.3:
            if self.is_high_mass:
                ga_instance.mutation_probability = 0.7
            else:
                ga_instance.mutation_probability = 0.5

        return ga_instance.mutation_probability

    def _inject_diversity(self, ga_instance):
        """Enhanced diversity injection mechanism"""
        # Inject diversity more frequently for high-mass systems
        gen = ga_instance.generations_completed
        injection_frequency = 3 if self.is_high_mass else 7
        extreme_injection_frequency = 9 if self.is_high_mass else 21

        should_inject = gen % injection_frequency == 0 and gen > 0
        should_inject_extreme = gen % extreme_injection_frequency == 0 and gen > 0

        if not (should_inject or should_inject_extreme):
            return

        population = ga_instance.population
        fitness = ga_instance.last_generation_fitness

        if fitness is None or len(fitness) == 0:
            return

        # Sort indices by fitness (ascending order since we're maximizing)
        sorted_indices = np.argsort(fitness)

        if should_inject:
            # Replace more solutions for high-mass systems
            replace_percent = 0.3 if self.is_high_mass else 0.15
            num_to_replace = max(3, int(replace_percent * len(population)))
            worst_indices = sorted_indices[:num_to_replace]

            print(f"Injecting diversity - replacing {num_to_replace} solutions")

            # Calculate exploration range factors based on system
            Q_factor = 50.0 if self.is_high_mass else 10.0
            R_factor = 0.1 if self.is_high_mass else 0.5  # Lower R values for more aggressive control

            for idx in worst_indices:
                if self.config['controller']['use_integral']:
                    # Generate wide-ranging values for Q (state weights)
                    population[idx, :8] = np.random.exponential(
                        scale=self.config['genetic_algorithm']['solution_space']['Q_range']['high'] * Q_factor,
                        size=8
                    )
                    # Keep some smaller to allow exploration of different regions
                    if np.random.random() < 0.5:
                        random_indices = np.random.choice(8, size=np.random.randint(1, 5), replace=False)
                        population[idx, random_indices] = population[idx, random_indices] * 0.01

                    # Generate I values (integral weights)
                    population[idx, 8:10] = np.random.exponential(
                        scale=self.config['genetic_algorithm']['solution_space']['I_range']['high'] * Q_factor,
                        size=2
                    )

                    # Generate R values (control weights) - smaller for more aggressive control
                    population[idx, 10:] = np.random.exponential(
                        scale=self.config['genetic_algorithm']['solution_space']['R_range']['high'] * R_factor,
                        size=2
                    )
                else:
                    # Similar logic for non-integral controllers
                    population[idx, :8] = np.random.exponential(
                        scale=self.config['genetic_algorithm']['solution_space']['Q_range']['high'] * Q_factor,
                        size=8
                    )
                    if np.random.random() < 0.5:
                        random_indices = np.random.choice(8, size=np.random.randint(1, 5), replace=False)
                        population[idx, random_indices] = population[idx, random_indices] * 0.01

                    population[idx, 8:] = np.random.exponential(
                        scale=self.config['genetic_algorithm']['solution_space']['R_range']['high'] * R_factor,
                        size=2
                    )

        # More extreme diversity injection
        if should_inject_extreme:
            # Create several extremely diverse solutions
            num_extreme = max(2, int(0.1 * len(population)))
            extreme_indices = sorted_indices[:num_extreme]

            print(f"Injecting {num_extreme} EXTREME solutions for exploration")

            # Much more extreme exploration factors
            extreme_Q_factor = 200.0 if self.is_high_mass else 50.0
            extreme_R_factor = 0.01 if self.is_high_mass else 0.1

            for idx in extreme_indices:
                if self.config['controller']['use_integral']:
                    # Create completely different solution profiles
                    if np.random.random() < 0.33:
                        # High Q, low R - aggressive control
                        population[idx, :8] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['Q_range']['high'] * extreme_Q_factor,
                            size=8
                        )
                        population[idx, 8:10] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['I_range']['high'] * extreme_Q_factor,
                            size=2
                        )
                        population[idx, 10:] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['R_range']['low'] * extreme_R_factor,
                            size=2
                        )
                    elif np.random.random() < 0.5:
                        # Low Q, high R - conservative control
                        population[idx, :8] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['Q_range']['low'] * 10,
                            size=8
                        )
                        population[idx, 8:10] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['I_range']['low'] * 10,
                            size=2
                        )
                        population[idx, 10:] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['R_range']['high'] * 10,
                            size=2
                        )
                    else:
                        # Completely random profile
                        Q_range = self.config['genetic_algorithm']['solution_space']['Q_range']
                        I_range = self.config['genetic_algorithm']['solution_space']['I_range']
                        R_range = self.config['genetic_algorithm']['solution_space']['R_range']

                        population[idx, :8] = np.random.uniform(
                            Q_range['low'] * 0.1, Q_range['high'] * extreme_Q_factor, size=8
                        )
                        population[idx, 8:10] = np.random.uniform(
                            I_range['low'] * 0.1, I_range['high'] * extreme_Q_factor, size=2
                        )
                        population[idx, 10:] = np.random.uniform(
                            R_range['low'] * 0.01, R_range['high'] * 10, size=2
                        )

                else:
                    # Similar logic for non-integral controllers
                    if np.random.random() < 0.33:
                        # High Q, low R
                        population[idx, :8] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['Q_range']['high'] * extreme_Q_factor,
                            size=8
                        )
                        population[idx, 8:] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['R_range']['low'] * extreme_R_factor,
                            size=2
                        )
                    elif np.random.random() < 0.5:
                        # Low Q, high R
                        population[idx, :8] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['Q_range']['low'] * 10,
                            size=8
                        )
                        population[idx, 8:] = np.random.exponential(
                            scale=self.config['genetic_algorithm']['solution_space']['R_range']['high'] * 10,
                            size=2
                        )
                    else:
                        # Completely random profile
                        Q_range = self.config['genetic_algorithm']['solution_space']['Q_range']
                        R_range = self.config['genetic_algorithm']['solution_space']['R_range']

                        population[idx, :8] = np.random.uniform(
                            Q_range['low'] * 0.1, Q_range['high'] * extreme_Q_factor, size=8
                        )
                        population[idx, 8:] = np.random.uniform(
                            R_range['low'] * 0.01, R_range['high'] * 10, size=2
                        )

            # Ensure no NaN or inf values in the population
            bad_values = ~np.isfinite(population)
            if np.any(bad_values):
                print("Warning: NaN or inf values detected in population - fixing")
                # Replace with random valid values
                valid_pop = self._generate_initial_population()
                for i, row in enumerate(population):
                    bad_indices = np.where(~np.isfinite(row))[0]
                    if len(bad_indices) > 0:
                        population[i, bad_indices] = valid_pop[0, bad_indices]

    def _on_generation(self, ga_instance: Any):
        gen = ga_instance.generations_completed

        # Apply adaptive mutation rate strategy
        current_mutation_rate = self._adapt_mutation_rate(ga_instance)

        # Inject diversity to avoid premature convergence
        self._inject_diversity(ga_instance)

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

        # Update global best solution if this is better
        if best_fitness > self.global_best_fitness:
            self.global_best_fitness = best_fitness
            self.global_best_solution = best_solution.copy()

        data = ([gen, best_fitness] +
                [self.times_to_target[-1]] +
                list(map(np.rad2deg, self.max_thetas[-1])) +
                self.max_deviations[-1] +
                self.max_accelerations[-1] +
                self.max_jerks[-1] +
                [current_mutation_rate, self.exploration_factor] +  # Add exploration factor tracking
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
        print(f"Current mutation rate: {current_mutation_rate}")
        print(f"Current exploration factor: {self.exploration_factor}")
        print(f"Generations without improvement: {self.generations_without_improvement}")
        print(f"time_to_target: {self.times_to_target[-1]}")
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

        # Create initial population with logarithmic distribution to explore more ranges
        if num_genes == 12:
            # Log-uniformly distributed values for better exploration of parameter space
            Q_low = np.log10(solution_space['Q_range']['low'])
            Q_high = np.log10(solution_space['Q_range']['high'])
            I_low = np.log10(solution_space['I_range']['low'])
            I_high = np.log10(solution_space['I_range']['high'])
            R_low = np.log10(solution_space['R_range']['low'])
            R_high = np.log10(solution_space['R_range']['high'])

            # Generate log-uniform values and convert back to linear space
            population[:, :8] = 10 ** np.random.uniform(Q_low, Q_high, (pop_size, 8))
            population[:, 8:10] = 10 ** np.random.uniform(I_low, I_high, (pop_size, 2))
            population[:, 10:] = 10 ** np.random.uniform(R_low, R_high, (pop_size, 2))
        elif num_genes == 10:
            # Log-uniformly distributed values for better exploration of parameter space
            Q_low = np.log10(solution_space['Q_range']['low'])
            Q_high = np.log10(solution_space['Q_range']['high'])
            R_low = np.log10(solution_space['R_range']['low'])
            R_high = np.log10(solution_space['R_range']['high'])

            # Generate log-uniform values and convert back to linear space
            population[:, :8] = 10 ** np.random.uniform(Q_low, Q_high, (pop_size, 8))
            population[:, 8:] = 10 ** np.random.uniform(R_low, R_high, (pop_size, 2))

        # Ensure wide coverage by creating some solutions at extremes
        if pop_size >= 10:
            # Set 10% of initial population with extreme values for better exploration
            extreme_count = max(1, pop_size // 10)

            # High Q, low R solutions
            for i in range(extreme_count):
                if num_genes == 12:
                    population[i, :8] = 10 ** np.random.uniform(Q_high - 1, Q_high, 8)
                    population[i, 8:10] = 10 ** np.random.uniform(I_high - 1, I_high, 2)
                    population[i, 10:] = 10 ** np.random.uniform(R_low, R_low + 1, 2)
                else:
                    population[i, :8] = 10 ** np.random.uniform(Q_high - 1, Q_high, 8)
                    population[i, 8:] = 10 ** np.random.uniform(R_low, R_low + 1, 2)

            # Low Q, high R solutions
            for i in range(extreme_count, 2 * extreme_count):
                if num_genes == 12:
                    population[i, :8] = 10 ** np.random.uniform(Q_low, Q_low + 1, 8)
                    population[i, 8:10] = 10 ** np.random.uniform(I_low, I_low + 1, 2)
                    population[i, 10:] = 10 ** np.random.uniform(R_high - 1, R_high, 2)
                else:
                    population[i, :8] = 10 ** np.random.uniform(Q_low, Q_low + 1, 8)
                    population[i, 8:] = 10 ** np.random.uniform(R_high - 1, R_high, 2)

        # If we have an initial solution, include it
        if self.initial_solution is not None:
            population[0] = self.initial_solution

        # Check for NaN or inf values and replace them
        bad_values = ~np.isfinite(population)
        if np.any(bad_values):
            # Replace any bad values with reasonable values from the middle of the range
            if num_genes == 12:
                mid_Q = 10 ** ((Q_high + Q_low) / 2)
                mid_I = 10 ** ((I_high + I_low) / 2)
                mid_R = 10 ** ((R_high + R_low) / 2)

                population[bad_values[:, :8]] = mid_Q
                population[bad_values[:, 8:10]] = mid_I
                population[bad_values[:, 10:]] = mid_R
            else:
                mid_Q = 10 ** ((Q_high + Q_low) / 2)
                mid_R = 10 ** ((R_high + R_low) / 2)

                population[bad_values[:, :8]] = mid_Q
                population[bad_values[:, 8:]] = mid_R

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

        # Save the best solution to a separate file
        best_solution, best_fitness, _ = self.ga_instance.best_solution()
        best_Q_values, best_R_values = self._unpack_solution(best_solution)

        best_solution_path = os.path.join(self.foldername, "best_solution.yaml")
        best_solution_data = {
            "fitness": float(best_fitness),
            "Q_values": best_Q_values.tolist(),
            "R_values": best_R_values.tolist(),
        }

        with open(best_solution_path, 'w') as f:
            yaml.dump(best_solution_data, f)

        print(f"Best solution saved to {best_solution_path}")
        print(f"Final Best Fitness: {best_fitness}")
        print(f"Final Best Q values: {best_Q_values.tolist()}")
        print(f"Final Best R values: {best_R_values.tolist()}")

    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        if self.ga_instance is None:
            raise RuntimeError("Optimization hasn't been run yet.")
        return self.ga_instance.best_solution()
