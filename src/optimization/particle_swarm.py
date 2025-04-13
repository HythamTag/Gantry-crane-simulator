import csv
import os.path
import time
import numpy as np
import yaml
import datetime
from typing import Tuple, Dict, Any, List
import pyswarms as ps  # You'll need to install pyswarms


class PSOOptimizer:
    def __init__(self, simulation: Any, config: str, initial_solution: np.ndarray = None, parallel_processing=None):
        self.parallel_processing = parallel_processing
        self.simulation = simulation
        self.config = config
        self.initial_solution = initial_solution
        self.filename_prefix = self._generate_filename_prefix()
        self.foldername = self._generate_filename_prefix()

        self.pso_instance = None
        self.pso_params = self._initialize_pso_params()

        self.best_fitnesses: List[float] = []
        self.best_Q_values: List[np.ndarray] = []
        self.best_R_values: List[np.ndarray] = []
        self.fitness_components: List[List[float]] = []
        self.max_thetas: List[List[float]] = []
        self.max_deviations: List[List[float]] = []
        self.times_to_target: List[float] = []
        self.max_accelerations: List[List[float]] = []
        self.max_jerks: List[List[float]] = []

        self.last_iter_time = time.time()
        self.iter_time = time.time()

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

    def _initialize_pso_params(self) -> Dict[str, Any]:
        pso_config = self.config.get('particle_swarm', {})
        solution_space = self.config['genetic_algorithm']['solution_space']  # Reuse the genetic algo ranges

        # Define bounds based on whether integral control is used
        if self.config['controller']['use_integral']:
            lb = ([solution_space['Q_range']['low']] * 8 +
                  [solution_space['I_range']['low']] * 2 +
                  [solution_space['R_range']['low']] * 2)
            ub = ([solution_space['Q_range']['high']] * 8 +
                  [solution_space['I_range']['high']] * 2 +
                  [solution_space['R_range']['high']] * 2)
        else:
            lb = ([solution_space['Q_range']['low']] * 8 +
                  [solution_space['R_range']['low']] * 2)
            ub = ([solution_space['Q_range']['high']] * 8 +
                  [solution_space['R_range']['high']] * 2)

        bounds = (lb, ub)

        # PSO parameters
        options = {
            'c1': pso_config.get('c1', 0.5),  # cognitive parameter
            'c2': pso_config.get('c2', 0.3),  # social parameter
            'w': pso_config.get('w', 0.9),  # inertia parameter
        }

        return {
            'n_particles': pso_config.get('n_particles', 30),
            'dimensions': len(lb),
            'options': options,
            'bounds': bounds,
            'iters': pso_config.get('iters', 100)
        }

    def _fitness_function(self, solutions, **kwargs):
        """
        Modified to accept additional keyword arguments that PySwarms might pass
        """
        n_particles = solutions.shape[0]
        fitnesses = np.zeros(n_particles)

        for i in range(n_particles):
            solution = solutions[i]
            Q_values, R_values = self._unpack_solution(solution)
            self._set_controller_matrices(Q_values, R_values)

            if self.simulation.simulate() is False:
                fitnesses[i] = float('inf')  # Maximize for PSO
                continue

            sim_data = self._get_simulation_data()
            fitness, components = self._calculate_fitness(sim_data)

            # PSO minimizes by default, so we need to negate our fitness
            fitnesses[i] = -fitness

        return fitnesses
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
        components['MSE_theta1'] = 0
        components['MSE_theta2'] = np.rad2deg(np.max(np.abs(q_values[3]))) ** 2 * weights['w4']
        components['q_ddot1_value'] = 0
        components['q_ddot3_value'] = 0
        # components['q_ddot1_value'] = np.max(np.abs(q_ddot_values[1])) ** 2 * weights['w5']
        # components['q_ddot3_value'] = np.max(np.abs(q_ddot_values[3])) ** 2 * weights['w6']

        # Time-to-target component
        components['MSE_time'] = 0

        # Apply time penalty for exceeding threshold
        if full_time_to_target > 7.6:
            components['MSE_time'] = (((full_time_to_target - 6.5789) * 10) ** 8) * weights['w7']

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
        headers = ['Iteration', 'Best_Fitness', 'Time_to_Target',
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

    def _on_iteration(self, iter_num, best_pos, best_cost):
        """Custom callback function to log progress and save data"""
        Q_values, R_values = self._unpack_solution(best_pos)

        start_time = time.time()
        self._set_controller_matrices(Q_values, R_values)
        res = self.simulation.simulate()
        end_time = time.time()
        trial_duration = end_time - start_time

        sim_data = self._get_simulation_data()
        # Negate back to original scale since PSO minimizes
        fitness, components = self._calculate_fitness(sim_data)

        self._update_optimization_data(fitness, Q_values, R_values, components)

        data = ([iter_num, fitness] +
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

        iter_duration = time.time() - self.iter_time
        self.iter_time = time.time()

        print(f"\n########################### Iteration {iter_num} #########################################")
        print(f"Simulation status: {res}")
        print(f'Iteration duration: {iter_duration}')
        print(f'Trial duration: {trial_duration}')
        print(f"time_to_target : {self.times_to_target[-1]}")
        print(f"max_theta1: {np.rad2deg(self.max_thetas[-1][0])}")
        print(f"max_theta2: {np.rad2deg(self.max_thetas[-1][1])}")
        print(f"max_x_acceleration: {self.max_accelerations[-1][0]}")
        print(f"max_l_acceleration: {self.max_accelerations[-1][1]}")
        print(f"max_x_jerk: {self.max_jerks[-1][0]}")
        print(f"max_l_jerk: {self.max_jerks[-1][1]}")
        print(f"Best Fitness: {fitness}")
        print(f"Best Q values: {Q_values.tolist()}")
        print(f"Best R values: {R_values.tolist()}")
        print("=" * 50)

        # Reset lists for the next iteration
        self.fitness_components = []
        self.max_thetas = []
        self.max_deviations = []
        self.times_to_target = []
        self.max_accelerations = []
        self.max_jerks = []

        return False  # Continue optimization

    def _update_optimization_data(self, best_fitness: float, Q_values: np.ndarray, R_values: np.ndarray, components: Dict[str, float]):
        self.best_fitnesses.append(best_fitness)
        self.best_Q_values.append(Q_values)
        self.best_R_values.append(R_values)
        self.fitness_components.append(list(components.values()))

    def run(self):
        print("Starting Particle Swarm Optimization")

        # Initialize the PSO optimizer with global best topology
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.pso_params['n_particles'],
            dimensions=self.pso_params['dimensions'],
            options=self.pso_params['options'],
            bounds=self.pso_params['bounds']
        )

        # If an initial solution was provided, use init_pos parameter when optimizing
        init_pos = None
        if self.initial_solution is not None:
            # Make sure the initial solution is within bounds
            lb, ub = self.pso_params['bounds']
            initial_solution_clipped = np.clip(self.initial_solution, lb, ub)

            # Create an initial position array where the first particle is our initial solution
            # and the rest are randomly initialized by PySwarms
            init_pos = np.random.uniform(
                low=lb, high=ub,
                size=(self.pso_params['n_particles'], self.pso_params['dimensions'])
            )
            init_pos[0] = initial_solution_clipped

            print(f"Using provided initial solution as first particle")

        # Run the optimization with optional init_pos parameter
        best_cost, best_pos = optimizer.optimize(
            self._fitness_function,
            iters=self.pso_params['iters'],
            verbose=True,
            init_pos=init_pos  # Pass the initial positions if we have them
        )

        # Handle the callback separately through iteration events if needed
        for i in range(self.pso_params['iters']):
            self._on_iteration(i, best_pos, best_cost)

        self.best_solution = best_pos
        self.best_cost = best_cost

        print("Particle Swarm Optimization Completed")
        print(f"Best cost (fitness): {-best_cost}")  # Negate to get back original fitness
        print(f"Best position (solution): {best_pos}")
    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        if not hasattr(self, 'best_solution'):
            raise RuntimeError("Optimization hasn't been run yet.")
        return self.best_solution, -self.best_cost  # Negate cost to get original fitness