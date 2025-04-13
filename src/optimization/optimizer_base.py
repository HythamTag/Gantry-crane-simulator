# optimizer_base.py
import os
import datetime
import time
import csv
import numpy as np
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Callable, Optional, Type


class OptimizerBase(ABC):
    """Base class for all optimization algorithms."""

    def __init__(self,
                 simulation: Any,
                 controller_class: Type,
                 config: Dict[str, Any],
                 fitness_function: Callable = None,
                 output_dir: str = "optimization_results"):
        """
        Initialize the optimizer base class.

        Args:
            simulation: The simulation object
            controller_class: The controller class to optimize
            config: Configuration dictionary
            fitness_function: Custom fitness function (optional)
            output_dir: Directory for saving results
        """
        self.simulation = simulation
        self.controller_class = controller_class
        self.config = config
        self.custom_fitness_function = fitness_function

        # Setup output directories and files
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.controller_name = controller_class.__name__
        self.foldername = os.path.join(output_dir, f"{self.controller_name}_{timestamp}")
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername, exist_ok=True)

        # Initialize data collection
        self.metrics = {
            'best_fitness': [],
            'parameters': [],
            'generation': [],
            'time_elapsed': [],
            'component_metrics': []
        }

        # Initialize CSV and config files
        self.csv_filepath = os.path.join(self.foldername, f"optimization_data_{timestamp}.csv")
        self._save_config()

    def _save_config(self):
        """Save the configuration to a YAML file."""
        config_save_path = os.path.join(self.foldername, "config.yaml")
        with open(config_save_path, 'w') as file:
            yaml.dump(self.config, file)

    @abstractmethod
    def optimize(self):
        """Run the optimization algorithm."""
        pass

    @abstractmethod
    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Get the best solution found by the optimizer."""
        pass

    def evaluate_solution(self, parameters: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a solution by running a simulation with the given parameters.

        Args:
            parameters: The parameters to evaluate

        Returns:
            Tuple of (fitness_value, component_metrics)
        """
        # Apply parameters to controller
        self._apply_parameters_to_controller(parameters)

        # Run simulation
        if self.simulation.simulate() is False:
            return -np.inf, {}

        # Get simulation data and calculate fitness
        sim_data = self._get_simulation_data()
        if self.custom_fitness_function:
            fitness, component_metrics = self.custom_fitness_function(sim_data, self.config)
        else:
            fitness, component_metrics = self._calculate_fitness(sim_data)

        return fitness, component_metrics

    @abstractmethod
    def _apply_parameters_to_controller(self, parameters: np.ndarray):
        """
        Apply parameters to the controller.
        This method should be implemented by each optimizer subclass.
        """
        pass

    def _get_simulation_data(self) -> Dict[str, np.ndarray]:
        """Get the simulation data for fitness calculation."""
        return {
            'q_values': np.array(self.simulation.q_values).T,
            'q_dot_values': np.array(self.simulation.q_dot_values).T,
            'q_ddot_values': np.array(self.simulation.q_ddot_values).T,
            'q_dddot_values': np.array(self.simulation.q_dddot_values).T if hasattr(self.simulation, 'q_dddot_values') else None,
            'control_inputs': np.array(self.simulation.control_inputs).T if hasattr(self.simulation, 'control_inputs') else None,
            'target_x': self.simulation.target[0] if hasattr(self.simulation, 'target') else None,
            'target_l': self.simulation.target[1] if hasattr(self.simulation, 'target') else None,
            'trajectory_pos_vector': getattr(self.simulation, 'trajectory_pos_vector', np.array([])).T,
            'initial_state': self.simulation.initial_state if hasattr(self.simulation, 'initial_state') else None
        }

    def _calculate_fitness(self, data: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Default fitness calculation method. Override for custom fitness calculations.

        Args:
            data: Simulation data dictionary

        Returns:
            Tuple of (fitness_value, component_metrics)
        """
        q_values = data['q_values']
        q_ddot_values = data['q_ddot_values']
        q_dddot_values = data['q_dddot_values']
        trajectory_pos_vector = data.get('trajectory_pos_vector', np.zeros_like(q_values[0]))

        target_x = data.get('target_x', 0)
        target_l = data.get('target_l', 0)

        # Get fitness weights from config
        weights = self.config['optimization']['fitness_function']['weights']

        # Calculate fitness components
        components = {}

        # Position tracking error
        components['ISE_x'] = np.sum((trajectory_pos_vector - q_values[0]) ** 2) * weights.get('position', 1.0)
        components['ISE_l'] = np.sum((target_l - q_values[1]) ** 2) * weights.get('length', 1.0)

        # Swing reduction
        if q_values.shape[0] > 2:
            components['ISE_theta1'] = np.sum(q_values[2] ** 2) * weights.get('swing1', 1.0)
        if q_values.shape[0] > 3:
            components['ISE_theta2'] = np.sum(q_values[3] ** 2) * weights.get('swing2', 1.0)

        # Maximum swing angles
        if q_values.shape[0] > 2:
            components['MSE_theta1'] = np.rad2deg(np.max(np.abs(q_values[2]))) ** 2 * weights.get('max_swing1', 1.0)
        if q_values.shape[0] > 3:
            components['MSE_theta2'] = np.rad2deg(np.max(np.abs(q_values[3]))) ** 2 * weights.get('max_swing2', 1.0)

        # Acceleration and jerk penalties
        if q_ddot_values is not None and q_ddot_values.shape[0] > 1:
            components['q_ddot1_value'] = np.max(np.abs(q_ddot_values[1])) ** 2 * weights.get('acceleration', 1.0)
        if q_ddot_values is not None and q_ddot_values.shape[0] > 3:
            components['q_ddot3_value'] = np.max(np.abs(q_ddot_values[3])) ** 2 * weights.get('acceleration2', 1.0)

        # Calculate time to target
        tolerance = self.config['crane_system']['target_positions'].get('target_tolerance', 0.05)
        trolley_reached = np.abs(q_values[0] - target_x) <= tolerance
        rope_reached = np.abs(q_values[1] - target_l) <= tolerance

        full_target_reached = trolley_reached & rope_reached

        if full_target_reached.any():
            full_target_reached_index = np.where(full_target_reached)[0][0]
            full_time_to_target = full_target_reached_index * self.config['simulation']['time_step']
        else:
            full_time_to_target = len(q_values[0]) * self.config['simulation']['time_step']

        max_time = self.config['optimization'].get('max_time_to_target', 10.0)
        time_penalty = 1.0 if full_time_to_target <= max_time else (full_time_to_target / max_time) ** 2
        components['MSE_time'] = time_penalty * weights.get('time', 1.0)

        # Calculate the base fitness (negative sum because we want to maximize)
        base_fitness = -sum(components.values())

        return base_fitness, components

    def initialize_csv(self, header_fields: List[str]):
        """Initialize the CSV file with headers."""
        with open(self.csv_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header_fields)

    def save_results(self, generation: int, best_solution: np.ndarray, best_fitness: float,
                     component_metrics: Dict[str, float], additional_data: Dict[str, Any] = None):
        """
        Save optimization results to CSV.

        Args:
            generation: Current generation number
            best_solution: Best solution parameters
            best_fitness: Best fitness value
            component_metrics: Fitness component metrics
            additional_data: Any additional data to save
        """
        row_data = [generation, best_fitness]

        # Add additional metrics if provided
        if additional_data:
            for key, value in additional_data.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    row_data.extend(value)
                else:
                    row_data.append(value)

        # Add solution parameters
        row_data.extend(best_solution)

        # Add fitness components
        for component, value in component_metrics.items():
            row_data.append(value)

        # Save to CSV
        with open(self.csv_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # Update internal metrics
        self.metrics['generation'].append(generation)
        self.metrics['best_fitness'].append(best_fitness)
        self.metrics['parameters'].append(best_solution)
        self.metrics['component_metrics'].append(component_metrics)
        self.metrics['time_elapsed'].append(time.time())

    def get_solution_summary(self, best_solution: np.ndarray, best_fitness: float) -> str:
        """
        Generate a summary string of the optimization results.

        Args:
            best_solution: Best solution parameters
            best_fitness: Best fitness value

        Returns:
            Summary string
        """
        summary = [
            f"Optimization Results for {self.controller_name}",
            f"Best Fitness: {best_fitness:.6f}",
            f"Best Solution Parameters: {best_solution.tolist()}"
        ]

        # Apply parameters and run simulation for final metrics
        self._apply_parameters_to_controller(best_solution)
        self.simulation.simulate()
        sim_data = self._get_simulation_data()

        # Get key performance metrics
        q_values = sim_data['q_values']
        target_x = sim_data.get('target_x', 0)
        target_l = sim_data.get('target_l', 0)

        # Calculate time to target
        tolerance = self.config['crane_system']['target_positions'].get('target_tolerance', 0.05)
        trolley_reached = np.abs(q_values[0] - target_x) <= tolerance
        rope_reached = np.abs(q_values[1] - target_l) <= tolerance

        full_target_reached = trolley_reached & rope_reached

        if full_target_reached.any():
            full_target_reached_index = np.where(full_target_reached)[0][0]
            full_time_to_target = full_target_reached_index * self.config['simulation']['time_step']
            summary.append(f"Time to Target: {full_time_to_target:.2f} seconds")
        else:
            summary.append("Target not reached during simulation")

        # Add max swing angles
        if q_values.shape[0] > 2:
            max_theta1 = np.rad2deg(np.max(np.abs(q_values[2])))
            summary.append(f"Max Hook Angle: {max_theta1:.2f} degrees")
        if q_values.shape[0] > 3:
            max_theta2 = np.rad2deg(np.max(np.abs(q_values[3])))
            summary.append(f"Max Load Angle: {max_theta2:.2f} degrees")

        return "\n".join(summary)