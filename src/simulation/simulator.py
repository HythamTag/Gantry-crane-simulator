import numpy as np
import yaml

from src.controllers.DPUG_controller import DPUGController
from src.controllers.LQR_controller import AdaptiveNeuralLQRController
from src.controllers.Amplitude_saturated_controller import SaturatedOFBController
from src.controllers.enhanced_coupling_adaptive_controller import EnhancedCouplingController
from src.core.base import CraneBase
from src.core.integrator import Integrator
from src.core.model import CraneModel
from src.visualization.visualizer import CraneVisualizer


class CraneSimulation(CraneBase):
    def __init__(self, params, render=None):
        super().__init__(params)
        self.params = params

        if render == False:
            self.params["visualizer"]["render"] = False
            self.params["visualizer"]["save_plots"] = False

        self.model = CraneModel(params)

        self.controller_type = self.simulation_parameters['controller_type']
        self.t_end = self.simulation_parameters['duration']
        self.dt = self.simulation_parameters['time_step']
        self.integration_method = self.simulation_parameters['integration_method']

        # Setup the integrator
        self.integrator = Integrator.get_integrator(self.integration_method)

        # Initialize the controller
        self._init_controller()

        self.visualizer = CraneVisualizer(self.params)

        # Get initial state and targets
        self.initial_state = self.model.get_initial_state()
        # Extract just the position part (first 4 elements)
        self.initial_position = self.initial_state[:4]
        # Extract just the velocity part (last 4 elements)
        self.initial_velocity = self.initial_state[4:8]

        self.target = [self.targets_parameters['trolley'], self.targets_parameters['rope']]

        self.num_steps = int(self.t_end / self.dt) + 1

        # Initialize flags for constraint tracking
        self.constraints_violated = False
        self.violation_index = -1

        self.reset()

    def _init_controller(self):
        """Initialize the appropriate controller based on configuration."""
        if self.controller_type == 'DPUG':
            self.controller = DPUGController(self.params)
        elif self.controller_type == 'ECA':
            self.controller = EnhancedCouplingController(self.params)
        elif self.controller_type == 'Amp':
            self.controller = SaturatedOFBController(self.params)
        elif self.controller_type == 'LQR':
            self.controller = AdaptiveNeuralLQRController(self.params, self.model)
        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")

    def reset(self):
        """Reset simulation state variables."""
        self.t_values = np.linspace(0, self.t_end, self.num_steps)

        # Position only (4 elements)
        self.q_values = np.zeros((self.num_steps, 4))
        # Velocity (4 elements)
        self.q_dot_values = np.zeros((self.num_steps, 4))
        # Acceleration (4 elements)
        self.q_ddot_values = np.zeros((self.num_steps, 4))
        # Jerk (4 elements)
        self.q_dddot_values = np.zeros((self.num_steps, 4))

        self.trajectory_pos_vector = np.zeros((self.num_steps, 1))
        self.integral_error_values = np.zeros((self.num_steps, 2))
        self.control_inputs = np.zeros((self.num_steps, 4))
        self.energy = np.zeros(self.num_steps)

        # Set initial position and velocity
        self.q_values[0] = self.initial_position
        self.q_dot_values[0] = self.initial_velocity

        # Reset constraint violation tracking
        self.constraints_violated = False
        self.violation_index = -1

        if self.controller_type == 'LQR':
            self.controller.reset()

    def simulate(self, print_data=None):
        """
        Run the simulation with the configured parameters.

        Args:
            print_data: Whether to print and visualize results

        Returns:
            bool: True if simulation completed successfully, False if constraints were violated
        """
        self.reset()

        # Run the simulation loop
        self._run_simulation()

        # If constraints were violated, handle it appropriately
        if self.constraints_violated:
            print(f"Simulation stopped due to constraint violation at step {self.violation_index}, time {self.t_values[self.violation_index]:.4f}s")
            # Truncate results to the point of violation
            self.t_values = self.t_values[:self.violation_index + 1]
            self.q_values = self.q_values[:self.violation_index + 1]
            self.q_dot_values = self.q_dot_values[:self.violation_index + 1]
            self.q_ddot_values = self.q_ddot_values[:self.violation_index + 1]
            self.q_dddot_values = self.q_dddot_values[:self.violation_index + 1]
            self.trajectory_pos_vector = self.trajectory_pos_vector[:self.violation_index + 1]
            self.integral_error_values = self.integral_error_values[:self.violation_index + 1]
            self.control_inputs = self.control_inputs[:self.violation_index + 1]
            self.energy = self.energy[:self.violation_index + 1]

        # Compute derivatives
        self.compute_jerk()

        if print_data:
            self.visualize_results()
            self.print_results()
            self.check_targets_reached()

        # Return success status (True if completed without constraint violations)
        return not self.constraints_violated

    def _run_simulation(self):
        """Run the time-stepping simulation with the chosen integration method."""
        # First state is already set in reset()

        for i in range(1, self.num_steps):
            current_time = self.t_values[i - 1]

            # Get current position and velocity
            current_position = self.q_values[i - 1]
            current_velocity = self.q_dot_values[i - 1]

            # Create full state by concatenating position and velocity
            current_state = np.concatenate((current_position, current_velocity))

            # Check constraints if needed
            if self.model.optimization_constrain and not self.model.check_constraints(current_position):
                print(f"Constraints violated at time: {current_time:.4f}s")
                # Set flags to indicate constraint violation
                self.constraints_violated = True
                self.violation_index = i - 1
                # Break out of the loop to stop the simulation
                break

            # Get previous acceleration for controllers that need it
            previous_q_ddot = self.q_ddot_values[max(0, i - 2)]

            # Compute control inputs
            control_inputs = self.controller.compute_control_inputs(current_time, current_state, previous_q_ddot)

            # Store control inputs
            self.control_inputs[i - 1] = control_inputs

            # Use model to take a step and get all needed data
            new_position, new_velocity, new_acceleration = self.model.step(
                current_state,
                self.dt,
                control_inputs,
                self.integration_method
            )

            # Store all the results
            self.q_values[i] = new_position
            self.q_dot_values[i] = new_velocity
            self.q_ddot_values[i] = new_acceleration

            # Update additional simulation data
            self._update_simulation_data(current_time, i, new_position, new_velocity, new_acceleration, control_inputs)

    def _update_simulation_data(self, time, index, position, velocity, acceleration, control_inputs):
        """
        Update simulation data for the current time step.

        Args:
            time: Current simulation time
            index: Current time step index
            position: Position vector
            velocity: Velocity vector
            acceleration: Acceleration vector
            control_inputs: Control inputs
        """
        try:
            self.trajectory_pos_vector[index] = self.controller.desired_trolley_pos
            self.integral_error_values[index] = np.zeros(2)  # Update this if you're tracking integral error
            self.energy[index] = self.model.energy(position, velocity)
        except Exception as e:
            print(f"Error in update_simulation_data method: {e}")
            raise

    def compute_jerk(self):
        """Compute jerk using finite differences on acceleration data."""
        # Handle empty arrays (in case of early termination)
        if len(self.q_ddot_values) < 3:
            return

        # Compute jerk using finite differences
        self.q_dddot_values[1:-1] = (self.q_ddot_values[2:] - self.q_ddot_values[:-2]) / (2 * self.dt)
        # Set the first and last jerk values (can be improved with one-sided differences if needed)
        if len(self.q_dddot_values) > 0:
            self.q_dddot_values[0] = self.q_dddot_values[1] if len(self.q_dddot_values) > 1 else 0
        if len(self.q_dddot_values) > 1:
            self.q_dddot_values[-1] = self.q_dddot_values[-2]

    def visualize_results(self):
        """Generate visualizations of the simulation results."""
        if self.visualization_parameters['render']:
            # Combine position and velocity for animation if needed
            full_state = np.zeros((len(self.t_values), 8))
            full_state[:, :4] = self.q_values  # Position
            full_state[:, 4:] = self.q_dot_values  # Velocity

            self.visualizer.animate(self.t_values, full_state, self.control_inputs, np.zeros_like(self.t_values))

        if self.visualization_parameters['save_plots']:
            self.visualizer.plot_results(
                self.t_values,
                self.q_values,  # Position
                self.q_dot_values,  # Velocity
                self.q_ddot_values,  # Acceleration
                self.q_dddot_values,  # Jerk
                np.zeros(len(self.t_values)),  # Adjusted for possible truncation
                self.control_inputs,
                self.energy,
                self.trajectory_pos_vector
            )

    def print_results(self):
        """Print key results from the simulation."""
        if len(self.q_values) == 0:
            print("No valid simulation data to print.")
            return

        max_theta1 = np.rad2deg(np.max(np.abs(self.q_values[:, 2])))
        max_theta2 = np.rad2deg(np.max(np.abs(self.q_values[:, 3])))
        print(f"Maximum absolute value of theta1: {max_theta1:.4f} degrees")
        print(f"Maximum absolute value of theta2: {max_theta2:.4f} degrees")

        # Print max jerk values
        if len(self.q_dddot_values) > 0:
            max_jerks = np.max(np.abs(self.q_dddot_values), axis=0)
            print(f"Maximum absolute jerk values: {max_jerks}")

        # Print constraint violation info if applicable
        if self.constraints_violated:
            print(f"Simulation terminated early due to constraint violation at t={self.t_values[self.violation_index]:.4f}s")

    def check_targets_reached(self):
        """Check if the target positions were reached during simulation."""
        if len(self.q_values) == 0:
            print("No valid simulation data to check targets.")
            return None

        tolerance = self.targets_parameters['target_tolerance']

        trolley_target_reached_index = np.where((np.abs(self.q_values[:, 0] - self.targets_parameters['trolley']) <= tolerance))[0]
        rope_target_reached_index = np.where((np.abs(self.q_values[:, 1] - self.targets_parameters['rope']) <= tolerance))[0]

        if len(trolley_target_reached_index) > 0:
            trolley_time_to_target = trolley_target_reached_index[0] * self.dt
        else:
            trolley_time_to_target = None

        if len(rope_target_reached_index) > 0:
            rope_time_to_target = rope_target_reached_index[0] * self.dt
        else:
            rope_time_to_target = None

        if trolley_time_to_target is not None and rope_time_to_target is not None:
            full_time_to_target = max(trolley_time_to_target, rope_time_to_target)
            print(f"Full time to target: {full_time_to_target:.4f} seconds")
            print(f"Trolley time to target: {trolley_time_to_target:.4f} seconds")
            print(f"Rope time to target: {rope_time_to_target:.4f} seconds")
            return full_time_to_target
        else:
            print("Targets were not reached within the simulation duration.")
            return None

        print("###############################################################\n")

    def get_optimization_metrics(self):
        """
        Calculate metrics for optimization.

        Returns:
            dict: Dictionary of optimization metrics
        """
        # This method can be used to return metrics for optimization
        # Handle early termination cases
        if len(self.q_dddot_values) == 0 or len(self.q_ddot_values) == 0:
            return {
                'success': False,
                'constraints_violated': True,
                'violation_time': self.t_values[self.violation_index] if self.violation_index >= 0 else 0
            }

        time_to_target = self.check_targets_reached()

        return {
            'success': not self.constraints_violated,
            'max_jerk': np.max(np.abs(self.q_dddot_values)),
            'mean_jerk': np.mean(np.abs(self.q_dddot_values)),
            'max_acceleration': np.max(np.abs(self.q_ddot_values)),
            'mean_acceleration': np.mean(np.abs(self.q_ddot_values)),
            'time_to_target': time_to_target,
            'energy': np.sum(self.energy),
            'duration': len(self.t_values) * self.dt,
            'constraints_violated': self.constraints_violated,
            'violation_time': self.t_values[self.violation_index] if self.violation_index >= 0 else None
        }