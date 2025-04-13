# model.py
"""
Core model components for the crane simulation system.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Callable
from functools import lru_cache

from src.core.base import ModelInterface, CraneBase
from src.core.physics_computations import (
    compute_system_energy,
    compute_state_derivative,
    apply_constraints
)


class CraneModel(CraneBase, ModelInterface):
    """
    Crane model implementation using physical principles.
    """

    def __init__(self, params: dict):
        """
        Initialize the crane model.

        Args:
            params: Configuration parameters
        """
        super().__init__(params)

        # Extract parameters
        self.g = self.physical_parameters['gravity']
        self.m_t = self.physical_parameters['masses']['trolley']
        self.m_h = self.physical_parameters['masses']['hook']
        self.m_l = self.physical_parameters['masses']['load']
        self.l_2 = self.physical_parameters['hook_to_load_distance']
        self.d_x = self.physical_parameters['damping']['trolley']
        self.d_r = self.physical_parameters['damping']['rope']
        self.d_h = self.physical_parameters['damping']['hook']
        self.d_l = self.physical_parameters['damping']['load']
        self.fr0x = self.physical_parameters['friction']['fr0x']
        self.epsx = self.physical_parameters['friction']['epsx']
        self.krx = self.physical_parameters['friction']['krx']

        # Constraints
        self.trolley_min = self.constraints_parameters['trolley']['min']
        self.trolley_max = self.constraints_parameters['trolley']['max']
        self.cable_min = self.constraints_parameters['rope']['min']
        self.cable_max = self.constraints_parameters['rope']['max']
        self.angle_min = -np.pi / 2
        self.angle_max = np.pi / 2

        self.apply_physical_constraints_flag = self.constraints_parameters['apply_physical_constraints']
        self.optimization_constrain = self.constraints_parameters['apply_optimization_constraints']

        # Integration parameters
        self.dt = self.simulation_parameters['time_step']

        # Pre-calculate common values
        self.total_load_mass = self.m_h + self.m_l
        self.gravity_force = -self.total_load_mass * self.g

    def get_initial_state(self) -> np.ndarray:
        """
        Get the initial state of the model.

        Returns:
            Initial state vector
        """
        return np.array([
            self.initial_conditions_parameters['trolley_position'],
            self.initial_conditions_parameters['rope_length'],
            self.initial_conditions_parameters['hook_angle'],
            self.initial_conditions_parameters['load_angle'],
            0, 0, 0, 0,  # Initial velocities
        ])

    def compute_state_derivative(self, state: np.ndarray, control_inputs: np.ndarray) -> np.ndarray:
        """
        Compute the derivatives of the state vector.

        Args:
            t: Current time
            state: Current state vector [q, q_dot]
            control_inputs: Control inputs vector

        Returns:
            Derivative of state vector [q_dot, q_ddot]
        """
        return compute_state_derivative(
            state, control_inputs, self.g, self.m_t, self.m_h, self.m_l, self.l_2,
            self.d_x, self.d_r, self.d_h, self.d_l, self.fr0x, self.epsx, self.krx,
        )

    def energy(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        """
        Compute the total energy of the system.

        Args:
            q: Position vector
            q_dot: Velocity vector

        Returns:
            Total energy of the system
        """
        return compute_system_energy(q, q_dot, self.g, self.m_t, self.m_h, self.m_l, self.l_2)

    def check_constraints(self, position: np.ndarray) -> bool:
        """
        Check if the position values satisfy the constraints.

        Args:
            position: Position vector [x, r, theta1, theta2]

        Returns:
            bool: True if constraints are satisfied, False otherwise
        """
        x, r, theta1, theta2 = position
        return (self.trolley_min <= x <= self.trolley_max and
                self.cable_min <= r <= self.cable_max and
                self.angle_min <= theta1 <= self.angle_max and
                self.angle_min <= theta2 <= self.angle_max)

    # @lru_cache(maxsize=8)
    def linearize(self, rope_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the model around an equilibrium point.
        Uses caching to avoid recalculating for the same rope length.

        Args:
            rope_length: Rope length at the linearization point

        Returns:
            Tuple of linearized system matrices (A, B, C, D)
        """
        g = self.g
        M_t = self.m_t
        m_h = self.m_h
        m_l = self.m_l
        L2 = self.l_2
        d_x = self.d_x
        d_r = self.d_r
        d_h = self.d_h
        d_l = self.d_l

        Fl = self.gravity_force

        A = np.zeros((8, 8))

        # Position to velocity mappings
        A[0, 4] = 1.0
        A[1, 5] = 1.0
        A[2, 6] = 1.0
        A[3, 7] = 1.0

        # Force equations
        A[4, 2] = -Fl / M_t
        A[4, 4] = -d_x / M_t
        A[4, 6] = d_h / (M_t * rope_length)

        A[5, 5] = -d_r / self.total_load_mass

        A[6, 2] = (Fl * M_t * m_l + Fl * m_h**2 + Fl * m_h * m_l -
                  M_t * g * m_h**2 - M_t * g * m_h * m_l) / (M_t * m_h * rope_length * self.total_load_mass)
        A[6, 3] = -Fl * m_l / (m_h * rope_length * self.total_load_mass)
        A[6, 4] = d_x / (M_t * rope_length)
        A[6, 6] = -d_h * (M_t + m_h) / (M_t * m_h * rope_length**2)
        A[6, 7] = d_l / (L2 * m_h * rope_length)

        A[7, 2] = -Fl / (L2 * m_h)
        A[7, 3] = Fl / (L2 * m_h)
        A[7, 6] = d_h / (L2 * m_h * rope_length)
        A[7, 7] = -d_l * self.total_load_mass / (L2**2 * m_h * m_l)

        B = np.zeros((8, 2))
        B[4, 0] = 1.0 / M_t
        B[5, 1] = 1.0 / self.total_load_mass
        B[6, 0] = -1.0 / (M_t * rope_length)

        C = np.eye(4, 8)
        D = np.zeros((4, 2))

        return A, B, C, D

    def step(self, state: np.ndarray, dt: float, control_inputs: np.ndarray, integration_method: str = 'rk4') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Take a single time step and return the new position, velocities, and accelerations.
        Includes robust error handling to prevent NaN/Inf propagation.

        Args:
            state: Current state vector (concatenation of position and velocity)
            dt: Time step size
            control_inputs: Control inputs vector
            integration_method: Integration method to use ('euler' or 'rk4')

        Returns:
            Tuple containing:
            - New position vector after the time step
            - New velocity vector after the time step
            - Acceleration vector
        """
        from src.core.integrator import Integrator

        # Input validation - ensure state and control inputs are finite
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(control_inputs)):
            # Return original state with zero accelerations if inputs are invalid
            return state[:4], state[4:8], np.zeros(4, dtype=state.dtype)

        # Get the integrator function
        integrator = Integrator.get_integrator(integration_method)

        # Define a wrapper that includes control inputs and captures accelerations
        accelerations = None

        def state_derivative_func(state_inner):
            nonlocal accelerations
            try:
                # Safeguard against invalid states
                if not np.all(np.isfinite(state_inner)):
                    # Return zeros if state is invalid to prevent propagation of NaNs
                    return np.zeros_like(state_inner)


                # print(f"control_inputs : {control_inputs.shape}")
                derivative = self.compute_state_derivative(state_inner, control_inputs)
                # print(f"derivative : {derivative.shape}")

                # Verify derivative is valid
                if not np.all(np.isfinite(derivative)):
                    # Fall back to a simple zero derivative
                    return np.zeros_like(state_inner)

                # Save accelerations from the derivative calculation
                accelerations = derivative[4:]  # Second half of derivative is accelerations
                return derivative

            except Exception as e:
                # In case of any exception, log it and return zeros
                print(f"Error in derivative calculation: {e}")
                return np.zeros_like(state_inner)

        # Try integration with error handling
        try:
            new_state = integrator(state, dt, state_derivative_func)

            # Verify integration result is valid
            if not np.all(np.isfinite(new_state)):
                # Fall back to simpler Euler integration
                try:
                    derivative = state_derivative_func(state)
                    new_state = state + dt * derivative

                    # If still invalid, keep original state
                    if not np.all(np.isfinite(new_state)):
                        new_state = state.copy()
                        # Ensure accelerations is defined
                        if accelerations is None:
                            accelerations = np.zeros(4, dtype=state.dtype)
                except:
                    # Last resort - keep original state
                    new_state = state.copy()
                    accelerations = np.zeros(4, dtype=state.dtype)
        except Exception as e:
            # Handle any integration errors
            print(f"Integration error: {e}")
            new_state = state.copy()
            accelerations = np.zeros(4, dtype=state.dtype)

        # Apply physical constraints if needed
        if self.apply_physical_constraints_flag:
            try:
                q, q_dot = new_state[:4], new_state[4:8]
                q, q_dot = apply_constraints(q, q_dot, self.trolley_min, self.trolley_max,
                                             self.cable_min, self.cable_max, self.angle_min, self.angle_max)
                new_state[:4] = q
                new_state[4:8] = q_dot
            except Exception as e:
                # If constraint application fails, log and continue with unconstrained state
                print(f"Error applying constraints: {e}")

        # Extract positions and velocities
        position = new_state[:4]
        velocities = new_state[4:8]

        # Final validation to ensure we return only finite values
        if not np.all(np.isfinite(position)):
            position = state[:4]
        if not np.all(np.isfinite(velocities)):
            velocities = state[4:8]
        if accelerations is None or not np.all(np.isfinite(accelerations)):
            accelerations = np.zeros(4, dtype=state.dtype)

        return position, velocities, accelerations