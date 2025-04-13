"""
Base classes for the crane simulation system.
"""
from abc import ABC, abstractmethod


class CraneBase:
    """Base class for all crane-related components."""

    def __init__(self, params: dict):
        """
        Initialize the crane base with configuration parameters.

        Args:
            params: Dictionary containing configuration parameters
        """
        self.params = params
        self.physical_parameters = params['crane_system']['physical_params']
        self.control_parameters = params['controller']
        self.constraints_parameters = params.get('crane_system', {}).get('constraints', {})
        self.initial_conditions_parameters = params['crane_system']['initial_conditions']
        self.targets_parameters = params['crane_system']['target_positions']
        self.simulation_parameters = params['simulation']
        self.visualization_parameters = params['visualizer']


class ModelInterface(ABC):
    """Interface for crane dynamic models."""

    @abstractmethod
    def get_initial_state(self):
        """
        Get the initial state of the model.

        Returns:
            Initial state vector
        """
        pass

    @abstractmethod
    def compute_state_derivative(self, state, control_inputs):
        """
        Compute the derivatives of the state vector.

        Args:
            state: Current state vector
            control_inputs: Control inputs vector

        Returns:
            Current state Acceleration
        """
        pass

    @abstractmethod
    def energy(self, q, q_dot):
        """
        Compute the total energy of the system.

        Args:
            q: Position vector
            q_dot: Velocity vector

        Returns:
            Total energy of the system
        """
        pass


class ControllerInterface(ABC):
    """Interface for crane controllers."""

    @abstractmethod
    def compute_control_inputs(self, t, state, previous_acceleration):
        """
        Compute control inputs based on current state.

        Args:
            t: Current time
            state: Current state vector
            previous_acceleration: Previous acceleration vector

        Returns:
            Control inputs vector
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset controller state."""
        pass


class IntegratorInterface(ABC):
    """Interface for numerical integrators."""

    @abstractmethod
    def step(self, t, y, h, f):
        """
        Perform one integration step.

        Args:
            t: Current time
            y: Current state vector
            h: Time step size
            f: Function that computes state derivatives

        Returns:
            New state vector after integration step
        """
        pass


class SimulationObserver(ABC):
    """Observer interface for simulation data collection."""

    @abstractmethod
    def update(self, t, state, derivatives, control_inputs):
        """
        Update observer with current simulation data.

        Args:
            t: Current time
            state: Current state vector
            derivatives: State derivatives
            control_inputs: Control inputs vector
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset observer data."""
        pass