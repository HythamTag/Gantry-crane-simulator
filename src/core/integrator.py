import numpy as np
from numba import jit
from typing import Callable


class Integrator:
    """
    Class that handles different integration methods for the crane simulation.
    Separates integration logic from physical model calculations.
    """

    @staticmethod
    def euler_step(t: float, y: np.ndarray, h: float, f: Callable) -> np.ndarray:
        """
        Perform one step of Euler integration.

        Args:
            t: Current time
            y: Current state vector
            h: Time step size
            f: Function that computes state derivatives

        Returns:
            New state vector after integration step
        """
        return y + h * f(t, y)

    @staticmethod
    def rk4_step(t: float, y: np.ndarray, h: float, f: Callable) -> np.ndarray:
        """
        Perform one step of 4th-order Runge-Kutta integration.

        Args:
            t: Current time
            y: Current state vector
            h: Time step size
            f: Function that computes state derivatives

        Returns:
            New state vector after integration step
        """
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)

        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def get_integrator(method: str):
        """
        Factory method to get the appropriate integration function.

        Args:
            method: Integration method name ('euler' or 'rk4')

        Returns:
            Integration step function
        """
        if method.lower() == 'euler':
            return Integrator.euler_step
        elif method.lower() == 'rk4':
            return Integrator.rk4_step
        else:
            raise ValueError(f"Unknown integration method: {method}")