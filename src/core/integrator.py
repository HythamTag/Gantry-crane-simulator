import numpy as np
from numba import jit
from typing import Callable


class Integrator:
    """
    Class that handles different integration methods for the crane simulation.
    Separates integration logic from physical model calculations.
    """

    @staticmethod
    def euler_step(y: np.ndarray, h: float, f: Callable) -> np.ndarray:
        """
        Perform one step of Euler integration.

        Args:
            y: Current state vector
            h: Time step size
            f: Function that computes state derivatives

        Returns:
            New state vector after integration step
        """
        return y + h * f(y)

    @staticmethod
    def rk4_step(y: np.ndarray, h: float, f: Callable) -> np.ndarray:
        """
        Perform one step of 4th-order Runge-Kutta integration.

        Args:
            y: Current state vector
            h: Time step size
            f: Function that computes state derivatives

        Returns:
            New state vector after integration step
        """
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)

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


# Create JIT-compiled versions for direct use
@jit(nopython=True)
def _jit_euler_step(y, h, deriv_vals):
    """
    JIT-compiled Euler integration step. Note that this takes pre-computed
    derivative values instead of a function to compute them.
    """
    return y + h * deriv_vals


@jit(nopython=True)
def _jit_rk4_step(y, h, f, *args):
    """
    JIT-compiled RK4 integration step.
    This assumes f is also a JIT-compiled function.
    """
    k1 = f(y, *args)
    k2 = f(y + 0.5 * h * k1, *args)
    k3 = f(y + 0.5 * h * k2, *args)
    k4 = f(y + h * k3, *args)

    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)