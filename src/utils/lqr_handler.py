import numpy as np
from scipy import linalg
from functools import lru_cache
import control as ct

class LQRHandler:
    def __init__(self, use_integral, use_derivative, dt):
        self.K = None
        self.scale_factor = None
        self.use_integral = use_integral
        self.use_derivative = use_derivative
        self.dt = dt

        self.prev_input = None

    def compute_gain(self, A, B, C, D, Q, R, Q_int=None, Q_der=None):
        # Convert numpy arrays to tuples for hashing
        A_tuple = tuple(map(tuple, A))
        B_tuple = tuple(map(tuple, B))
        C_tuple = tuple(map(tuple, C))
        D_tuple = tuple(map(tuple, D))
        Q_tuple = tuple(map(tuple, Q))
        R_tuple = tuple(map(tuple, R))
        Q_int_tuple = tuple(map(tuple, Q_int)) if Q_int is not None else None
        Q_der_tuple = tuple(map(tuple, Q_der)) if Q_der is not None else None

        return self._compute_gain_cached(A_tuple, B_tuple, C_tuple, D_tuple, Q_tuple, R_tuple, Q_int_tuple, Q_der_tuple)

    # @lru_cache(maxsize=100000)
    def _compute_gain_cached(self, A, B, C, D, Q, R, Q_int=None, Q_der=None):
        try:
            # Convert tuples back to numpy arrays
            A, B, C, D, Q, R = map(np.array, (A, B, C, D, Q, R))
            Q_int = np.array(Q_int) if Q_int is not None else None
            Q_der = np.array(Q_der) if Q_der is not None else None

            if self.use_integral:
                A, B = self._augment_matrices_for_integral(A, B, C)
                Q = self._augment_Q_for_integral(Q, Q_int)

            if self.use_derivative:
                A, B = self._augment_matrices_for_derivative(A, B, C, self.dt)
                Q = self._augment_Q_for_derivative(Q, Q_der)

            if self.dt is not None:
                A, B = self._discretize_system(A, B, self.dt)

            # Scale matrices for numerical stability
            A_scaled, B_scaled, self.scale_factor = self._scale_matrices(A, B)

            # Increase regularization for better numerical stability
            Q_reg = self._regularize_matrix(Q, epsilon=1e-4)
            R_reg = R + 1e-4 * np.eye(R.shape[0])  # Also regularize R

            # Check system controllability
            n = A_scaled.shape[0]
            controllability_matrix = np.zeros((n, n * B_scaled.shape[1]))

            # Build controllability matrix
            temp = B_scaled.copy()
            controllability_matrix[:, :B_scaled.shape[1]] = temp

            for i in range(1, n):
                temp = A_scaled @ temp
                controllability_matrix[:, i * B_scaled.shape[1]:(i + 1) * B_scaled.shape[1]] = temp

            rank = np.linalg.matrix_rank(controllability_matrix)
            if rank < n:
                print(f"Warning: System not fully controllable. Rank: {rank}/{n}")
                # Continue anyway as we want to try computing a solution

            # Try solving with increased regularization
            P = linalg.solve_discrete_are(A_scaled, B_scaled, Q_reg, R_reg)

            # Compute the gain
            K = np.linalg.multi_dot([linalg.inv(R_reg + B_scaled.T @ P @ B_scaled), B_scaled.T, P, A_scaled])

            # print(f"k size {K.shape}")

            # Scale back the gain
            self.K = K / self.scale_factor

            # Verify K has reasonable values
            if not np.all(np.isfinite(self.K)):
                raise ValueError("LQR gain contains NaN or infinite values")

            return self.K

        except Exception as e:
            print(f"Error in LQR computation: {e}")
            # Create a zero gain matrix with appropriate dimensions
            if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
                self.K = np.zeros((B.shape[1], A.shape[0]))
            else:
                # If dimensions are unknown, create a default matrix
                print("Unable to determine appropriate dimensions for K matrix")
                self.K = np.zeros((2, 4))  # Common dimensions for simple systems

            return self.K
    def get_control_input(self, error):
        try:
            if self.K is None:
                raise ValueError("LQR gain matrix K has not been computed yet.")

            self.prev_input = -self.K @ error

            # print(f"prev_input :{self.prev_input.shape}")
            #
            # print(f"k size {self.K.shape}")
            #
            # print(f"error size {error.shape}")
            #
            # print(f"##################")


            return -self.K @ error
        except Exception as e:
            print(f"Error in LQR computation: {e}")
            print(error.shape)
            return self.prev_input

    def _augment_matrices_for_integral(self, A, B, C):
        A_aug = np.block([
            [A, np.zeros((A.shape[0], 2))],
            [-C[:2], np.zeros((2, A.shape[1] - C.shape[1] + 2))],
        ])
        B_aug = np.vstack([B, np.zeros((2, B.shape[1]))])
        return A_aug, B_aug

    def _augment_matrices_for_derivative(self, A, B, C, dt):
        A_aug = np.block([
            [A, np.zeros((A.shape[0], 2))],
            [-C[:2]/dt, np.zeros((2, A.shape[1] - C.shape[1] + 2))],
        ])
        B_aug = np.vstack([B, np.zeros((2, B.shape[1]))])
        return A_aug, B_aug

    def _augment_Q_for_derivative(self, Q, Q_der):
        return np.block([
            [Q, np.zeros((Q.shape[0], 2))],
            [np.zeros((2, Q.shape[1])), Q_der]
        ])

    def _augment_Q_for_integral(self, Q, Q_int):
        return np.block([
            [Q, np.zeros((Q.shape[0], 2))],
            [np.zeros((2, Q.shape[1])), Q_int]
        ])

    def _discretize_system(self, A, B, dt):
        sys_c = ct.ss(A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1])))
        sys_d = ct.c2d(sys_c, dt)
        return sys_d.A, sys_d.B

    def _scale_matrices(self, A, B):
        scale_factor = np.max(np.abs(np.hstack((A, B))))
        A_scaled = A / scale_factor
        B_scaled = B / scale_factor
        return A_scaled, B_scaled, scale_factor

    def _regularize_matrix(self, Q, epsilon=1e-6):
        return Q + epsilon * np.eye(Q.shape[0])