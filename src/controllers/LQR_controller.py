import numpy as np
from typing import Dict, Any, Optional
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import logging

from src.utils.lqr_handler import LQRHandler
from src.core.base import CraneBase
from src.utils.utils import compute_target_position_trajectory, compute_target_velocity_trajectory
from src.Network.qr_neural_network import QRNeuralNetwork
from scipy import linalg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadialBasisNetwork:
    def __init__(self, n_centers=3, variance=0.1, eta=50, alpha=0.5):
        # Create centers for each state variable
        self.n_centers = n_centers
        self.n_states = 8  # Number of state variables
        centers_per_state = np.linspace(-1, 1, n_centers)

        # Create centers matrix with shape (n_states, n_centers)
        self.centers = np.array([centers_per_state for _ in range(self.n_states)])
        self.variance = variance
        self.eta = eta
        self.alpha = alpha
        self.w = np.zeros((self.n_states * n_centers, 1))

    def gaussian_mask(self, x):
        # Reshape x to match centers dimensions
        x_reshaped = x.reshape(-1, 1)

        # Calculate Gaussian for each state variable and center
        masks = []
        for i in range(self.n_states):
            diff = x_reshaped[i] - self.centers[i].reshape(-1, 1)
            mask = np.exp(-np.square(diff) / self.variance)
            masks.append(mask)

        # Combine all masks
        return np.vstack(masks)

    def gamma(self, x):
        # Adaptive gain based on system state
        return np.clip(np.linalg.norm(x) / 0.2, 0, 1)

    def get_control(self, state):
        # Process full state vector
        mask = self.gaussian_mask(state)
        return float((self.gamma(state) * self.w.T @ mask)[0, 0])

    def update_weights(self, state, dt):
        mask = self.gaussian_mask(state)
        # Use the error metric based on state deviation
        error = np.linalg.norm(state[2:4])  # Using angles as error metric
        dw = -self.eta * mask * error - self.alpha * self.w
        self.w += dw * dt


class AdaptiveNeuralLQRController(CraneBase):
    def __init__(self, params: Dict[str, Any], model: Any):
        super().__init__(params)
        self.model = model

        self.use_neural_network: bool = self.control_parameters.get('use_neural_network', False)
        self.use_radial: bool = self.control_parameters.get('use_radial', False)
        self.known_load: bool = self.control_parameters.get('known_load', True)

        self.initialize_controller()
        self.reset()

        if self.use_neural_network:
            csv_path: str = self.control_parameters.get('qr_data_csv_path', 'qr_data.csv')
            self.qr_nn = QRNeuralNetwork(csv_path)

        if self.use_radial:
            radial_params = self.control_parameters.get('radial_params', {})
            self.radial_network = RadialBasisNetwork(
                n_centers=radial_params.get('n_centers', 3),
                variance=radial_params.get('variance', 0.1),
                eta=radial_params.get('eta', 50),
                alpha=radial_params.get('alpha', 0.5)
            )

        if not self.known_load:
            self.ukf = self.initialize_ukf()

    def reset(self) -> None:
        self.prev_rope_length: Optional[float] = None
        self.integral_error = np.zeros(2)
        self.derivative_error = np.zeros(2)
        self.prev_error = np.array([0, self.targets_parameters['rope'] - self.initial_conditions_parameters['rope_length']])
        self.prev_time_i = self.prev_time_d = 0
        self.prev_input = np.zeros(2)
        self.mass_estimate: float = self.m_l if self.known_load else self.m_l  # Initial guess
        self.state_history: list = []
        self.control_history: list = []
        self.history_length: int = 50

    def initialize_controller(self) -> None:
        self.initialize_weights()
        self.initialize_control_options()
        self.lqr_controller = LQRHandler(
            use_integral=self.use_integral,
            use_derivative=self.use_derivative,
            dt=self.dt
        )
        self.m_t, self.m_h, self.m_l = self.get_masses()

    def initialize_weights(self) -> None:
        weights = self.control_parameters['weights']
        self.Q_state = np.diag(weights['state']).astype(np.float64)
        self.Q_int = np.diag(weights['integral']).astype(np.float64)
        self.Q_der = np.diag(weights['derivative']).astype(np.float64)
        self.R = np.diag(weights['control']).astype(np.float64)

    def initialize_control_options(self) -> None:
        self.use_integral: bool = self.control_parameters.get('use_integral', False)
        self.use_derivative: bool = self.control_parameters.get('use_derivative', False)
        self.enable_trajectory_velocity_tracking: bool = self.control_parameters.get('enable_trajectory_velocity_tracking', False)
        self.dt: float = self.simulation_parameters['time_step']

    def get_masses(self) -> tuple[float, float, float]:
        masses = self.physical_parameters['masses']
        return masses['trolley'], masses['hook'], masses['load']

    def initialize_ukf(self) -> UnscentedKalmanFilter:
        def fx(x: np.ndarray, dt: float) -> np.ndarray:
            return x  # State transition function (mass doesn't change)

        def hx(x: np.ndarray) -> np.ndarray:
            return x  # Measurement function (identity for this case)

        points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2., kappa=0)
        ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=self.dt, fx=fx, hx=hx, points=points)
        ukf.x = np.array([self.m_l])
        ukf.P *= 2.0
        ukf.R = np.array([[0.1]])
        ukf.Q = np.array([[0.01]])
        return ukf

    def compute_control_inputs(self, t: float, state: np.ndarray, q_ddot: np.ndarray) -> np.ndarray:
        trolley_pos, rope_length, *_ = state
        self.update_errors(t, trolley_pos, rope_length)

        if not self.known_load:
            self.mass_estimate = self.estimate_mass(state, self.prev_input)

        if self.should_recalculate_gain(rope_length):
            self.recalculate_gain(rope_length)

        error = self.compute_error(t, state)
        feedback_control = self.lqr_controller.get_control_input(error)
        feedforward_control = self.calculate_feedforward(t, state, q_ddot)

        control_inputs = feedforward_control + feedback_control[:2]

        # Add radial network contribution if enabled
        if self.use_radial:
            radial_control = np.array([self.radial_network.get_control(state), 0])
            control_inputs += radial_control
            self.radial_network.update_weights(state, self.dt)

        self.prev_input = self.apply_constraints(control_inputs)

        return np.hstack((self.prev_input, np.zeros(2)))

    def update_errors(self, t: float, trolley_pos: float, rope_length: float) -> None:
        self.desired_trolley_pos = self.compute_desired_trolley_position(t)
        desired_rope_length = self.targets_parameters['rope']

        current_trolley_error = self.desired_trolley_pos - trolley_pos
        current_rope_error = desired_rope_length - rope_length

        if self.use_integral:
            self.update_integral_error(t, current_trolley_error, current_rope_error)
        if self.use_derivative:
            self.update_derivative_error(t, current_trolley_error, current_rope_error)

    def should_recalculate_gain(self, rope_length: float) -> bool:
        return (self.prev_rope_length is None or
                abs(rope_length - self.prev_rope_length) > self.control_parameters['recalculation_threshold'])

    def recalculate_gain(self, rope_length: float) -> None:
        A, B, C, D = self.model.linearize(rope_length)
        if self.use_neural_network:
            Q, R = self.qr_nn.predict_qr(self.mass_estimate)
        else:
            Q, R = self.Q_state, self.R
        self.lqr_controller.compute_gain(A, B, C, D, Q, R, Q_int=self.Q_int, Q_der=self.Q_der)
        self.prev_rope_length = rope_length

    def update_history(self, state: np.ndarray, control_input: np.ndarray) -> None:
        self.state_history.append(state)
        self.control_history.append(control_input)
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)
            self.control_history.pop(0)

    def estimate_mass(self, state: np.ndarray, control_input: np.ndarray) -> float:
        self.update_history(state, control_input)

        if len(self.state_history) < 2:
            return self.mass_estimate

        x_ddot = (state[4] - self.state_history[-2][4]) / self.dt

        x, l, theta1, theta2 = state[:4]
        x_dot, l_dot, theta1_dot, theta2_dot = state[4:8]
        F_x, F_l = control_input[:2]

        M_t, m_h = self.m_t, self.m_h
        g = self.model.g

        denominator = x_ddot - l * theta1_dot ** 2 - g * np.sin(theta1)
        epsilon = 1e-12  # Small value to prevent division by zero
        m_l_est = (F_x - M_t * x_ddot - m_h * denominator) / (denominator + epsilon)

        self.ukf.predict()
        self.ukf.update(m_l_est)

        filtered_mass = self.ukf.x[0]
        estimated_mass = np.clip(filtered_mass, 0.5, 0.5)

        logger.info(f"Estimated mass: {estimated_mass}")
        return estimated_mass

    def update_integral_error(self, t: float, current_trolley_error: float, current_rope_error: float) -> None:
        dt = t - self.prev_time_i
        self.integral_error = np.array([
            self.update_single_integral_error(self.integral_error[0], current_trolley_error, dt),
            self.update_single_integral_error(self.integral_error[1], current_rope_error, dt)
        ])
        self.prev_time_i = t

    def update_single_integral_error(self, integral_error: float, current_error: float, dt: float) -> float:
        if self.control_parameters['is_anti_windup']:
            return 0 if integral_error != 0 and np.sign(current_error) != np.sign(integral_error) else integral_error + current_error * dt
        return integral_error + current_error * dt

    def update_derivative_error(self, t: float, current_trolley_error: float, current_rope_error: float) -> None:
        dt = t - self.prev_time_d
        if dt > 0:
            self.derivative_error = np.array([
                -(current_trolley_error - self.prev_error[0]) / dt,
                (current_rope_error - self.prev_error[1]) / dt
            ])
        self.prev_error = np.array([current_trolley_error, current_rope_error])
        self.prev_time_d = t

    def apply_constraints(self, control_inputs: np.ndarray) -> np.ndarray:
        return np.clip(control_inputs,
                       self.control_parameters['control_limits']['min'],
                       self.control_parameters['control_limits']['max'])

    def compute_error(self, t: float, state: np.ndarray) -> np.ndarray:
        desired_state = self.compute_desired_state(t)
        error = state - desired_state
        if self.use_integral:
            error = np.concatenate([error, self.integral_error])
        if self.use_derivative:
            error = np.concatenate([error, self.derivative_error])
        return error

    def compute_desired_state(self, t: float) -> np.ndarray:
        desired_trolley_pos = self.compute_desired_trolley_position(t)
        desired_trolley_vel = self.compute_desired_trolley_velocity(t) if self.enable_trajectory_velocity_tracking else 0
        desired_rope_length = self.targets_parameters['rope']
        return np.array([
            desired_trolley_pos,
            desired_rope_length,
            0, 0,  # Desired hook and load angles
            desired_trolley_vel,
            0, 0, 0,  # Desired velocities
        ])

    def compute_desired_trolley_position(self, t: float) -> float:
        return compute_target_position_trajectory(t, self.params)

    def compute_desired_trolley_velocity(self, t: float) -> float:
        return compute_target_velocity_trajectory(t, self.params)

    def calculate_feedforward(self, t: float, state: np.ndarray, q_ddot: np.ndarray) -> np.ndarray:
        _, rope_length, hook_angle, load_angle, *velocities = state
        x_dot, _, theta_1_dot, theta_2_dot = velocities
        x_ddot, _, theta_1_ddot, theta_2_ddot = q_ddot

        m_1, m_2 = self.m_h, self.mass_estimate
        g, l_2 = self.model.g, self.model.l_2

        # Friction compensation for trolley
        fr0x = self.model.physical_parameters['friction']['fr0x']
        epsx = self.model.physical_parameters['friction']['epsx']
        krx = self.model.physical_parameters['friction']['krx']
        d_x = self.model.physical_parameters['damping']['trolley']

        # Compute friction compensation terms
        ff_trolley = (
                fr0x * np.tanh(x_dot / epsx) +  # Coulomb friction
                krx * np.abs(x_dot) * x_dot +  # Quadratic friction
                d_x * x_dot  # Air Viscous damping
        )

        ff_rope = (m_1 * (-g * np.cos(hook_angle) - rope_length * theta_1_dot ** 2 + x_ddot * np.sin(hook_angle))
                   + m_2 * (l_2 * np.sin(hook_angle - load_angle) * theta_2_ddot - l_2 * np.cos(hook_angle - load_angle) * theta_2_dot ** 2
                            - g * np.cos(hook_angle) - rope_length * theta_1_dot ** 2 + x_ddot * np.sin(hook_angle)))

        return np.array([ff_trolley, ff_rope])

    def analyze_robustness(self, A, B, C, D):
        # Compute the sensitivity function
        S = linalg.inv(np.eye(A.shape[0]) + B @ self.K)

        # Compute the complementary sensitivity function
        T = np.eye(A.shape[0]) - S

        # Compute and print the H-infinity norms
        print(f"||S||_∞ = {np.max(np.abs(linalg.eigvals(S))):.4f}")
        print(f"||T||_∞ = {np.max(np.abs(linalg.eigvals(T))):.4f}")
        print(f"||KS||_∞ = {np.max(np.abs(linalg.eigvals(self.K @ S))):.4f}")