from src.core.base import CraneBase
from src.utils.utils import *


class DPUGController(CraneBase):
    def __init__(self, params: dict):
        super().__init__(params)

        # Controller parameters
        # Nonlinear controller parameters
        self.k_1 = 5  # You might want to add these to your config
        self.k_2 = 0.5
        self.k_px = 4
        self.k_dx = 50
        self.k_pl = 12
        self.k_dl = 18

        # Trajectory parameters
        self.k_a = params['trajectory']['k_a']
        self.k_v = params['trajectory']['k_v']
        self.epsilon = params['trajectory']['epsilon']

        # Target positions
        self.p_dx = self.targets_parameters['trolley']
        self.p_dl = self.targets_parameters['rope']

        # Physical parameters
        self.g = self.physical_parameters['gravity']
        self.M_t = self.physical_parameters['masses']['trolley']
        self.m_h = self.physical_parameters['masses']['hook']
        self.m_l = self.physical_parameters['masses']['load']
        self.L2 = self.physical_parameters['hook_to_load_distance']

        # Damping and friction
        self.d_x = self.physical_parameters['damping']['trolley']
        self.d_r = self.physical_parameters['damping']['rope']
        self.fr0x = self.physical_parameters['friction']['fr0x']
        self.epsx = self.physical_parameters['friction']['epsx']
        self.krx = self.physical_parameters['friction']['krx']

        # Control limits
        self.control_limits = params['controller']['control_limits']

        self.dt = self.simulation_parameters['time_step']

        self.integral_theta1 = 0
        self.integral_theta2 = 0

    def compute_control_inputs(self, t, state, q_ddot):
        q = state[:4]
        q_dot = state[4:8]
        x, l1, theta1, theta2 = q
        x_dot, l1_dot, theta1_dot, theta2_dot = q_dot
        x_ddot, l1_ddot, theta1_ddot, theta2_ddot = q_ddot

        x_r = compute_target_position_trajectory(t, self.params)
        x_r_dot = compute_target_velocity_trajectory(t, self.params)
        x_r_ddot = compute_target_acceleration_trajectory(t, self.params)

        self.integral_theta1 += theta1 * self.dt
        self.integral_theta2 += theta2 * self.dt

        x_c = x_r + self.k_1 * l1 * self.integral_theta1 + self.k_2 * self.L2 * self.integral_theta2
        x_c_dot = x_r_dot + self.k_1 * (l1_dot * self.integral_theta1 + l1 * theta1) + self.k_2 * self.L2 * theta2
        x_c_ddot = x_r_ddot + self.k_1 * (l1_ddot * self.integral_theta1 + 2 * l1_dot * theta1 + l1 * theta1_dot) + self.k_2 * self.L2 * theta2_dot

        e_x = x - x_c
        e_l1 = l1 - self.p_dl
        e_dot_x = x_dot - x_c_dot
        e_dot_l1 = l1_dot

        F_rx = self.fr0x * np.tanh(x_dot / self.epsx) + self.krx * np.abs(x_dot) * x_dot

        Fx = F_rx + self.d_x * x_dot - self.k_px * e_x - self.k_dx * e_dot_x + (
                self.M_t + self.m_h + self.m_l) * x_c_ddot

        Fl = self.d_r * l1_dot - self.k_pl * e_l1 - self.k_dl * e_dot_l1 - (self.m_h + self.m_l) * (
                self.g - x_c_ddot * np.sin(theta1))

        # # Apply control limits
        # Fx = np.clip(Fx, -self.control_limits[0], self.control_limits[0])
        # Fl = np.clip(Fl, -self.control_limits[1], self.control_limits[1])

        actual_control = np.array([Fx, Fl])

        self.desired_trolley_pos = x_r

        return np.hstack((actual_control, np.zeros(2)))