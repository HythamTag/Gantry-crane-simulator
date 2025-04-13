from src.core.base import CraneBase
from src.utils.utils import *
import numpy as np


class SaturatedOFBController(CraneBase):
    def __init__(self, params: dict):
        super().__init__(params)

        # Controller parameters (as defined in the paper)
        self.kp = 11.5  # Proportional gain
        self.kd = 16  # Derivative gain
        self.alpha = -1.3  # Composite signal parameter

        # Trajectory parameters (if needed)
        self.k_a = params['trajectory']['k_a']
        self.k_v = params['trajectory']['k_v']
        self.epsilon = params['trajectory']['epsilon']

        # Target positions
        self.p_dx = self.targets_parameters['trolley']  # Target x position
        self.p_dl = self.targets_parameters['rope']  # Not used in this controller, but keeping to be consistent

        # Physical parameters
        self.g = self.physical_parameters['gravity']
        self.M_t = self.physical_parameters['masses']['trolley']
        self.m_h = self.physical_parameters['masses']['hook']
        self.m_l = self.physical_parameters['masses']['load']
        self.L1 = self.physical_parameters['rope_length']
        self.L2 = self.physical_parameters['hook_to_load_distance']

        # Control limits
        self.control_limits = params['controller']['control_limits']
        self.Umax = self.control_limits[0]  # Max force

        self.dt = self.simulation_parameters['time_step']

        # Additional parameters and state variables
        self.beta = (self.m_h + self.m_l) * self.L1 / (self.m_l * self.L2)
        self.mt = self.M_t + self.m_h + self.m_l
        self.aux_x = 0
        self.integral_e_phi = 0
        self.previous_state = None

    def compute_control_inputs(self, t, state, q_ddot):
        q = state[:4]
        q_dot = state[4:8]
        x, l1, theta1, theta2 = q
        x_dot, l1_dot, theta1_dot, theta2_dot = q_dot

        if self.previous_state is None:
            x_prev = x
            l1_prev = l1
            theta1_prev = theta1
            theta2_prev = theta2
        else:
            x_prev = self.previous_state[0]
            l1_prev = self.previous_state[1]
            theta1_prev = self.previous_state[2]
            theta2_prev = self.previous_state[3]

        self.previous_state = q
        # Target position
        x_d = compute_target_position_trajectory(t, self.params)

        # 1. Compute Composite Signal (phi):
        phi = x + self.alpha * (self.beta * np.sin(theta1) + np.sin(theta2))

        # 2. Compute Position Error:
        e_phi = phi - x_d

        # 3. Compute Auxiliary Signal (x_aux)
        self.aux_x += self.dt * (- self.kd * e_phi - self.aux_x)  # Changed this line

        # 4. Compute Control Input (u)
        u = (2 * self.kp / np.pi) * np.arctan(e_phi) - (2 * self.kd / np.pi) * np.arctan(self.aux_x + self.kd * e_phi)

        # Clipping the control to the defined max
        u = np.clip(u, -self.Umax, self.Umax)

        actual_control = np.array([u, 0])  # The second control input is set to 0 (as no vertical control)

        self.desired_trolley_pos = x_d
        return np.hstack((actual_control, np.zeros(2)))