from src.core.base import CraneBase
import numpy as np

class EnhancedCouplingController(CraneBase):
    def __init__(self, params: dict):
        super().__init__(params)

        # Controller parameters
        self.k_alpha1 = 2.8  # Gain for trolley positioning
        self.k_beta1 = 9.2   # Gain for swing suppression
        self.k_alpha2 = 70    # Gain for rope length control
        self.k_beta2 = 120    # Gain for rope velocity control
        self.k = 0.015         # Parameter for composite signal
        self.lambda_ = 0.01   # Parameter for rope length constraint

        # Set maximum rope length from YAML file
        # self.L = self.constraints_parameters['rope']['max']  # 1.005 meters
        self.L = 2

        # Target positions
        self.p_dx = self.targets_parameters['trolley']  # Desired trolley position
        self.p_dl = self.targets_parameters['rope']     # Desired rope length
        self.desired_trolley_pos = self.p_dx

        # Physical parameters
        self.g = self.physical_parameters['gravity']  # Gravity constant
        self.M_t = self.physical_parameters['masses']['trolley']  # Trolley mass
        self.m_h = self.physical_parameters['masses']['hook']     # Hook mass
        self.m_l = self.physical_parameters['masses']['load']     # Payload mass (may be unknown)
        self.L2 = self.physical_parameters['hook_to_load_distance']  # Rope length (hook to payload)

        # Damping and friction
        self.d_x = self.physical_parameters['damping']['trolley']  # Trolley damping
        self.d_r = self.physical_parameters['damping']['rope']     # Rope damping

        # Control limits
        self.control_limits = params['controller']['control_limits']

        # Time step
        self.dt = self.simulation_parameters['time_step']

        # Adaptive parameters (only used if payload mass is unknown)
        self.k_delta = 50     # Gain for payload mass estimation
        self.k_gamma = 10     # Gain for supplementary estimation
        self.m2_p_hat = 2.0   # Initial estimate of payload mass (kg)
        self.m2_s_hat = 0.0   # Initial supplementary estimate of payload mass (kg)

        # Flag to choose between known and unknown payload mass
        # self.payload_mass_known = params['controller'].get('known_load')
        self.payload_mass_known = True


    def compute_control_inputs(self, t, state, q_ddot):
        # Extract state variables
        q = state[:4]
        q_dot = state[4:8]
        x, l1, theta1, theta2 = q
        x_dot, l1_dot, theta1_dot, theta2_dot = q_dot

        # Compute composite signal Xi
        s1 = np.sin(theta1)
        s2 = np.sin(theta2)
        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        Xi = l1 * s1 + self.L2 * s2

        # Compute lambda1 and lambda2
        if self.payload_mass_known:
            # Payload mass is known
            lambda1 = self.k * (self.m_h + self.m_l)
            lambda2 = self.k * self.m_l
        else:
            # Payload mass is unknown, use estimates
            lambda1 = self.k * (self.m_h + self.m2_p_hat)
            lambda2 = self.k * self.m2_p_hat

        # Compute composite signal xi
        xi = x - lambda1 * l1 * s1 - lambda2 * self.L2 * s2

        # Compute error signal e_xi
        e_xi = xi - self.p_dx

        # Compute known function H
        H = x_dot - lambda1 * (l1_dot * s1 + l1 * c1 * theta1_dot) - lambda2 * self.L2 * c2 * theta2_dot

        # Compute control input Fx
        K = 1 + self.k * (self.M_t + self.m_h + self.m_l)
        Fx = -K * (self.k_alpha1 * e_xi + self.k_beta1 * H)

        # Compute error signal e1
        e1 = l1 - self.p_dl
        dot_e1 = l1_dot

        # Compute function f
        f = self.lambda_ * (e1 * ((self.L - 2 * l1) * self.p_dl + self.L * l1)) / (l1**2 * (self.L - l1)**2)

        # Compute control input Fl
        if self.payload_mass_known:
            # Payload mass is known
            Fl = -self.k_alpha2 * e1 - self.k_beta2 * dot_e1 - (self.m_h + self.m_l) * self.g - f
        else:
            # Payload mass is unknown, use estimates
            Fl = -self.k_alpha2 * e1 - self.k_beta2 * dot_e1 - (self.m_h + self.m2_p_hat + self.m2_s_hat) * self.g - f

            # Adaptive estimation of payload mass
            chi1 = self.integral_theta1  # Integral of theta1

            self.m2_s_hat = (self.k_gamma * chi1) / (1 + (chi1 * dot_e1)**2)

            # Update m2_p_hat using adaptation law
            dot_m2_p_hat = -(self.k * (self.k_alpha1 * Xi - self.k_beta1 * (l1_dot * s1 + l1 * c1 * theta1_dot + self.L2 * c2 * theta2_dot)) * H - self.g * dot_e1) / (self.k_delta - self.k_alpha1 * self.k**2 * Xi**2)
            self.m2_p_hat += dot_m2_p_hat * self.dt


        # Apply control limits (optional)
        # Fx = np.clip(Fx, self.control_limits['min'][0], self.control_limits['max'][0])
        # Fl = np.clip(Fl, self.control_limits['min'][1], self.control_limits['max'][1])

        # Return control inputs
        actual_control = np.array([Fx, Fl])
        return np.hstack((actual_control, np.zeros(2)))