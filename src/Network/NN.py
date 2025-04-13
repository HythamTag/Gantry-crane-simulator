import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import solve_continuous_are
from typing import Dict, Any
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
import pandas as pd
import ast

from src.controllers.crane_controllers import LQRController


class QRParametersNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(QRParametersNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

class AdvancedNeuralGainSchedulingLQR(LQRController):
    def __init__(self, params: Dict[str, Any], model: Any, known_mass: bool = True, csv_path: str = 'qr_data.csv'):
        super().__init__(params, model)
        self.known_mass = known_mass
        self.csv_path = csv_path
        self.qr_data = self.load_qr_data()
        self.nn_model = self.build_and_train_nn()
        self.mass_estimate = self.m_l  # Initial estimate

        # Parameters for advanced mass estimation
        self.mass_history = []
        self.control_history = []
        self.state_history = []
        self.history_length = 50  # Number of past data points to keep
        self.ukf = self.initialize_ukf()

    def load_qr_data(self):
        df = pd.read_csv(self.csv_path)
        self.masses = df['Mass'].values
        self.q_values = df['Q_values'].apply(ast.literal_eval).values
        self.r_values = df['R_values'].apply(ast.literal_eval).values
        return df

    def build_and_train_nn(self):
        X = torch.FloatTensor(self.masses.reshape(-1, 1))
        y = torch.FloatTensor(np.hstack((self.q_values, self.r_values)))

        model = QRParametersNN(1, y.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()  # Set the model to evaluation mode
        return model



    def compute_control_inputs(self, t, state, q_ddot):
        if self.known_mass:
            mass = self.model.m_l
        else:
            mass = self.estimate_mass(state, self.prev_input)

        # Get Q and R parameters from neural network
        Q, R = self.get_q_r_parameters(mass)

        # Linearize the system
        A, B, C, D = self.model.linearize(state[1])  # Assuming state[1] is the rope length

        # Solve Riccati equation
        P = solve_continuous_are(A, B, Q, R)

        # Compute LQR gain
        K = np.linalg.inv(R) @ B.T @ P

        # Compute error
        error = self.compute_error(t, state)

        # Compute control input
        control_input = -K @ error

        # Apply constraints
        actual_control = self.apply_constraints(control_input[:2])

        self.prev_input = actual_control
        self.desired_trolley_pos = self.compute_desired_trolley_position(t)

        return np.hstack((actual_control, np.zeros(2)))

    def update_history(self, state, control_input):
        self.state_history.append(state)
        self.control_history.append(control_input)
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)
            self.control_history.pop(0)
