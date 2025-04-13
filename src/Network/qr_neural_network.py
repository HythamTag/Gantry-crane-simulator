import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import ast

class QRParametersNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

class QRNeuralNetwork:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.qr_data = self.load_qr_data()
        self.nn_model = self.build_and_train_nn()

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

        for _ in range(1000):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    def predict_qr(self, mass):
        with torch.no_grad():
            qr_params = self.nn_model(torch.FloatTensor([[mass]])).numpy()[0]
        q_dim = len(self.q_values[0])
        return np.diag(qr_params[:q_dim]), np.diag(qr_params[q_dim:])