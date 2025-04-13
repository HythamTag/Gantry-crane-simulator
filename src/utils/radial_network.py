import numpy as np


class RadialBasisNetwork:
    def __init__(self, centers=np.array([-0.5, 0, 0.5]), variance=0.1, eta=50, alpha=0.5):
        self.centers = centers.reshape(-1, 1)
        self.variance = variance
        self.eta = eta  # Learning rate
        self.alpha = alpha  # Damping rate
        self.w = np.zeros((len(centers), 1))  # Initial weights

    def gaussian_mask(self, x):
        diff = x - self.centers
        return np.exp(-np.square(diff) / self.variance)

    def gamma(self, x):
        return np.clip(np.abs(x) / 0.2, 0, 1)

    def get_control(self, state):
        x_features = state[1:].reshape(-1, 1)
        mask = self.gaussian_mask(x_features)
        return float((self.gamma(state[2]) * self.w.T @ mask)[0, 0])

    def update_weights(self, state, dt):
        x_features = state[1:].reshape(-1, 1)
        mask = self.gaussian_mask(x_features)
        dw = -self.eta * mask * state[3] - self.alpha * self.w
        self.w += dw * dt