import numpy as np


class TimeDomainCoherenceCalculator:
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.h = np.zeros(block_length)
        self.rxy = np.zeros(block_length)
        self.Rxx_inv = np.identity(block_length) * 1e-5
        self.var_y = 0.

    def calculate_rho(self, estimate, observation):
        self.var_y = self.lambda_coherence * self.var_y + observation**2

        self.rxy = self.lambda_coherence * self.rxy + (estimate * observation)

        kalman_gain = ((self.Rxx_inv @ estimate)) / (self.lambda_coherence + (estimate @ self.Rxx_inv @ estimate))

        self.h = self.h + kalman_gain * (observation - estimate @ self.h)

        if estimate.sum() != 0.:
            self.Rxx_inv = (self.Rxx_inv - np.outer(kalman_gain, (estimate @ self.Rxx_inv))) / self.lambda_coherence

        rho = np.sqrt(abs(self.rxy @ self.h) / (self.var_y + 1e-5))

        print(self.Rxx_inv)

        return rho
