import numpy as np
from numpy.fft import fft, ifft

np.seterr(all='raise')

class CoherenceCalculator:
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.coherence_12 = np.zeros(block_length)
        # self.coherence_12 = 0.
        # self.coherence_11 = np.ones(block_length)
        # self.coherence_22 = np.ones(block_length)
        self.coherence_11 = np.zeros(block_length)
        self.coherence_22 = np.zeros(block_length)
        # self.coherence_22 = 0.

    def calculate_rho(self, estimate, observation):
        self.coherence_12 = self.lambda_coherence * self.coherence_12 + (1 - self.lambda_coherence) * np.multiply(estimate, np.conjugate(observation))
        # self.coherence_12 = self.lambda_coherence * self.coherence_12 + (1 - self.lambda_coherence) * (estimate * observation).sum()

        self.coherence_11 = self.lambda_coherence * self.coherence_11 + (1 - self.lambda_coherence) * np.square(np.absolute(estimate))
        # self.coherence_11 = self.lambda_coherence * self.coherence_11 + (1 - self.lambda_coherence) * (estimate**2).sum()

        # self.coherence_22 = self.lambda_coherence * self.coherence_22 + (1 - self.lambda_coherence) * (observation**2).sum()
        self.coherence_22 = self.lambda_coherence * self.coherence_22 + (1 - self.lambda_coherence) * np.square(np.absolute(observation))

        coherence_b = np.divide(np.square(np.absolute(self.coherence_12)), np.multiply(self.coherence_11, self.coherence_22) + 1e-10)
        # coherence_b = self.coherence_12 / (self.coherence_22 + 1e-10)
        # coherence_b = np.sqrt(self.coherence_11 / (self.coherence_22 + 1e-10))

        weights = (1.0 / np.sum(self.coherence_11)) * self.coherence_11
        # weights = (1.0 / np.sum(self.coherence_22)) * self.coherence_22

        # rho = np.sum(np.multiply(weights, coherence_b))
        rho = np.sum(np.multiply(weights, coherence_b)[[19,77,115]])
        # rho = 1 - coherence_b
        # rho = coherence_b.mean()
        return rho


class CrossCorrelationCalculator:
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.rxy = np.zeros(block_length)
        self.var_y = 0.
        self.var_x = np.zeros(block_length)

    def calculate_rho(self, estimate, observation):

        # self.var_y = self.lambda_coherence * self.var_y + (1-self.lambda_coherence) * observation**2
        self.var_y = self.lambda_coherence * self.var_y + observation**2

        # self.rxy = self.lambda_coherence * self.rxy + (1-self.lambda_coherence) * (estimate * observation)
        self.rxy = self.lambda_coherence * self.rxy + (estimate * observation)
        # self.var_x = self.lambda_coherence * self.var_x + (1-self.lambda_coherence) * estimate**2
        self.var_x = self.lambda_coherence * self.var_x + estimate**2

        rho = abs(self.rxy) / np.sqrt((self.var_y * self.var_x) + 1e-15)

        rho = rho.max()

        # print(rho)
        return rho


class NormalizedCrossCorrelationCalculator:
    def __init__(self, block_length, lambda_coherence, lambda_rls):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.lambda_rls = lambda_rls
        self.h = np.zeros(block_length)
        self.rxy = np.zeros(block_length)
        self.Rxx_inv = np.identity(block_length)
        self.var_y = 0.

    def calculate_rho(self, estimate, observation):

        if (self.lambda_coherence * self.rxy) @ self.h >= (estimate * observation) @ self.h:
        # if (self.lambda_coherence * self.rxy + (1-self.lambda_coherence) * (estimate * observation)) @ self.h >= 0:

            # self.var_y = self.lambda_coherence * self.var_y + observation[-1]**2
            # self.var_y = self.lambda_coherence * self.var_y + (1-self.lambda_coherence) * observation[-1]**2
            # self.var_y = self.lambda_coherence * self.var_y + (1-self.lambda_coherence) * observation**2
            self.var_y = (self.lambda_coherence * self.var_y) + observation**2

            # self.var_x = self.lambda_coherence * self.var_x + (1-self.lambda_coherence) * estimate**2
            # self.var_x = self.lambda_coherence * self.var_x + (1-self.lambda_coherence) * (estimate**2).sum()

            # self.ryhaty = self.lambda_coherence * self.ryhaty + (1-self.lambda_coherence) * (estimate * observation)
            # self.ryhaty = self.lambda_coherence * self.ryhaty + (estimate * observation)

            # self.rxy = self.lambda_coherence * self.rxy + (estimate * observation[-1])
            # self.rxy = self.lambda_coherence * self.rxy + (1-self.lambda_coherence) * (estimate * observation[-1])
            # self.rxy = self.lambda_coherence * self.rxy + (1-self.lambda_coherence) * (estimate * observation)
            self.rxy = (self.lambda_coherence * self.rxy) + (estimate * observation)

        try:
            rho = np.sqrt((self.rxy @ self.h) / (self.var_y + 1e-10))
            # np.sqrt(-1)
        except FloatingPointError:
            a = 0
        # print(rho)

        if abs(estimate).sum() >= 0.01:

            kalman_gain = ((self.Rxx_inv @ estimate)) / (self.lambda_rls + (estimate @ self.Rxx_inv @ estimate))
            # kalman_gain = ((1-self.lambda_coherence) * (self.Rxx_inv @ estimate)) / (self.lambda_coherence + (1-self.lambda_coherence) * (estimate @ self.Rxx_inv @ estimate))

            self.h = self.h + kalman_gain * (observation - estimate @ self.h)
            # self.h = self.h + kalman_gain * (observation[-1] - estimate @ self.h)
            # self.h = self.h + ((self.mu * (observation - estimate @ self.h) * estimate) / (self.var_x + 1e-10))

            # # # self.eta = self.lambda_coherence * self.eta + observation - ((estimate @ self.h) * (self.lambda_coherence / (self.lambda_coherence + (estimate @ self.Rxx_inv @ estimate))))

            self.Rxx_inv = (self.Rxx_inv - np.outer(kalman_gain, (estimate @ self.Rxx_inv))) / self.lambda_rls
            # if estimate.sum() != 0.:
            #     self.Rxx_inv = (self.Rxx_inv - np.outer(kalman_gain, (estimate @ self.Rxx_inv))) / self.lambda_coherence
            # self.Rxx = self.lambda_coherence * self.Rxx + (1-self.lambda_coherence) * np.outer(estimate, estimate)

            # self.h = np.linalg.solve(self.Rxx + (1e-10 * np.identity(self.block_length)), self.rxy)

            # self.Rxx_inv /= np.trace(self.Rxx_inv)

            # self.Rxx_inv /= np.trace(self.Rxx_inv)

        # self.var_y = self.lambda_coherence * self.var_y + observation**2
        # self.var_x = self.lambda_coherence * self.var_x + (1-self.lambda_coherence) * estimate**2
        # self.var_y = self.lambda_coherence * self.var_y + (1-self.lambda_coherence) * observation**2


        # rho = np.sqrt(abs(self.eta) / (self.var_y + 1e-5))
        # rho = self.rxy / (np.sqrt(self.var_y * self.var_x) + 1e-10)
        # rho = rho.max()
        # rho = abs(rho).mean()
        # rho = np.sqrt(abs(self.ryhaty) / (self.var_y + 1e-10))

        # print(self.Rxx_inv)

        # print(rho)

        # if rho > 1.:
        #     rho = 1.

        return rho

class RobustNormalizedCrossCorrelationCalculator:
    def __init__(self, block_length, lambda_coherence, lambda_rls):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.lambda_rls = lambda_rls
        self.h = np.zeros(block_length)
        self.rxy = np.zeros(block_length)
        self.Rxx_inv = np.identity(block_length)
        self.var_y = 0.
        self.var_e = 0.
        self.eta = 0.

    def calculate_rho(self, estimate, observation):

        self.alpha = self.lambda_rls + (estimate @ self.Rxx_inv @ estimate)

        self.phi = self.lambda_rls / self.alpha



        e = observation - estimate @ self.h

        self.var_e = self.lambda_coherence * self.var_e + (1-self.lambda_coherence) * e**2

        observation_sq = observation**2
        e_sq = e**2


        if  observation_sq - self.phi * e_sq >= 0.:

            self.var_y = self.lambda_coherence * self.var_y + (1-self.lambda_coherence) * observation_sq
            self.eta = self.lambda_coherence * self.eta + (1-self.lambda_coherence) * (observation_sq - self.phi * e_sq)

        rho = np.sqrt(self.eta / (self.var_y + 1e-10))

        # print(rho)


        if abs(estimate).sum() >= 0.01:

            kalman_gain = (self.Rxx_inv @ estimate)

            self.h = self.h + (kalman_gain * e  / self.alpha)

            self.Rxx_inv = (self.Rxx_inv - (np.outer(kalman_gain, kalman_gain) / self.alpha)) / self.lambda_rls

        return rho


class TimeDomainCoherenceCalculator:
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.h = np.zeros(block_length)
        self.rxy = np.zeros(block_length)
        self.Rxx_inv = np.identity(block_length) * 1e-5
        # self.Rxx = np.identity(block_length)
        # self.Rxx_inv = np.identity(block_length)
        self.var_y = 0.
        self.var_x = 0.
        self.eta = 0.
        self.ryhaty=0.
        self.mu = 0.1

    def calculate_rho(self, estimate, observation):
        # if abs(np.trace(self.Rxx_inv)) >= 1e80 or abs(np.trace(self.Rxx_inv)) == 0.:
        #     self.Rxx_inv = np.identity(self.block_length)
        #     self.var_y = 0.
        #     self.rxy = np.zeros(self.block_length)
        #     self.h = np.zeros(self.block_length)

        # self.var_y = self.lambda_coherence * self.var_y + observation**2
        self.var_y = self.lambda_coherence * self.var_y + (1-self.lambda_coherence) * observation**2

        # self.var_x = self.lambda_coherence * self.var_x + (1-self.lambda_coherence) * estimate**2
        self.var_x = self.lambda_coherence * self.var_x + (1-self.lambda_coherence) * (estimate**2).sum()

        # self.ryhaty = self.lambda_coherence * self.ryhaty + (1-self.lambda_coherence) * (estimate * observation)
        # self.ryhaty = self.lambda_coherence * self.ryhaty + (estimate * observation)

        # # self.rxy = self.lambda_coherence * self.rxy + (estimate * observation)
        self.rxy = self.lambda_coherence * self.rxy + (1-self.lambda_coherence) * (estimate * observation)

        # # kalman_gain = ((self.Rxx_inv @ estimate)) / (self.lambda_coherence + (estimate @ self.Rxx_inv @ estimate))
        # kalman_gain = ((1-self.lambda_coherence) * (self.Rxx_inv @ estimate)) / (self.lambda_coherence + (1-self.lambda_coherence) * (estimate @ self.Rxx_inv @ estimate))

        # self.h = self.h + kalman_gain * (observation - estimate @ self.h)
        self.h = self.h + ((self.mu * (observation - estimate @ self.h) * estimate) / (self.var_x + 1e-10))
        # # # self.eta = self.lambda_coherence * self.eta + observation - ((estimate @ self.h) * (self.lambda_coherence / (self.lambda_coherence + (estimate @ self.Rxx_inv @ estimate))))

        # if estimate.sum() != 0.:
        #     self.Rxx_inv = (self.Rxx_inv - np.outer(kalman_gain, (estimate @ self.Rxx_inv))) / self.lambda_coherence
        # # self.Rxx = self.lambda_coherence * self.Rxx + (1-self.lambda_coherence) * np.outer(estimate, estimate)

        # self.h = np.linalg.solve(self.Rxx + (1e-10 * np.identity(self.block_length)), self.rxy)

        # self.Rxx_inv /= np.trace(self.Rxx_inv)

        # self.Rxx_inv /= np.trace(self.Rxx_inv)
        # self.var_y = self.lambda_coherence * self.var_y + observation**2
        # self.var_x = self.lambda_coherence * self.var_x + (1-self.lambda_coherence) * estimate**2
        # self.var_y = self.lambda_coherence * self.var_y + (1-self.lambda_coherence) * observation**2

        rho = np.sqrt(abs(self.rxy @ self.h) / (self.var_y + 1e-10))

        # rho = np.sqrt(abs(self.eta) / (self.var_y + 1e-5))
        # rho = self.rxy / (np.sqrt(self.var_y * self.var_x) + 1e-10)
        # rho = rho.max()
        # rho = abs(rho).mean()
        # rho = np.sqrt(abs(self.ryhaty) / (self.var_y + 1e-10))

        # print(self.Rxx_inv)
        print(rho)
        if rho > 1.:
            rho = 1.

        # if rho is np.nan:
        #     a = 0

        # kalman_gain = ((self.Rxx_inv @ estimate)) / (self.lambda_coherence + (estimate @ self.Rxx_inv @ estimate))
        # self.Rxx_inv = (self.Rxx_inv - (np.outer(kalman_gain, estimate) @ self.Rxx_inv)) / self.lambda_coherence
        # self.h = self.h + kalman_gain * (observation - estimate @ self.h)

        return rho


class MDFCoherenceCalculator:
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.h = np.zeros(block_length)
        self.rxy = np.zeros(block_length)
        self.Rxx = np.identity(block_length)
        self.Rxx_inv = np.identity(block_length)
        self.var_y = 0.
        self.var_x = 0.
        self.eta = 0.
        self.ryhaty=0.

    def calculate_rho(self, estimate, observation):

        X_k = fft(estimate)

        Y_k = self.H_k * X_k

        y_t = ifft(Y_k)[self.block_length:]

        e_t = observation - y_t

        padding = [(0, 0) for _ in range(estimate.ndim)]
        padding[-1] = (self.block_length, 0)
        e_t = np.pad(e_t, padding, mode="constant")

        E_k = fft(e_t)

        G_k = E_k * X_k.conj()
        g_t = ifft(G_k)

        # Constraint on gradient
        g_t[self.block_length:] = 0.
        G_k = fft(g_t)

        # Update statistics
        self.var_X_k = self.lambda_coherence * self.var_X_k + (1-self.lambda_coherence) * abs(X_k * X_k.conjugate())

        # Divide by var_X_k
        self.H_k += 2 * self.mu * G_k / (self.var_X_k + 1e-10)

        #TODO: return E_k

        rho = np.sqrt(self.ryhaty / (self.var_y + 1e-10))

        # print(self.Rxx_inv)
        print(rho)

        return rho
