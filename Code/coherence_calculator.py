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

class MDFCoherenceCalculator:
    def __init__(self, block_length, filter_length, lambda_coherence):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.N_fft = 2*block_length
        self.var_D = 0.
        self.eta = 0.
        self.K = filter_length // block_length
        self.x_old = np.zeros(block_length,dtype="float64") # time domain
        self.U = np.zeros((self.K, self.N_fft), dtype="complex128") # TF domain
        self.H = np.zeros((self.K, self.N_fft), dtype="complex128") # TF domain
        self.rxy = np.zeros((self.K, self.N_fft), dtype="complex128") # TF domain
        self.var_X = np.zeros((self.K, self.N_fft), dtype="complex128") # TF domain

        # self.U_diag = np.zeros((self.K * self.N_fft, self.N_fft), dtype="complex128") # TF domain
        # self.var_X = np.zeros((self.K * self.N_fft, self.N_fft), dtype="complex128") # TF domain

    def calculate_rho(self, estimate, observation):

        # X_k = fft(estimate)

        # Update multiframe input avaMic u_n = [x_n, ..., x_n-k+1]
        # for convolution operation h_n * u_n
        x_now = np.concatenate([self.x_old,estimate])
        X = fft(x_now)
        self.U[:-1] = self.U[1:]
        self.U[-1] = X
        self.x_old = estimate

        Yhat = (self.H * self.U).sum(axis=0)
        yhat_t = ifft(Yhat)
        yhat_t[:self.block_length] = 0.
        Yhat = fft(yhat_t)

        # yhat_t = ifft(Yhat)[self.block_length:]

        # External mic in TF domain
        d_fft = np.zeros(shape=(self.N_fft,),dtype="float64")
        d_fft[self.block_length:] = observation

        D = fft(d_fft)

        # e_t = observation - yhat_t

        # e_fft = np.zeros(shape=(self.N_fft,),dtype="float64")
        # e_fft[self.block_length:] = e_t
        # E = fft(e_fft)

        E = D - Yhat

        # Update statistics
        # self.var_X = self.lambda_coherence * self.var_X + (1-self.lambda_coherence) * (abs(self.U)**2).sum(axis=0)
        # self.var_X = self.lambda_coherence * self.var_X + (abs(self.U)**2).sum(axis=0)
        # self.var_X = self.lambda_coherence * self.var_X + (1-self.lambda_coherence) * (abs(self.U)**2)
        # self.var_X = self.lambda_coherence * self.var_X + (abs(self.U)**2)

        U_01 = ifft(self.U)
        U_01[:, :self.block_length] = 0
        U_01 = fft(U_01)
        self.var_X = self.lambda_coherence * self.var_X + (1-self.lambda_coherence) * (self.U.conj() * U_01)
        # self.var_X = self.lambda_coherence * self.var_X + (self.U.conj() * U_01)

        #TODO: compute as diagonal matrices


        G = (E * self.U.conj()) / (self.var_X + 1e-10)
        # G = np.linalg.solve(self.var_X + np.identity(self.var_X.shape[0] * 1e-10),(E * self.U.conj()).reshape)



        # self.H += (1-self.lambda_coherence) * G
        # self.H += G


        # # Apply IFFT on each block separately
        # h = ifft(self.H)

        # # Constraint on gradient
        # h[:,self.block_length:] = 0.

        # # Apply FFT on each block separately
        # self.H = fft(h)



        # Apply IFFT on each block separately
        g = ifft(G)

        # Constraint on gradient
        g[:,self.block_length:] = 0.

        # Apply FFT on each block separately
        G = fft(g)

        # Update EC filter
        # self.H += 2 * (1-self.lambda_coherence) * G
        self.H += (1-self.lambda_coherence) * G
        # self.H += G

        # Compute statistics
        self.var_D = self.lambda_coherence * self.var_D + (1-self.lambda_coherence) * (abs(D)**2).sum()
        # self.var_D = self.lambda_coherence * self.var_D + (abs(D)**2).sum()
        self.rxy = self.lambda_coherence * self.rxy + (1-self.lambda_coherence) * (self.U.conj() * D)
        # self.rxy = self.lambda_coherence * self.rxy + (self.U.conj() * D)

        # Do not update eta if not positive (see robust Benesty DTD)
        self.eta = (self.rxy[:,None] @ self.H[...,None].conj()).sum()

        rho = np.sqrt(abs(self.eta) / (self.var_D + 1e-10))

        # print(self.Rxx_inv)
        print(rho)

        return rho
