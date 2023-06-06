import numpy as np
from coherence_calculator import (
    CoherenceCalculator,
    CrossCorrelationCalculator,
    MDFCoherenceCalculator,
    NormalizedCrossCorrelationCalculator,
    RobustNormalizedCrossCorrelationCalculator,
)
from numpy.fft import fft


class CoherenceDoubleTalkDetector:
    """
        A double-talk detector based on coherence between loudspeaker and microphone signal (open loop),
        and estimated loudspeaker signal at microphone and microphone signal (closed loop).
    """
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.open_loop_coherence = CoherenceCalculator(block_length, lambda_coherence)
        self.closed_loop_coherence = CoherenceCalculator(block_length, lambda_coherence)

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block, microphone_samples_estimate):
        """
        Returns
        -------
        open_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present. Typically more noisy
            than its closed-loop equivalent.
        closed_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present.
        """
        D_b = fft(microphone_samples_block, axis=0)
        X_b = fft(loudspeaker_samples_block, axis=0)
        Y_hat = fft(microphone_samples_estimate, axis=0)

        open_loop_rho = self.open_loop_coherence.calculate_rho(X_b, D_b)
        closed_loop_rho = self.closed_loop_coherence.calculate_rho(Y_hat, D_b)
        # iqbal_rho = self.open_loop_coherence.calculate_rho(loudspeaker_samples_block, microphone_samples_block)


        return open_loop_rho, closed_loop_rho
        # return iqbal_rho

class CrossCorrelationDoubleTalkDetector:
    """
        A double-talk detector based on coherence between loudspeaker and microphone signal (open loop),
        and estimated loudspeaker signal at microphone and microphone signal (closed loop).
    """
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.cross_correlation = CrossCorrelationCalculator(block_length, lambda_coherence)

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block):
        """
        Returns
        -------
        benesty_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present. Typically more noisy
            than its closed-loop equivalent.
        """
        cross_correlation_loop_rho = self.cross_correlation.calculate_rho(loudspeaker_samples_block, microphone_samples_block)

        return cross_correlation_loop_rho


class RobustBenestyDoubleTalkDetector:
    """
        A double-talk detector based on coherence between loudspeaker and microphone signal (open loop),
        and estimated loudspeaker signal at microphone and microphone signal (closed loop).
    """
    def __init__(self, block_length, lambda_coherence, lambda_rls):
        self.block_length = block_length
        self.benesty_coherence = RobustNormalizedCrossCorrelationCalculator(block_length, lambda_coherence, lambda_rls)

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block):
        """
        Returns
        -------
        benesty_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present. Typically more noisy
            than its closed-loop equivalent.
        """
        benesty_loop_rho = self.benesty_coherence.calculate_rho(loudspeaker_samples_block, microphone_samples_block)

        return benesty_loop_rho



class BenestyDoubleTalkDetector:
    """
        A double-talk detector based on coherence between loudspeaker and microphone signal (open loop),
        and estimated loudspeaker signal at microphone and microphone signal (closed loop).
    """
    def __init__(self, block_length, lambda_coherence, lambda_rls):
        self.block_length = block_length
        self.benesty_coherence = NormalizedCrossCorrelationCalculator(block_length, lambda_coherence, lambda_rls)

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block):
        """
        Returns
        -------
        benesty_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present. Typically more noisy
            than its closed-loop equivalent.
        """
        benesty_loop_rho = self.benesty_coherence.calculate_rho(loudspeaker_samples_block, microphone_samples_block)

        return benesty_loop_rho


class MDFDoubleTalkDetector:
    """
        A double-talk detector based on coherence between loudspeaker and microphone signal (open loop),
        and estimated loudspeaker signal at microphone and microphone signal (closed loop).
    """
    def __init__(self, block_length, filter_length, lambda_coherence):
        self.block_length = block_length
        self.mdf_coherence = MDFCoherenceCalculator(block_length, filter_length, lambda_coherence)

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block):
        """
        Returns
        -------
        benesty_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present. Typically more noisy
            than its closed-loop equivalent.
        """
        mdf_loop_rho = self.mdf_coherence.calculate_rho(loudspeaker_samples_block, microphone_samples_block)

        return mdf_loop_rho

