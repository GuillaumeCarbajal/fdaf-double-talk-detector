from coherence_calculator_clean import TimeDomainCoherenceCalculator


class BenestyCoherenceDoubleTalkDetector:
    """
        A double-talk detector based on coherence between loudspeaker and microphone signal (open loop),
        and estimated loudspeaker signal at microphone and microphone signal (closed loop).
    """
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.benesty_coherence = TimeDomainCoherenceCalculator(block_length, lambda_coherence)

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
