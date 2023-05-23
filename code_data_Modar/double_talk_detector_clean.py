import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from coherence_double_talk_detector_clean import BenestyCoherenceDoubleTalkDetector
from librosa import util
from tqdm import tqdm
#from utils import generate_signals


def plot_results(signal_microphone, signal_noise, detector_output):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(signal_microphone)
    plt.legend(["microphone signal"], loc="upper left")
    plt.subplot(3, 1, 2)
    plt.plot(signal_noise)
    plt.legend(["signal_loudspeaker"], loc="upper left")
    plt.subplot(3, 1, 3)
    plt.plot(detector_output)
    plt.legend([r"$\xi $"], loc="upper left")
    plt.show()


def main():
    signal_microphone, fe = sf.read(
        "629e8071a8fce5001f505436_0_1919_1919-142785_external.wav"
    )
    signal_loudspeaker, fe = sf.read(
        "629e8071a8fce5001f505436_0_1919_1919-142785_internal.wav"
    )
    N = 1024

    dtd_benesty = BenestyCoherenceDoubleTalkDetector(block_length=N, lambda_coherence=0.98)

    detector_output = np.zeros((len(signal_loudspeaker),))
    detector_benchmark = np.zeros_like(detector_output)

    # pad 0 at beginning of signal_loudspeaker
    padding = [(0, 0) for _ in range(signal_loudspeaker.ndim)]
    padding[-1] = (N-1, 0)
    signal_loudspeaker_padded = np.pad(signal_loudspeaker, padding, mode="constant")

    # Window the time series.
    signal_loudspeaker_frames = util.frame(signal_loudspeaker_padded, frame_length=N, hop_length=1)

    # for i in tqdm(range(len(signal_microphone)-1)):
    for i in range(len(signal_microphone)-1):
        mic_block = signal_microphone[i]
        speaker_block = signal_loudspeaker_frames[:,i]
        detector_output[i: (i + 1)] = dtd_benesty.is_double_talk(speaker_block, mic_block)


    plot_results(signal_microphone, signal_loudspeaker, detector_output)


if __name__ == "__main__":
    main()
