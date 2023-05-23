import time

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from coherence_double_talk_detector import (
    BenestyDoubleTalkDetector,
    CoherenceDoubleTalkDetector,
    CrossCorrelationDoubleTalkDetector,
    MDFDoubleTalkDetector,
    RobustBenestyDoubleTalkDetector,
)
from emdf_double_talk_detector import EMDFDoubleTalkDetector
from librosa import util
from signal_processing.stft import stft
from tqdm import tqdm
from utils import generate_signals


def plot_results(signal_microphone, signal_noise, detector_output, signal_nearend):
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(signal_microphone)
    plt.legend(["microphone signal"], loc="upper left")
    plt.subplot(4, 1, 2)
    plt.plot(signal_noise)
    plt.legend(["noise (double-talk) signal"], loc="upper left")
    plt.subplot(4, 1, 3)
    plt.plot(signal_nearend)
    plt.legend(["noise (double-talk) signal"], loc="upper left")
    ax1 = plt.subplot(4, 1, 4)
    ax1.plot(detector_output)
    # plt.plot(detector_benchmark)
    # plt.legend([r"$\xi $", "double-talk active"], loc="upper left")
    ax1.legend([r"$\xi $"], loc="upper left")
    # ax1.set_ylim([0,2])
    plt.show()


def main():
    # signal_microphone, signal_loudspeaker, _, _, noise_signal = generate_signals(noise_start_in_seconds=1.0, length_in_seconds=3.0)
    signal_microphone, fe = sf.read(
        # "../../../../../../stages/echo-reduction/datasets/add-artificial-delay/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_external.wav"
        # "stages/echo-reduction/datasets/add-artificial-delay/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_external.wav"
        # "stages/echo-reduction/datasets/preprocess/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_external.wav"
        "stages/echo-reduction/datasets/preprocess/outs/AvaEchoCancellationDatasets/DoubleTalk/6290fa2176495d001c96e9b8/6290fa2176495d001c96e9b8_0_652_652-130737_external.wav"
        # "stages/echo-reduction/residual-echo-suppression/evaluate/fdaf-double-talk-detector/audio/Example_2/input-nearend_16k.wav"
    )
    signal_loudspeaker, fe = sf.read(
        # "../../outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_ECOutput.wav"
        # "stages/echo-reduction/echo-cancellation/evaluate/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_ECOutput.wav"
        # "stages/echo-reduction/residual-echo-suppression/evaluate/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_ECOutput.wav"
        # "../../../../../../stages/echo-reduction/datasets/add-artificial-delay/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_internal.wav"
        # "stages/echo-reduction/datasets/add-artificial-delay/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_internal.wav"
        # "stages/echo-reduction/datasets/preprocess/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_internal.wav"
        "stages/echo-reduction/datasets/preprocess/outs/AvaEchoCancellationDatasets/DoubleTalk/6290fa2176495d001c96e9b8/6290fa2176495d001c96e9b8_0_652_652-130737_internal.wav"
        # "stages/echo-reduction/residual-echo-suppression/evaluate/fdaf-double-talk-detector/audio/Example_2/input-farend_16k.wav"
    )
    noise_signal, fe = sf.read(
        # "../../../../../../stages/echo-reduction/datasets/add-artificial-delay/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_nearend.wav"
        # "stages/echo-reduction/datasets/add-artificial-delay/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_nearend.wav"
        # "stages/echo-reduction/datasets/preprocess/outs/AvaEchoCancellationDatasets/DoubleTalk/629e8071a8fce5001f505436/629e8071a8fce5001f505436_0_1919_1919-142785_nearend.wav"
        "stages/echo-reduction/datasets/preprocess/outs/AvaEchoCancellationDatasets/DoubleTalk/6290fa2176495d001c96e9b8/6290fa2176495d001c96e9b8_0_652_652-130737_nearend.wav"
        # "stages/echo-reduction/residual-echo-suppression/evaluate/fdaf-double-talk-detector/audio/Example_2/input-nearend-only.wav"
    )

    # N = 4096
    N = 512
    # O = int(N - 1)

    dtd = CoherenceDoubleTalkDetector(block_length=N, lambda_coherence=0.9)
    dtd_cc = CrossCorrelationDoubleTalkDetector(block_length=N, lambda_coherence=0.9999)
    dtd_benesty = BenestyDoubleTalkDetector(block_length=N, lambda_coherence=0.999, lambda_rls=0.9999)
    dtd_robust_benesty = RobustBenestyDoubleTalkDetector(block_length=N, lambda_coherence=0.9999, lambda_rls=0.99999)
    dtd_mdf = MDFDoubleTalkDetector(block_length=N, lambda_coherence=0.9)

    noise_power_threshold = 0.0010  # power of noise block to account as active (for benchmark purposes only)

    detector_output = np.zeros((len(signal_loudspeaker),))
    detector_benchmark = np.zeros_like(detector_output)

    time_accumulator = 0.0

    nb_iterations = len(signal_loudspeaker) // N
    # nb_iterations = len(signal_loudspeaker) // O

    signal_loudspeaker /= (signal_loudspeaker.max() / 4)

    # To realign external / internal audio
    signal_microphone = signal_microphone[int(0.08*fe):]
    signal_microphone = np.pad(signal_microphone, pad_width=(0,int(0.08*fe)), mode="constant")

    # pad 0 at beginning of signal_loudspeaker
    padding = [(0, 0) for _ in range(signal_loudspeaker.ndim)]
    padding[-1] = (N-1, 0)
    signal_loudspeaker_padded = np.pad(signal_loudspeaker, padding, mode="constant")

    # Window the time series.
    signal_loudspeaker_frames = util.frame(signal_loudspeaker_padded, frame_length=N, hop_length=1)

    for i in tqdm(range(len(signal_microphone)-1)):
    # for i in range(len(signal_loudspeaker)-1):
        mic_block = signal_microphone[i]
        speaker_block = signal_loudspeaker_frames[:,i]
        # speaker_block = signal_loudspeaker[i]

        #TODO: add more samples for X = (x1, x2, ..., xN) where x1 is vector of length N
        #TODO: run it again....
        #TODO: make it faster with MDF
        detector_output[i: (i + 1)] = dtd_robust_benesty.is_double_talk(speaker_block, mic_block)
        # detector_output[i: (i + 1)] = dtd_benesty.is_double_talk(speaker_block, mic_block)
        # detector_output[i: (i + 1)] = dtd_cc.is_double_talk(speaker_block, mic_block)

    # # Gansler
    # # Chunk size = FFT size
    # WINDOW_LEN_SEC = N / fe  # window length (in seconds)
    # WINDOW = "rectangular"  # window function
    # HOP_PERCENT = 0.5  # hop (in percentage)
    # CENTER = False

    # D = stft(
    #     x=signal_microphone,
    #     sample_rate=fe,
    #     window_len_sec=WINDOW_LEN_SEC,
    #     window=WINDOW,
    #     hop_percent=HOP_PERCENT,
    #     center=CENTER,
    # )

    # X = stft(
    #     x=signal_loudspeaker,
    #     sample_rate=fe,
    #     window_len_sec=WINDOW_LEN_SEC,
    #     window=WINDOW,
    #     hop_percent=HOP_PERCENT,
    #     center=CENTER,
    # )
    # nFrames = D.shape[0]

    # hop_samples = int(N * HOP_PERCENT)

    # lambda_coherence = 0.6
    # coherence_xd = np.zeros(D.shape[1])
    # coherence_xx = np.zeros(D.shape[1])
    # coherence_dd = np.zeros(D.shape[1])

    # # Iterate over the TF frames
    # for i in range(nFrames):
    #     coherence_xd = lambda_coherence * coherence_xd + (1 - lambda_coherence) * (X[i] * D[i].conjugate())
    #     coherence_xx = lambda_coherence * coherence_xx + (1 - lambda_coherence) * abs(X[i])**2
    #     coherence_dd = lambda_coherence * coherence_dd + (1 - lambda_coherence) * abs(D[i])**2

    #     # corr = abs(coherence_xd)**2 / ((coherence_xx * coherence_dd) + 1e-15)
    #     corr = abs(coherence_xd) / np.sqrt((coherence_xx * coherence_dd) + 1e-15)

    #     detector_output[i * hop_samples  : (i + 1) * hop_samples] = corr[128:512].mean()
    #     # detector_output[i * hop_samples  : (i + 1) * hop_samples] = corr.mean()




    # # MDF, Gansler
    # for i in range(0, nb_iterations):
    #     # print(f"Iteration {i+1} out of {nb_iterations}")

    #     start = time.time()

    #     # Gansler
    #     mic_block = signal_microphone[i * N : (i + 1) * N]
    #     speaker_block = signal_loudspeaker[i * N : (i + 1) * N]
    #     noise_block = noise_signal[i * N : (i + 1) * N]

    #     # Benesty
    #     # mic_block = signal_microphone[i * O : i * O + N]
    #     # speaker_block = signal_loudspeaker[i * O : i * O + N]
    #     # noise_block = noise_signal[i * O : i * O + N]


    #     # noise_block_power = np.linalg.norm(noise_block, 2) / len(noise_block)
    #     # if noise_block_power > noise_power_threshold:
    #     #     detector_benchmark[i * N : (i + 1) * N] = np.ones((N,))

    #     # #TODO: add more samples for X = (x1, x2, ..., xN) where x1 is vector of length N
    #     # detector_output[i * N : (i + 1) * N] = dtd.is_double_talk(
    #     #     speaker_block, mic_block, speaker_block
    #     # )[0] * np.ones(
    #     #     (N,)
    #     # )  # take only open loop result

    #     # Benesty's algorithm
    #     detector_output[i * N : (i + 1) * N] = dtd_benesty.is_double_talk(speaker_block, mic_block) * np.ones((N,))
    #     # detector_output[i * O : i * O + N] = dtd_benesty.is_double_talk(speaker_block, mic_block) * np.ones((N,))

    #     # # MDF
    #     # mic_block = signal_microphone[i * N : (i + 1) * N]
    #     # speaker_block = signal_loudspeaker[i * N : (i + 2) * N]
    #     # detector_output[i * N : (i + 1) * N] = dtd_mdf.is_double_talk(speaker_block, mic_block, speaker_block)

    #     end = time.time()
    #     time_accumulator += end - start
    #     # print(f"Average iteration time: {time_accumulator / (i+1)}")

    # plot_results(signal_microphone, noise_signal, detector_output, detector_benchmark)
    # plot_results(signal_microphone, signal_loudspeaker, detector_output, detector_benchmark)
    plot_results(signal_microphone, signal_loudspeaker, detector_output, noise_signal)


if __name__ == "__main__":
    main()
