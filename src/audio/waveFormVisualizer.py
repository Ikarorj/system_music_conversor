import matplotlib.pyplot as plt
import numpy as np


class WaveFormVisualizer:

    def plotWaveform(self, audioSignal: np.ndarray, sampleRate: int):

        time = np.arange(len(audioSignal)) / sampleRate

        plt.figure(figsize=(15, 4))

        plt.plot(time, audioSignal)

        plt.title("Forma de Onda")

        plt.xlabel("Tempo (s)")

        plt.ylabel("Amplitude")

        plt.grid(True)

        plt.tight_layout()

        plt.show()