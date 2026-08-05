import librosa
import numpy as np


class TempoExtractor:

    def extractTempo(
        self,
        audioSignal,
        sampleRate
    ):
        """
        Estima o andamento e as batidas do áudio com librosa.

        Args:
            audioSignal (np.ndarray): Sinal de áudio.
            sampleRate (int): Taxa de amostragem em Hz.

        Returns:
            dict: Com tempo (BPM), beatFrames (índices de frames
                  das batidas) e beatTimes (tempo em segundos de
                  cada batida).
        """

        tempo, beatFrames = librosa.beat.beat_track(
            y=audioSignal,
            sr=sampleRate,
            units="frames"
        )

        tempo = float(np.asarray(tempo).reshape(-1)[0])

        beatTimes = librosa.frames_to_time(
            beatFrames,
            sr=sampleRate
        )

        return {
            "tempo": tempo,
            "beatFrames": np.asarray(beatFrames, dtype=int),
            "beatTimes": np.asarray(beatTimes, dtype=float)
        }

    def segmentByBeats(
        self,
        beatTimes,
        beatsPerWindow=2
    ):
        """
        Agrupa as batidas em janelas de um número fixo de batidas.

        Args:
            beatTimes (np.ndarray): Tempo (s) de cada batida.
            beatsPerWindow (int): Quantas batidas por janela
                (2 = compasso binário comum em pop).

        Returns:
            list: Tuplas (start, end) em segundos de cada janela.
        """

        beatTimes = np.asarray(beatTimes, dtype=float)

        windows = []

        for i in range(
            0,
            max(0, len(beatTimes) - beatsPerWindow),
            beatsPerWindow
        ):
            windows.append((
                float(beatTimes[i]),
                float(beatTimes[i + beatsPerWindow])
            ))

        return windows
