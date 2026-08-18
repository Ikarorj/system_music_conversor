import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class SpectralExtractor:

    def __init__(self, hopLength=512):
        """
        Extrai features espectrais do áudio (centroid, bandwidth,
        rolloff, flatness, contrast).

        Args:
            hopLength (int): Salto entre frames (em amostras).
        """

        self.hopLength = hopLength

    def extract(self, audioSignal, sampleRate):
        """
        Extrai todas as features espectrais.

        Args:
            audioSignal (np.ndarray): Sinal de áudio.
            sampleRate (int): Taxa de amostragem em Hz.

        Returns:
            dict: Chaves são os nomes das features, valores são
                  np.ndarray 1D (um valor por frame).
        """

        features = {}

        features["centroid"] = self._centroid(audioSignal, sampleRate)
        features["bandwidth"] = self._bandwidth(
            audioSignal, sampleRate
        )
        features["rolloff"] = self._rolloff(audioSignal, sampleRate)
        features["flatness"] = self._flatness(audioSignal)
        features["contrast"] = self._contrast(
            audioSignal, sampleRate
        )
        features["rms"] = self._rms(audioSignal)
        features["zcr"] = self._zcr(audioSignal)

        logger.info(
            "Features espectrais extraídas: %d features, %d frames",
            len(features), features["centroid"].shape[0]
        )

        return features

    def _centroid(self, audioSignal, sampleRate):
        return librosa.feature.spectral_centroid(
            y=audioSignal, sr=sampleRate,
            hop_length=self.hopLength
        )[0]

    def _bandwidth(self, audioSignal, sampleRate):
        return librosa.feature.spectral_bandwidth(
            y=audioSignal, sr=sampleRate,
            hop_length=self.hopLength
        )[0]

    def _rolloff(self, audioSignal, sampleRate):
        return librosa.feature.spectral_rolloff(
            y=audioSignal, sr=sampleRate,
            hop_length=self.hopLength
        )[0]

    def _flatness(self, audioSignal):
        return librosa.feature.spectral_flatness(
            y=audioSignal,
            hop_length=self.hopLength
        )[0]

    def _contrast(self, audioSignal, sampleRate):
        return librosa.feature.spectral_contrast(
            y=audioSignal, sr=sampleRate,
            hop_length=self.hopLength
        )[0]

    def _rms(self, audioSignal):
        return librosa.feature.rms(
            y=audioSignal,
            hop_length=self.hopLength
        )[0]

    def _zcr(self, audioSignal):
        return librosa.feature.zero_crossing_rate(
            y=audioSignal,
            hop_length=self.hopLength
        )[0]
