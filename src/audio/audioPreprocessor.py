import logging

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


class AudioPreprocessor:

    def __init__(
        self,
        targetSampleRate=None,
        normalize=False,
        removeNoise=False,
        noiseCutLow=60.0,
        noiseCutHigh=16000.0,
        trimSilence=False,
        trimTopDb=40,
        padToSeconds=None
    ):
        """
        Pré-processamento de áudio antes da extração de features.

        Args:
            targetSampleRate (int): Taxa de amostragem alvo. Se None,
                mantém a original (recomendado para preservar质量).
            normalize (bool): Normaliza o pico para [-1, 1].
            removeNoise (bool): Aplica filtro passa-banda para
                reduzir ruído fora da faixa musical.
            noiseCutLow (float): Frequência de corte baixa (Hz).
            noiseCutHigh (float): Frequência de corte alta (Hz).
            trimSilence (bool): Remove silêncio do início/fim.
            trimTopDb (float): Limiar em dB para considerar silêncio.
            padToSeconds (float): Se definido, faz padding/trunc
                para atingir essa duração em segundos.
        """

        self.targetSampleRate = targetSampleRate
        self.normalize = normalize
        self.removeNoise = removeNoise
        self.noiseCutLow = noiseCutLow
        self.noiseCutHigh = noiseCutHigh
        self.trimSilence = trimSilence
        self.trimTopDb = trimTopDb
        self.padToSeconds = padToSeconds

    def process(self, audioSignal, sampleRate):
        """
        Aplica as etapas de pré-processamento em sequência.

        Args:
            audioSignal (np.ndarray): Sinal de áudio bruto.
            sampleRate (int): Taxa de amostragem original.

        Returns:
            tuple: (audioSignal, sampleRate) processados.
        """

        logger.info(
            "Pré-processamento: %d amostras, %d Hz, %.2fs",
            len(audioSignal), sampleRate,
            len(audioSignal) / sampleRate
        )

        if self.targetSampleRate and sampleRate != self.targetSampleRate:
            audioSignal = librosa.resample(
                audioSignal,
                orig_sr=sampleRate,
                target_sr=self.targetSampleRate
            )
            sampleRate = self.targetSampleRate
            logger.info("Resample para %d Hz", sampleRate)

        if self.trimSilence:
            audioSignal, _ = librosa.effects.trim(
                audioSignal, top_db=self.trimTopDb
            )
            logger.info(
                "Silêncio removido: %.2fs", len(audioSignal) / sampleRate
            )

        if self.removeNoise:
            audioSignal = self._bandpassFilter(
                audioSignal, sampleRate
            )
            logger.info("Filtro passa-banda aplicado")

        if self.normalize:
            peak = np.max(np.abs(audioSignal))
            if peak > 1e-8:
                audioSignal = audioSignal / peak
                logger.info("Normalizado (pico: %.4f -> 1.0)", peak)

        if self.padToSeconds is not None:
            targetSamples = int(self.padToSeconds * sampleRate)
            audioSignal = self._padOrTrim(
                audioSignal, targetSamples
            )
            logger.info(
                "Ajustado para %.1fs (%d amostras)",
                self.padToSeconds, len(audioSignal)
            )

        return audioSignal, sampleRate

    def _bandpassFilter(self, audioSignal, sampleRate):
        low = self.noiseCutLow / (sampleRate / 2)
        high = self.noiseCutHigh / (sampleRate / 2)
        low = max(low, 0.001)
        high = min(high, 0.999)
        sos = butter(4, [low, high], btype="band", output="sos")
        return sosfilt(sos, audioSignal).astype(np.float32)

    @staticmethod
    def _padOrTrim(audioSignal, targetSamples):
        if len(audioSignal) > targetSamples:
            return audioSignal[:targetSamples]
        if len(audioSignal) < targetSamples:
            padLen = targetSamples - len(audioSignal)
            return np.pad(audioSignal, (0, padLen), mode="constant")
        return audioSignal
