import librosa
import numpy as np


class ChromaExtractor:

    def extractChroma(
        self,
        audioSignal,
        sampleRate,
        hopLength=512,
        harmonic=True,
        method="cqt"
    ):
        """
        Extrai a representação cromática (chroma) do áudio.

        Quando harmonic=True, aplica HPSS para separar a parte
        harmônica (cordas/sinos) da percussiva (bateria/ruído),
        melhorando a detecção de acordes em músicas com bateria.

        method pode ser:
          - "cqt": chroma_cqt tradicional;
          - "nnls": ativação de notas esparsa via NNLS (melhor
            discriminação entre acordes, usada por Chordino).

        Args:
            audioSignal (np.ndarray): Sinal de áudio.
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            harmonic (bool): Se True, usa apenas a parte harmônica.
            method (str): "cqt" ou "nnls".

        Returns:
            np.ndarray: Matriz chroma (12, n_frames).
        """

        source = audioSignal

        if harmonic:
            harmonicPart, _ = librosa.effects.hpss(audioSignal)
            source = harmonicPart

        if method == "nnls":
            return self._extractNnlsChroma(
                source,
                sampleRate,
                hopLength
            )

        chroma = librosa.feature.chroma_cqt(
            y=source,
            sr=sampleRate,
            hop_length=hopLength
        )

        return chroma

    @staticmethod
    def _extractNnlsChroma(audioSignal, sampleRate, hopLength):
        """
        Chroma obtido por NNLS: constrói um dicionário espectral em
        que cada coluna é o espectro esperado de uma nota (fundamental
        + harmônicos), e resolve a decomposição não-negativa do CQT
        sobre esse dicionário. O resultado são ativações esparsas de
        notas, que são somadas em 12 classes de altura.

        Returns:
            np.ndarray: Matriz chroma (12, n_frames).
        """

        fmin = librosa.note_to_hz("C1")
        nBins = 84

        cqtMagnitude = np.abs(librosa.cqt(
            y=audioSignal,
            sr=sampleRate,
            fmin=fmin,
            n_bins=nBins,
            hop_length=hopLength,
            bins_per_octave=12
        ))

        basis = np.zeros((nBins, nBins))

        for note in range(nBins):
            for harmonicIndex in range(1, 16):
                harmonicBin = note + 12 * np.log2(harmonicIndex)
                harmonicBinRounded = int(round(harmonicBin))
                if harmonicBinRounded >= nBins:
                    break
                basis[harmonicBinRounded, note] = (
                    1.0 / harmonicIndex
                )

        activations = librosa.util.nnls(basis, cqtMagnitude)

        chroma = np.zeros((12, activations.shape[1]))

        for note in range(nBins):
            chroma[note % 12] += activations[note]

        return chroma
