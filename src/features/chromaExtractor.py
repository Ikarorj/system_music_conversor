import librosa


class ChromaExtractor:

    def extractChroma(
        self,
        audioSignal,
        sampleRate,
        hopLength=512,
        harmonic=True
    ):
        """
        Extrai a representação cromática (chroma) do áudio.

        Quando harmonic=True, aplica HPSS para separar a parte
        harmônica (cordas/sinos) da percussiva (bateria/ruído),
        melhorando a detecção de acordes em músicas com bateria.

        Args:
            audioSignal (np.ndarray): Sinal de áudio.
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            harmonic (bool): Se True, usa apenas a parte harmônica.

        Returns:
            np.ndarray: Matriz chroma (12, n_frames).
        """

        source = audioSignal

        if harmonic:
            harmonicPart, _ = librosa.effects.hpss(audioSignal)
            source = harmonicPart

        chroma = librosa.feature.chroma_cqt(
            y=source,
            sr=sampleRate,
            hop_length=hopLength
        )

        return chroma
