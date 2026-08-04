import librosa


class ChromaExtractor:

    def extractChroma(self, audioSignal, sampleRate, hopLength=512):
        """
        Extrai a representação cromática (chroma) do áudio.
        Utiliza chroma_cqt, que possui melhor resolução de frequência
        e cobre múltiplas oitavas em comparação com o chroma_stft.

        Args:
            audioSignal (np.ndarray): Sinal de áudio.
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).

        Returns:
            np.ndarray: Matriz chroma (12, n_frames).
        """

        chroma = librosa.feature.chroma_cqt(
            y=audioSignal,
            sr=sampleRate,
            hop_length=hopLength
        )

        return chroma
