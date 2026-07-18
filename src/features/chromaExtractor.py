import librosa


class ChromaExtractor:

    def extractChroma(self, audioSignal, sampleRate):

        chroma = librosa.feature.chroma_stft(
            y=audioSignal,
            sr=sampleRate
        )

        return chroma