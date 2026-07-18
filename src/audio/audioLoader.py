import os

import librosa


class AudioLoader:

    def loadAudio(self, audioPath: str):
        """
        Carrega um arquivo de áudio utilizando o Librosa.

        Args:
            audioPath (str): Caminho do arquivo de áudio.

        Returns:
            tuple: (audioSignal, sampleRate)
        """

        if not os.path.exists(audioPath):
            raise FileNotFoundError(
                f'Arquivo não encontrado: {audioPath}'
            )

        audioSignal, sampleRate = librosa.load(
            audioPath,
            sr=None
        )

        return audioSignal, sampleRate