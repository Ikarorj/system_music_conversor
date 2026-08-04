import librosa
import numpy as np

from harmony.noteUtils import NOTE_NAMES


class ChordCandidateDetector:

    def detectChordCandidates(
        self,
        chroma,
        sampleRate,
        hopLength=512,
        topNotes=3
    ):
        """
        Identifica, em cada frame, quais notas parecem estar
        presentes e formam um acorde, ordenadas por intensidade.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            topNotes (int): Quantidade de notas candidatas por frame.

        Returns:
            list: Lista de dicionários com time e a lista de notes
                  (cada uma com note e intensity).
        """

        detectedNotes = []

        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sampleRate,
            hop_length=hopLength
        )

        for frameIndex, column in enumerate(chroma.T):

            topIndexes = np.argsort(column)[-topNotes:][::-1]

            notes = []

            for index in topIndexes:

                notes.append({
                    "note": NOTE_NAMES[index],
                    "intensity": float(column[index])
                })

            detectedNotes.append({
                "time": float(times[frameIndex]),
                "notes": notes
            })

        return detectedNotes
