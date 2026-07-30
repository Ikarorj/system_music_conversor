import librosa
import numpy as np


class NoteDetector:

    NOTE_NAMES = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B"
    ]

    def detectNotes(
        self,
        chroma,
        sampleRate,
        hopLength=512,
        topNotes=3
    ):

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
                    "note": self.NOTE_NAMES[index],
                    "intensity": float(column[index])
                })

            detectedNotes.append({
                "time": float(times[frameIndex]),
                "notes": notes
            })

        return detectedNotes


