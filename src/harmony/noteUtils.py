import numpy as np

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


def frequencyToNote(frequency):
    """
    Converte uma frequência (Hz) para a nota musical mais próxima,
    sua oitava e o desvio em cents.

    Returns:
        tuple: (noteName, octave, cents) ou (None, None, None)
               quando a frequência é inválida.
    """

    if (
        frequency is None
        or np.isnan(frequency)
        or frequency <= 0
    ):
        return None, None, None

    midi = 12 * np.log2(frequency / 440.0) + 69

    midiRounded = int(round(midi))

    noteName = NOTE_NAMES[midiRounded % 12]

    octave = midiRounded // 12 - 1

    cents = int(round((midi - midiRounded) * 100))

    return noteName, octave, cents
