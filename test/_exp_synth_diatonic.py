import sys

sys.path.insert(0, "src")

import numpy as np

from features.chromaExtractor import ChromaExtractor
from harmony.chordDetector import ChordDetector, diatonicChordsForKey
from harmony.keyDetector import KeyDetector

SR = 44100
NOTE_HZ = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88
}
QUALITY = {"": [0, 4, 7], "m": [0, 3, 7]}
CHORD_SECONDS = 2.0

progression = ["C", "Am", "F", "G"]
chunks = []
for name in progression:
    quality = "m" if name.endswith("m") else ""
    rootName = name[:-1] if quality else name
    signal = np.zeros(int(SR * CHORD_SECONDS))
    t = np.arange(len(signal)) / SR
    for semitone in QUALITY[quality]:
        noteHz = NOTE_HZ[rootName] * 2 ** (semitone / 12)
        envelope = np.exp(-2.0 * t)
        signal += envelope * np.sin(2 * np.pi * noteHz * t)
        signal += 0.5 * envelope * np.sin(2 * np.pi * noteHz * 2 * t)
        signal += 0.25 * envelope * np.sin(2 * np.pi * noteHz * 3 * t)
    chunks.append(signal / np.max(np.abs(signal)))

audio = np.concatenate(chunks)

chroma = ChromaExtractor().extractChroma(audio, SR)
key = KeyDetector().detectKey(chroma)
labels = diatonicChordsForKey(key["root"], key["mode"])
print("tom:", key["key"], "| labels:", labels)

for smooth in (1, 3):
    summary = ChordDetector().detectChordSummary(
        chroma, SR, windowSeconds=2.0, smoothWindows=smooth, labels=labels
    )
    print(f"--- smoothWindows={smooth}:")
    for entry in summary:
        cand = ", ".join(
            f"{c['chord']} ({c['score']:.2f})"
            for c in entry["candidates"]
        )
        print(f"   {entry['time']:.1f}s {entry['chord']:5s} | {cand}")
