import sys

sys.path.insert(0, "src")

import numpy as np

from features.chromaExtractor import ChromaExtractor
from harmony.chordDetector import ChordDetector
from evaluate_chords import (
    GUITAR_CHORDS,
    CHORD_SECONDS,
    parseChord,
    synthesizeChord
)

SR = 44100


def evaluate(method, nSongs=15, seed=42):
    rng = np.random.default_rng(seed)
    n = 0
    exact = 0
    root = 0
    quality = 0

    for _ in range(nSongs):
        progression = rng.choice(GUITAR_CHORDS, size=4, replace=True).tolist()
        audio = np.concatenate([
            synthesizeChord(*parseChord(label)) for label in progression
        ])

        chroma = ChromaExtractor().extractChroma(
            audio, SR, harmonic=False, method=method
        )
        summary = ChordDetector().detectChordSummary(
            chroma, SR, windowSeconds=CHORD_SECONDS, smoothWindows=1
        )
        predicted = [entry["chord"] for entry in summary]

        for i in range(4):
            n += 1
            if predicted[i] == progression[i]:
                exact += 1
            if parseChord(predicted[i])[0] == parseChord(progression[i])[0]:
                root += 1
            if parseChord(predicted[i])[1] == parseChord(progression[i])[1]:
                quality += 1

    print(f"--- method={method}: {n} acordes")
    print(f"   exato : {exact}/{n} ({exact / n * 100:.1f}%)")
    print(f"   tônica: {root}/{n} ({root / n * 100:.1f}%)")
    print(f"   qualid: {quality}/{n} ({quality / n * 100:.1f}%)")


for method in ("cqt", "nnls"):
    evaluate(method)
