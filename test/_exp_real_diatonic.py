import sys

sys.path.insert(0, "src")

from audio.audioLoader import AudioLoader
from features.chromaExtractor import ChromaExtractor
from harmony.chordDetector import ChordDetector, diatonicChordsForKey
from harmony.keyDetector import KeyDetector

signal, sr = AudioLoader().loadAudio("audios/samples/musicatest2.mpeg")
signal = signal[: int(60 * sr)]

for method in ("cqt", "nnls"):
    chroma = ChromaExtractor().extractChroma(
        signal, sr, method=method
    )
    key = KeyDetector().detectKey(chroma)
    labels = diatonicChordsForKey(key["root"], key["mode"])
    summary = ChordDetector().detectChordSummary(
        chroma, sr, windowSeconds=2.0, smoothWindows=3, labels=labels
    )
    print(f"--- {method} | tom: {key['key']} | "
          f"campo harmônico: {labels}")
    for entry in summary:
        cand = ", ".join(
            f"{c['chord']} ({c['score']:.2f})"
            for c in entry["candidates"]
        )
        print(f"   {entry['time']:5.1f}s {entry['chord']:9s} | {cand}")
    print(flush=True)
