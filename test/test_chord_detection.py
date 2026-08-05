import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio.audioLoader import AudioLoader
from features.chromaExtractor import ChromaExtractor
from harmony.chordDetector import (
    ChordDetector,
    diatonicChordsForKey
)
from harmony.keyDetector import KeyDetector
from output.shordSheetGenerator import ChordSheetGenerator


def detectChart(audioSignal, sampleRate):
    chroma = ChromaExtractor().extractChroma(audioSignal, sampleRate)
    key = KeyDetector().detectKey(chroma)
    labels = diatonicChordsForKey(key["root"], key["mode"])
    summary = ChordDetector().detectChordSummary(
        chroma,
        sampleRate,
        windowSeconds=2.0,
        labels=labels
    )
    return key, summary


def testSynthetic():
    import numpy as np

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

    key, summary = detectChart(audio, SR)
    detected = [entry["chord"] for entry in summary]

    print("Teste sintético (C, Am, F, G em Dó Maior):")
    print(f"  tom: {key['key']}")
    print(f"  detectado: {detected}")
    ok = detected == progression
    print(f"  {'PASSOU' if ok else 'FALHOU'}")
    return ok


def testReal(audioPath, durationSeconds, outputPath=None):
    signal, sr = AudioLoader().loadAudio(audioPath)
    if durationSeconds and durationSeconds < len(signal) / sr:
        signal = signal[: int(durationSeconds * sr)]

    t0 = time.time()
    key, summary = detectChart(signal, sr)
    sheet = ChordSheetGenerator().generate(
        summary,
        key,
        outputPath=outputPath
    )

    print(f"Áudio real: {audioPath} "
          f"({len(signal) / sr:.0f}s) em {time.time() - t0:.1f}s")
    print(sheet)


def main():
    ok = testSynthetic()
    print()
    testReal(
        sys.argv[1] if len(sys.argv) > 1 else "audios/samples/musicatest2.mpeg",
        float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    )
    print()
    print("RESULTADO SINTÉTICO:", "PASSOU" if ok else "FALHOU")


if __name__ == "__main__":
    main()
