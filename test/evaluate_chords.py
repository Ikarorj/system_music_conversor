import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from audio.audioLoader import AudioLoader
from features.chromaExtractor import ChromaExtractor
from harmony.chordDetector import (
    CHORD_QUALITIES,
    ChordDetector,
    diatonicChordsForKey
)
from harmony.keyDetector import KeyDetector

SR = 44100
NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]
NOTE_HZ = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88
}

CHORD_SECONDS = 2.0


def parseChord(label):
    if label[1:2] == "#":
        rootName = label[:2]
        quality = label[2:]
    else:
        rootName = label[0]
        quality = label[1:]
    return rootName, quality


def synthesizeChord(rootName, quality, seconds=CHORD_SECONDS, sr=SR):
    semitones = CHORD_QUALITIES[quality]

    t = np.arange(int(seconds * sr)) / sr
    rootFreq = NOTE_HZ[rootName]
    signal = np.zeros_like(t)

    for index, semitone in enumerate(semitones):
        freq = rootFreq * 2 ** (semitone / 12)
        detune = 1 + 0.003 * np.sin(2 * np.pi * (3 + index) * t)
        envelope = np.exp(-2.0 * t) * (1 - np.exp(-t * 60))
        attack = np.exp(-2.5 * (index * 0.05))
        signal += attack * envelope * np.sin(2 * np.pi * freq * detune * t)
        signal += 0.5 * attack * envelope * np.sin(2 * np.pi * freq * 2 * t)
        signal += 0.25 * attack * envelope * np.sin(2 * np.pi * freq * 3 * t)

    signal += 0.01 * np.random.default_rng(0).standard_normal(len(t))

    return signal / np.max(np.abs(signal))


def detectProgressionFromAudio(
    audioSignal,
    sr=SR,
    method="nnls",
    diatonic=True
):
    chroma = ChromaExtractor().extractChroma(
        audioSignal, sr, method=method
    )
    detector = ChordDetector()

    labels = None
    if diatonic:
        key = KeyDetector().detectKey(chroma)
        labels = diatonicChordsForKey(key["root"], key["mode"])

    summary = detector.detectChordSummary(
        chroma,
        sr,
        windowSeconds=CHORD_SECONDS,
        smoothWindows=1,
        labels=labels
    )
    return [entry["chord"] for entry in summary]


def pitchClass(rootName):
    return NOTE_NAMES.index(rootName)


def printReport(name, reference, predicted):
    n = min(len(predicted), len(reference))

    exactHits = 0
    rootHits = 0
    qualityHits = 0
    rootDistances = []
    examples = []
    qualityExamples = []

    for i in range(n):
        pred = predicted[i]
        ref = reference[i]

        predRoot, predQuality = parseChord(pred)
        refRoot, refQuality = parseChord(ref)

        if pred == ref:
            exactHits += 1
        else:
            if len(examples) < 5:
                examples.append((ref, pred))

        if predRoot == refRoot:
            rootHits += 1
        else:
            distance = abs(pitchClass(predRoot) - pitchClass(refRoot))
            rootDistances.append(min(distance, 12 - distance))

        if predQuality == refQuality:
            qualityHits += 1
        elif predRoot == refRoot and len(qualityExamples) < 5:
            qualityExamples.append((ref, pred))

    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Acordes avaliados: {n}")
    print()
    print("Acertos:")
    print(f"  Acorde exato      : {exactHits:4d} / {n}  "
          f"({exactHits / n * 100:5.1f}%)")
    print(f"  Tônica correta    : {rootHits:4d} / {n}  "
          f"({rootHits / n * 100:5.1f}%)")
    print(f"  Qualidade correta : {qualityHits:4d} / {n}  "
          f"({qualityHits / n * 100:5.1f}%)")
    if rootDistances:
        print(f"  Distância média da tônica (semitons): "
              f"{np.mean(rootDistances):.2f}")
    if examples:
        print()
        print("Exemplos de erro (real -> previsto):")
        for ref, pred in examples:
            print(f"  {ref} -> {pred}")
    if qualityExamples:
        print()
        print("Exemplos de erro de qualidade (mesma tônica):")
        for ref, pred in qualityExamples:
            print(f"  {ref} -> {pred}")
    print()


def runSyntheticEvaluation(
    nSongs=30,
    chordsPerSong=4,
    seed=42,
    method="nnls",
    diatonic=True
):
    rng = np.random.default_rng(seed)

    allReference = []
    allPredicted = []

    for _ in range(nSongs):
        root = NOTE_NAMES[rng.integers(12)]
        mode = rng.choice(["major", "minor"])
        diatonicChords = diatonicChordsForKey(root, mode)

        progression = rng.choice(
            diatonicChords,
            size=chordsPerSong,
            replace=True
        ).tolist()

        audio = np.concatenate([
            synthesizeChord(*parseChord(label)) for label in progression
        ])

        predicted = detectProgressionFromAudio(
            audio,
            method=method,
            diatonic=diatonic
        )

        allReference.extend(progression)
        allPredicted.extend(predicted)

    return allReference, allPredicted


def runRealEvaluation(audioPath, labelsPath, method="nnls"):
    labels = []
    with open(labelsPath, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            labels.append((float(parts[0]), parts[1]))

    predicted = detectProgressionFromAudio(
        AudioLoader().loadAudio(audioPath)[0],
        method=method,
        diatonic=True
    )

    reference = []
    matchedPredicted = []
    for startSeconds, chord in labels:
        windowIndex = int(startSeconds / CHORD_SECONDS)
        if 0 <= windowIndex < len(predicted):
            reference.append(chord)
            matchedPredicted.append(predicted[windowIndex])

    return reference, matchedPredicted


def main():
    parser = argparse.ArgumentParser(
        description="Avalia o reconhecimento de acordes "
                    "(sintético ou áudio real com rótulos)."
    )
    parser.add_argument("--audio", help="Arquivo de áudio real")
    parser.add_argument(
        "--labels",
        help="Arquivo TSV: <segundo> <acorde> por linha"
    )
    parser.add_argument(
        "--songs", type=int, default=30,
        help="Número de progressões sintéticas (padrão: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--method", choices=["cqt", "nnls"], default="nnls"
    )
    parser.add_argument(
        "--no-diatonic", action="store_true",
        help="Não restringe a detecção ao campo harmônico do tom"
    )
    args = parser.parse_args()

    if args.audio and args.labels:
        reference, predicted = runRealEvaluation(
            args.audio, args.labels, method=args.method
        )
        printReport(f"Áudio real: {args.audio}", reference, predicted)
    else:
        reference, predicted = runSyntheticEvaluation(
            nSongs=args.songs,
            seed=args.seed,
            method=args.method,
            diatonic=not args.no_diatonic
        )
        title = "Avaliação sintética (progressões no campo harmônico"
        if not args.no_diatonic:
            title += " do tom"
        title += ")"
        printReport(title, reference, predicted)


if __name__ == "__main__":
    main()
