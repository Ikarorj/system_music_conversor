import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio.audioLoader import AudioLoader
from features.chromaExtractor import ChromaExtractor
from harmony.chordDetector import (
    ChordDetector,
    diatonicChordsForKey
)
from harmony.keyDetector import KeyDetector
from lyrics.lyricsTranscriber import LyricsTranscriber
from output.shordSheetGenerator import ChordSheetGenerator


def main():
    audioPath = sys.argv[1] if len(sys.argv) > 1 else (
        "audios/samples/musicatest2.mpeg"
    )
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    modelSize = sys.argv[3] if len(sys.argv) > 3 else "small"
    hotwords = sys.argv[4] if (
        len(sys.argv) > 4 and not sys.argv[4].startswith("--")
    ) else None
    isolateVocals = "--isolate" in sys.argv

    signal, sr = AudioLoader().loadAudio(audioPath)

    if duration and duration < len(signal) / sr:
        signal = signal[: int(duration * sr)]

    print(f"Áudio: {audioPath} | trecho: {len(signal) / sr:.0f}s | "
          f"modelo: {modelSize} | isolamento de voz: "
          f"{'sim' if isolateVocals else 'não'}")
    print()

    t0 = time.time()

    chroma = ChromaExtractor().extractChroma(signal, sr)
    key = KeyDetector().detectKey(chroma)
    labels = diatonicChordsForKey(key["root"], key["mode"])
    summary = ChordDetector().detectChordSummary(
        chroma,
        sr,
        windowSeconds=2.0,
        labels=labels
    )

    print(f"Acordes e tom em {time.time() - t0:.1f}s")
    print(f"Tom: {key['key']}")

    print()
    print("Transcrevendo letra (faster-whisper)...")
    if isolateVocals:
        print("Isolando a voz com Demucs (pode demorar)...")
    print("Primeiro uso baixa o modelo automaticamente. Aguarde...")

    t0 = time.time()

    transcriber = LyricsTranscriber(modelSize=modelSize)
    lyrics = transcriber.transcribe(
        signal,
        sr,
        language="pt",
        beamSize=8,
        hotwords=hotwords,
        isolateVocals=isolateVocals
    )

    print(f"Letra transcrita em {time.time() - t0:.1f}s")
    print()

    outputPath = Path(__file__).resolve().parents[1] / "audios" / "outputs" / "cifra_letra.txt"

    sheet = ChordSheetGenerator().generateWithLyrics(
        summary,
        key,
        lyrics,
        outputPath=str(outputPath)
    )

    print(sheet)
    print()
    print(f"Cifra salva em: {outputPath}")


if __name__ == "__main__":
    main()
