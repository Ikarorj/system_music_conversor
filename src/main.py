from audio.audioLoader import AudioLoader
from audio.waveFormVisualizer import WaveFormVisualizer
from features.chromaExtractor import ChromaExtractor
from features.tempoExtractor import TempoExtractor
from harmony.chordDetector import (
    ChordDetector,
    diatonicChordsForKey
)
from harmony.chords.chordCandidateDetector import ChordCandidateDetector
from harmony.keyDetector import KeyDetector
from harmony.pitch.predominantPitchDetector import PredominantPitchDetector
from output.shordSheetGenerator import ChordSheetGenerator


def formatTime(seconds):
    minutes = int(seconds // 60)
    remainingSeconds = seconds - minutes * 60

    return f"{minutes:02d}:{remainingSeconds:05.2f}"


def main():

    loader = AudioLoader()

    audioSignal, sampleRate = loader.loadAudio(
        "audios/samples/musica.mpeg"
    )

    print("=" * 50)
    print("Áudio carregado com sucesso!")
    print("=" * 50)

    print(f"Sample Rate: {sampleRate} Hz")
    print(f"Número de amostras: {len(audioSignal)}")

    duration = len(audioSignal) / sampleRate

    print(f"Duração: {duration:.2f}s")

    waveformVisualizer = WaveFormVisualizer()

    waveformVisualizer.plotWaveform(
        audioSignal,
        sampleRate
    )

    pitchDetector = PredominantPitchDetector()
    chordDetector = ChordCandidateDetector()

    print()
    print("Detectando nota predominante em cada frame (pYIN)...")
    print("Isso pode levar alguns minutos. Aguarde...")

    predominantNotes = pitchDetector.detectPredominantNote(
        audioSignal,
        sampleRate
    )

    print("Detecção concluída!")
    print()

    segments = pitchDetector.detectNoteSegments(
        predominantNotes
    )

    print("=" * 50)
    print("Nota predominante por momento do áudio")
    print("=" * 50)

    for segment in segments:

        noteName = segment["note"]

        if segment["octave"] is not None:
            noteName = f"{noteName}{segment['octave']}"

        start = formatTime(segment["start"])
        end = formatTime(segment["end"])

        print(
            f"{start} - {end} "
            f"({segment['duration']:.2f}s) -> {noteName}"
        )

    print()
    print("=" * 50)
    print("Visão geral (chroma): notas mais fortes")
    print("=" * 50)

    chromaExtractor = ChromaExtractor()

    chroma = chromaExtractor.extractChroma(
        audioSignal,
        sampleRate
    )

    detectedNotes = chordDetector.detectChordCandidates(
        chroma,
        sampleRate
    )

    print()
    print("=" * 50)
    print("Tonalidade estimada (Krumhansl-Schmuckler)")
    print("=" * 50)

    keyDetector = KeyDetector()

    estimatedKey = keyDetector.detectKey(chroma)

    print(
        f"Tonalidade: {estimatedKey['key']} "
        f"(correlação {estimatedKey['score']:.3f})"
    )

    chordLabels = diatonicChordsForKey(
        estimatedKey["root"],
        estimatedKey["mode"]
    )

    print(
        "Campo harmônico: "
        + ", ".join(chordLabels)
    )

    tempoExtractor = TempoExtractor()

    tempoInfo = tempoExtractor.extractTempo(
        audioSignal,
        sampleRate
    )

    print(
        f"Tempo: {tempoInfo['tempo']:.1f} BPM "
        f"({len(tempoInfo['beatTimes'])} batidas detectadas)"
    )

    print()
    print("=" * 50)
    print("Cifra simplificada para violão")
    print("=" * 50)

    chordDetector = ChordDetector()

    chordSummary = chordDetector.detectChordSummaryWithBeats(
        chroma,
        sampleRate,
        beatTimes=tempoInfo["beatTimes"],
        beatsPerWindow=2,
        labels=chordLabels
    )

    sheetGenerator = ChordSheetGenerator()

    sheet = sheetGenerator.generate(
        chordSummary,
        estimatedKey
    )

    print(sheet)

    print()
    print("=" * 50)
    print("Detecção precisa por frame (primeiras 10)")
    print("=" * 50)

    for result in predominantNotes[:10]:

        noteName = result["note"]

        if noteName is None:

            print(
                f"Tempo: {result['time']:.2f}s -> "
                f"sem nota detectada"
            )

            continue

        if result["octave"] is not None:
            noteName = f"{noteName}{result['octave']}"

        print(
            f"Tempo: {result['time']:.2f}s -> "
            f"{noteName} "
            f"({result['frequency']:.2f} Hz, "
            f"{result['cents']:+d} cents, "
            f"confiança {result['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
