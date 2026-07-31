from audio.audioLoader import AudioLoader
from audio.waveFormVisualizer import WaveFormVisualizer
from features.chromaExtractor import ChromaExtractor
from harmony.noteDetector import NoteDetector


def main():

    loader = AudioLoader()

    audioSignal, sampleRate = loader.loadAudio(
        "audios/samples/musicatest2.mpeg"
    )

    print("=" * 50)
    print("Áudio carregado com sucesso!")
    print("=" * 50)

    print(f"Sample Rate: {sampleRate} Hz")
    print(f"Número de amostras: {len(audioSignal)}")

    waveformVisualizer = WaveFormVisualizer()

    waveformVisualizer.plotWaveform(
        audioSignal,
        sampleRate
    )

    chromaExtractor = ChromaExtractor()

    chroma = chromaExtractor.extractChroma(
        audioSignal,
        sampleRate
    )

    print()
    print("Chroma extraído com sucesso!")
    print(f"Dimensões: {chroma.shape}")

    noteDetector = NoteDetector()

    detectedNotes = noteDetector.detectNotes(
        chroma,
        sampleRate
    )

    print()
    print("Primeiras notas detectadas:")

    for result in detectedNotes[:10]:

        print(f"\nTempo: {result['time']:.2f}s")

        for note in result["notes"]:

            print(
                f"{note['note']} -> {note['intensity']:.3f}"
            )


if __name__ == "__main__":
    main()