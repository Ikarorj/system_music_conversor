from audio.audioLoader import AudioLoader
from audio.waveFormVisualizer import WaveFormVisualizer
from features.chromaExtractor import ChromaExtractor


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

    visualizer = WaveFormVisualizer()

    visualizer.plotWaveform(audioSignal, sampleRate)

    visualizer = ChromaExtractor()

    chromaExtractor = ChromaExtractor()


    chroma = chromaExtractor.extractChroma(
    audioSignal,
    sampleRate
    )

    print()

    print("Chroma extraído com sucesso!")

    print(f"Dimensões: {chroma.shape}")


if __name__ == "__main__":
    main()
