from audio.audioLoader import AudioLoader
from audio.waveFormVisualizer import WaveFormVisualizer

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

    visualizer = WaveFormVisualizer();

    visualizer.plotWaveform(audioSignal, sampleRate)


if __name__ == "__main__":
    main()