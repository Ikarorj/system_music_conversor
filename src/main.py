from audio.audioLoader import AudioLoader


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


if __name__ == "__main__":
    main()