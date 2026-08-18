import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicAnalyzer import MusicAnalyzer
from audio.audioPreprocessor import AudioPreprocessor
from audio.waveFormVisualizer import WaveFormVisualizer


def formatTime(seconds):
    minutes = int(seconds // 60)
    remainingSeconds = seconds - minutes * 60
    return f"{minutes:02d}:{remainingSeconds:05.2f}"


def setupLogging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Music POC - Reconhecimento de acordes "
                    "e geração de cifras para violão"
    )

    parser.add_argument(
        "audio",
        nargs="?",
        help="Caminho do arquivo de áudio"
    )
    parser.add_argument(
        "--list-chords", action="store_true",
        help="Lista os acordes detectados por janela"
    )
    parser.add_argument(
        "--list-segments", action="store_true",
        help="Lista segmentos de progressão (acorde repetido)"
    )
    parser.add_argument(
        "--lyrics", action="store_true",
        help="Transcreve a letra da música"
    )
    parser.add_argument(
        "--isolate", action="store_true",
        help="Isola a voz com Demucs antes de transcrever"
    )
    parser.add_argument(
        "--language", default=None,
        help="Idioma para transcrição (ex.: pt)"
    )
    parser.add_argument(
        "--export", nargs="+",
        choices=["txt", "json", "csv", "musicxml"],
        default=[],
        help="Formatos de exportação"
    )
    parser.add_argument(
        "--method", choices=["cqt", "nnls"], default="nnls",
        help="Método de extração de chroma (padrão: nnls)"
    )
    parser.add_argument(
        "--no-simplify", action="store_true",
        help="Não simplifica acordes complexos"
    )
    parser.add_argument(
        "--no-chunked", action="store_true",
        help="Desativa chunking no Viterbi"
    )
    parser.add_argument(
        "--chunk-beats", type=int, default=200,
        help="Batidas por chunk no Viterbi (padrão: 200)"
    )
    parser.add_argument(
        "--beats-per-window", type=int, default=4,
        help="Batidas por janela de acorde (padrão: 4 = um compasso)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Não exibe gráficos"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Diretório de saída (padrão: output)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Saída detalhada (DEBUG)"
    )

    args = parser.parse_args()

    setupLogging(args.verbose)

    if not args.audio:
        parser.print_help()
        sys.exit(1)

    audioPath = args.audio

    if not Path(audioPath).exists():
        print(f"Erro: arquivo não encontrado: {audioPath}")
        sys.exit(1)

    preprocessor = AudioPreprocessor()

    analyzer = MusicAnalyzer(
        preprocessor=preprocessor,
        chunkedDetection=not args.no_chunked,
        chunkBeats=args.chunk_beats,
        simplifyChords=not args.no_simplify,
        outputDir=args.output_dir
    )

    result = analyzer.analyze(
        audioPath,
        language=args.language,
        transcribeLyrics=args.lyrics,
        isolateVocals=args.isolate,
        exportFormats=args.export if args.export else None,
        chordMethod=args.method,
        beatsPerWindow=args.beats_per_window
    )

    print()
    print("=" * 50)
    print("Áudio carregado com sucesso!")
    print("=" * 50)

    info = result["audioInfo"]
    print(f"Sample Rate: {info['sampleRate']} Hz")
    print(f"Número de amostras: {info['samples']}")
    print(f"Duração: {info['duration']:.2f}s")

    if not args.no_plot:
        try:
            visualizer = WaveFormVisualizer()
            originalAudio, originalSr = result["_originalAudio"]
            visualizer.plotWaveform(originalAudio, originalSr)
        except Exception:
            pass

    key = result["key"]
    print()
    print("=" * 50)
    print("Tonalidade estimada (Krumhansl-Schmuckler)")
    print("=" * 50)
    print(
        f"Tonalidade: {key['key']} "
        f"(escore {key['score']:.3f})"
    )
    print(
        "Campo harmônico: "
        + ", ".join(result["diatonicChords"])
    )

    tempo = result["tempo"]
    print(
        f"Tempo: {tempo['tempo']:.1f} BPM "
        f"({len(tempo['beatTimes'])} batidas detectadas)"
    )

    if args.list_chords:
        print()
        print("=" * 50)
        print("Cifra simplificada para violão")
        print("=" * 50)
        print(result["sheet"])

    if args.list_segments and "progression" in result:
        print()
        print("=" * 50)
        print("Progressão de acordes")
        print("=" * 50)
        for seg in result["progression"]:
            start = formatTime(seg["start"])
            end = formatTime(seg["end"])
            print(
                f"{start} - {end} "
                f"({seg['duration']:.2f}s) -> {seg['chord']}"
            )

    if "simplifiedChords" in result:
        simplified = [
            e for e in result["simplifiedChords"]
            if e.get("simplified")
        ]
        if simplified:
            print()
            print("=" * 50)
            print("Acordes simplificados")
            print("=" * 50)
            for e in simplified:
                print(
                    f"  {e['originalChord']} -> {e['chord']}"
                )

    if args.lyrics and "lyrics" in result:
        print()
        print("=" * 50)
        print("Letra transcrita")
        print("=" * 50)
        for seg in result["lyrics"]:
            time = formatTime(seg["start"])
            print(f"{time}  {seg['text']}")

    print()
    print(
        f"Tempo de processamento: "
        f"{result['processingTime']:.2f}s"
    )

    if args.export:
        print()
        print(f"Arquivos exportados em: {args.output_dir}/")


if __name__ == "__main__":
    main()
