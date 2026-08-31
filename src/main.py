import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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

    outputGroup = parser.add_argument_group("saída")
    outputGroup.add_argument(
        "--list-chords", action="store_true",
        help="Lista os acordes detectados por janela"
    )
    outputGroup.add_argument(
        "--list-segments", action="store_true",
        help="Lista segmentos de progressão"
    )
    outputGroup.add_argument(
        "--export", nargs="+",
        choices=["txt", "json", "csv", "musicxml"],
        default=[],
        help="Formatos de exportação"
    )
    outputGroup.add_argument(
        "--no-plot", action="store_true",
        help="Não exibe gráficos"
    )
    outputGroup.add_argument(
        "--output-dir", default="output",
        help="Diretório de saída"
    )

    audioGroup = parser.add_argument_group("processamento de áudio")
    audioGroup.add_argument(
        "--method", choices=["cqt", "nnls"], default="nnls",
        help="Método de chroma (padrão: nnls)"
    )
    audioGroup.add_argument(
        "--no-simplify", action="store_true",
        help="Não simplifica acordes"
    )
    audioGroup.add_argument(
        "--no-chunked", action="store_true",
        help="Desativa chunking no Viterbi"
    )
    audioGroup.add_argument(
        "--chunk-beats", type=int, default=200,
        help="Batidas por chunk no Viterbi"
    )
    audioGroup.add_argument(
        "--beats-per-window", type=int, default=4,
        help="Batidas por janela de acorde"
    )

    interpGroup = parser.add_argument_group("experimentos de interpretação")
    interpGroup.add_argument(
        "--experiment", choices=["a", "b", "c"], default="a",
        help=(
            "a=DSP puro (padrão), "
            "b=DSP+regras musicais, "
            "c=DSP+LLM (Groq)"
        )
    )
    interpGroup.add_argument(
        "--beats-per-chunk", type=int, default=8,
        help="Batidas por chunk para evidências (padrão: 8)"
    )
    interpGroup.add_argument(
        "--groq-api-key", default=None,
        help="Chave API Groq (ou defina GROQ_API_KEY)"
    )
    interpGroup.add_argument(
        "--groq-model", default="qwen/qwen3.6-27b",
        help="Modelo Groq (padrão: qwen/qwen3.6-27b)"
    )

    lyricsGroup = parser.add_argument_group("letra")
    lyricsGroup.add_argument(
        "--lyrics", action="store_true",
        help="Transcreve a letra"
    )
    lyricsGroup.add_argument(
        "--isolate", action="store_true",
        help="Isola voz com Demucs"
    )
    lyricsGroup.add_argument(
        "--language", default=None,
        help="Idioma para transcrição (ex.: pt)"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Saída detalhada"
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

    if args.experiment == "c" and not args.groq_api_key:
        import os
        if not os.environ.get("GROQ_API_KEY"):
            print(
                "Erro: experimento 'c' requer chave Groq.\n"
                "Use --groq-api-key ou defina GROQ_API_KEY."
            )
            sys.exit(1)

    experimentNames = {
        "a": "DSP puro",
        "b": "DSP + regras musicais",
        "c": "DSP + LLM (Groq)"
    }

    preprocessor = AudioPreprocessor()

    analyzer = MusicAnalyzer(
        preprocessor=preprocessor,
        chunkedDetection=not args.no_chunked,
        chunkBeats=args.chunk_beats,
        simplifyChords=not args.no_simplify,
        outputDir=args.output_dir
    )

    print(f"Experimento: {experimentNames[args.experiment]}")

    result = analyzer.analyze(
        audioPath,
        language=args.language,
        transcribeLyrics=args.lyrics,
        isolateVocals=args.isolate,
        exportFormats=args.export if args.export else None,
        chordMethod=args.method,
        beatsPerWindow=args.beats_per_window,
        experiment=args.experiment,
        groqApiKey=args.groq_api_key,
        groqModel=args.groq_model,
        beatsPerChunk=args.beats_per_chunk,
        verbose=args.verbose
    )

    print()
    print("=" * 50)
    print("Áudio carregado com sucesso!")
    print("=" * 50)

    info = result["audioInfo"]
    print(f"Sample Rate: {info['sampleRate']} Hz")
    if info.get("originalSampleRate") != info["sampleRate"]:
        print(f"Sample Rate Original: {info['originalSampleRate']} Hz")
    print(f"Duração: {info['duration']:.2f}s")

    if info["sampleRate"] < 22050:
        print()
        print("AVISO: Sample rate baixo (< 22050 Hz).")
        print("A análise de acordes pode ser menos precisa.")

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
    print("Tonalidade estimada")
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
        f"({len(tempo['beatTimes'])} batidas)"
    )

    if "chunkEvidences" in result and args.verbose:
        print()
        print("=" * 50)
        print("Evidências por chunk")
        print("=" * 50)
        for ev in result["chunkEvidences"]:
            print()
            print(f"  Chunk {ev['start']}s - {ev['end']}s:")
            for w in ev["windowChords"]:
                cands = ", ".join(
                    f"{c['chord']}({c['score']:.2f})"
                    for c in w["candidates"]
                )
                print(f"    {w['time']}s: {cands}")

    if "interpreterResults" in result:
        print()
        print("=" * 50)
        print("Decisões do interpretador")
        print("=" * 50)
        for chunkResult in result["interpreterResults"]:
            for d in chunkResult["decisions"]:
                marker = ""
                if "originalChord" in d:
                    marker = f" (DSP dizia: {d['originalChord']})"
                print(
                    f"  {d['time']:.1f}s -> {d['chord']} "
                    f"(confiança: {d['confidence']:.2f}){marker}"
                )

    if args.list_chords:
        print()
        print("=" * 50)
        print("Cifra")
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
                print(f"  {e['originalChord']} -> {e['chord']}")

    if args.lyrics and "lyrics" in result:
        print()
        print("=" * 50)
        print("Letra transcrita")
        print("=" * 50)
        for seg in result["lyrics"]:
            t = formatTime(seg["start"])
            print(f"{t}  {seg['text']}")

    print()
    print(f"Tempo total: {result['processingTime']:.2f}s")

    if args.export:
        print(f"Arquivos exportados em: {args.output_dir}/")


if __name__ == "__main__":
    main()
