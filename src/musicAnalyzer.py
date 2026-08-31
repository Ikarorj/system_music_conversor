import logging
import time

import numpy as np

from audio.audioLoader import AudioLoader
from audio.audioPreprocessor import AudioPreprocessor
from audio.audioSementer import AudioSegmenter
from features.chromaExtractor import ChromaExtractor
from features.tempoExtractor import TempoExtractor
from harmony.chordDetector import (
    ChordDetector,
    diatonicChordsForKey
)
from harmony.chordSimplifier import ChordSimplifier
from harmony.chunkEvidence import ChunkEvidenceBuilder
from harmony.keyDetector import KeyDetector
from harmony.progressionAnalyzer import ProgressionAnalyzer
from output.sheetGenerator import ChordSheetGenerator
from output.fileExporter import FileExporter

logger = logging.getLogger(__name__)

CHROMA_HOP = 512


def hopSeconds(sampleRate):
    """Duração de um frame de chroma (hop=512) em segundos."""
    return CHROMA_HOP / sampleRate


class MusicAnalyzer:

    def __init__(
        self,
        preprocessor=None,
        segmenter=None,
        chunkedDetection=True,
        chunkBeats=200,
        simplifyChords=True,
        outputDir="output"
    ):
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.segmenter = segmenter
        self.chunkedDetection = chunkedDetection
        self.chunkBeats = chunkBeats
        self.simplifyChords = simplifyChords

        self.loader = AudioLoader()
        self.chromaExtractor = ChromaExtractor()
        self.tempoExtractor = TempoExtractor()
        self.keyDetector = KeyDetector()
        self.chordDetector = ChordDetector()
        self.progressionAnalyzer = ProgressionAnalyzer()
        self.evidenceBuilder = ChunkEvidenceBuilder()
        self.sheetGenerator = ChordSheetGenerator()
        self.exporter = FileExporter(outputDir)

    def analyze(
        self,
        audioPath,
        language=None,
        transcribeLyrics=False,
        isolateVocals=False,
        exportFormats=None,
        chordMethod="nnls",
        beatsPerWindow=4,
        experiment=None,
        groqApiKey=None,
        groqModel="llama-3.3-70b-versatile",
        beatsPerChunk=8,
        verbose=False
    ):
        """
        Análise completa de um arquivo de áudio.

        Args:
            audioPath: Caminho do áudio.
            experiment (str): "a" (DSP puro), "b" (DSP+regras),
                "c" (DSP+LLM), ou None (DSP puro).
            groqApiKey (str): Chave Groq para experimento C.
            groqModel (str): Modelo Groq para experimento C.
            beatsPerChunk (int): Batidas por chunk para evidências.
            verbose (bool): Mostrar evidências detalhadas.
        """

        t0 = time.time()
        result = {}

        logger.info("Carregando áudio: %s", audioPath)
        audioSignal, sampleRate = self.loader.loadAudio(audioPath)
        result["_originalAudio"] = (audioSignal.copy(), sampleRate)
        originalSampleRate = sampleRate

        audioSignal, sampleRate = self.preprocessor.process(
            audioSignal, sampleRate
        )

        duration = len(audioSignal) / sampleRate
        result["audioInfo"] = {
            "path": audioPath,
            "sampleRate": sampleRate,
            "originalSampleRate": originalSampleRate,
            "duration": duration,
            "samples": len(audioSignal)
        }

        if originalSampleRate < 22050:
            logger.warning(
                "Sample rate original (%d Hz) é baixo. "
                "Resultado pode ser menos preciso.",
                originalSampleRate
            )

        logger.info("Extraindo chroma (%s)...", chordMethod)
        chroma = self.chromaExtractor.extractChroma(
            audioSignal, sampleRate, method=chordMethod
        )

        logger.info("Detectando tonalidade...")
        key = self.keyDetector.detectKey(chroma, sampleRate)
        result["key"] = key

        chordLabels = diatonicChordsForKey(
            key["root"], key["mode"]
        )
        result["diatonicChords"] = chordLabels

        logger.info("Tom: %s (score: %.3f)", key["key"], key["score"])

        logger.info("Detectando batidas...")
        tempoInfo = self.tempoExtractor.extractTempo(
            audioSignal, sampleRate
        )
        result["tempo"] = tempoInfo

        tempo = tempoInfo["tempo"]
        beatTimes = tempoInfo["beatTimes"]
        logger.info("Tempo: %.1f BPM (%d batidas)", tempo, len(beatTimes))

        logger.info("Detectando acordes (DSP puro)...")
        chordSummary = self.chordDetector.detectChordSummary(
            chroma, sampleRate,
            windowSeconds=max(2.0, beatsPerWindow * 2.0 * hopSeconds(sampleRate)),
            smoothWindows=1,
            labels=chordLabels
        )

        if self._isDegenerate(chordSummary):
            logger.warning(
                "Detecção degenerada com campo harmônico diatônico "
                "(%d acordes únicos). Retentando com vocabulário "
                "completo.",
                len(set(e["chord"] for e in chordSummary))
            )
            chordSummary = self.chordDetector.detectChordSummary(
                chroma, sampleRate,
                windowSeconds=max(2.0, beatsPerWindow * 2.0 * hopSeconds(sampleRate)),
                smoothWindows=1,
                labels=None
            )

        result["chordSummary"] = chordSummary
        result["chords"] = [e["chord"] for e in chordSummary]

        if experiment in ("b", "c"):
            logger.info("Construindo evidências por chunk...")
            chunkEvidences = self._buildChunkEvidences(
                chordSummary, key, tempo, beatTimes,
                chroma, sampleRate, beatsPerChunk
            )

            result["chunkEvidences"] = chunkEvidences

            if verbose:
                for ev in chunkEvidences:
                    logger.info(
                        "\n%s",
                        self.evidenceBuilder.formatForLLM(ev)
                    )

            if experiment == "c":
                logger.info("Interpretando com LLM...")
                from harmony.llmInterpreter import LLMInterpreter
                interpreter = LLMInterpreter(apiKey=groqApiKey, model=groqModel)
            else:
                logger.info("Interpretando com regras musicais...")
                from harmony.ruleInterpreter import RuleInterpreter
                interpreter = RuleInterpreter()

            interpResults = interpreter.interpretBatch(chunkEvidences)
            result["interpreterResults"] = interpResults

            chordSummary = self._mergeInterpretation(
                interpResults, chordSummary
            )
            result["chordSummary"] = chordSummary
            result["chords"] = [e["chord"] for e in chordSummary]

        logger.info("Analisando progressão...")
        progression = self.progressionAnalyzer.detectProgression(
            chordSummary
        )
        result["progression"] = progression

        if self.simplifyChords:
            logger.info("Simplificando acordes...")
            simplifier = ChordSimplifier()
            simplified = simplifier.simplifyProgression(chordSummary)
            result["simplifiedChords"] = simplified

        logger.info("Gerando cifra...")
        sheet = self.sheetGenerator.generate(chordSummary, key)
        result["sheet"] = sheet

        if transcribeLyrics:
            logger.info("Transcrevendo letras...")
            from lyrics.lyricsTranscriber import LyricsTranscriber
            transcriber = LyricsTranscriber()
            lyrics = transcriber.transcribe(
                audioSignal, sampleRate,
                language=language,
                isolateVocals=isolateVocals
            )
            result["lyrics"] = lyrics
            sheetWithLyrics = self.sheetGenerator.generateWithLyrics(
                chordSummary, key, lyrics
            )
            result["sheetWithLyrics"] = sheetWithLyrics

        if exportFormats:
            self._exportResults(result, exportFormats, audioPath)

        elapsed = time.time() - t0
        result["processingTime"] = elapsed
        logger.info("Análise concluída em %.2fs", elapsed)

        return result

    def _buildChunkEvidences(
        self, chordSummary, key, tempo, beatTimes,
        chroma, sampleRate, beatsPerChunk
    ):
        beatTimes = np.asarray(beatTimes, dtype=float)

        if len(beatTimes) == 0:
            return []

        boundaries = [float(beatTimes[0])]
        for i in range(beatsPerChunk, len(beatTimes), beatsPerChunk):
            boundaries.append(float(beatTimes[i]))
        boundaries.append(float(beatTimes[-1]) + 1.0)

        chunkEvidences = []

        for i in range(len(boundaries) - 1):
            chunkStart = boundaries[i]
            chunkEnd = boundaries[i + 1]

            chunkWindows = [
                w for w in chordSummary
                if w["time"] >= chunkStart - 0.5
                and w["time"] < chunkEnd
            ]

            if not chunkWindows:
                continue

            evidence = self.evidenceBuilder.buildChunkEvidence(
                chunkStart, chunkEnd,
                chunkWindows, key, tempo,
                beatTimes, chroma, sampleRate
            )
            chunkEvidences.append(evidence)

        return chunkEvidences

    @staticmethod
    def _mergeInterpretation(interpResults, originalSummary):
        interpMap = {}
        for result in interpResults:
            for decision in result["decisions"]:
                interpMap[round(decision["time"], 2)] = decision

        merged = []
        for entry in originalSummary:
            newEntry = dict(entry)
            timeKey = round(entry["time"], 2)
            if timeKey in interpMap:
                newEntry["chord"] = interpMap[timeKey]["chord"]
                newEntry["interpConfidence"] = interpMap[timeKey]["confidence"]
                newEntry["originalChord"] = entry["chord"]
                root, quality = newEntry["chord"][:1], newEntry["chord"][1:]
                if len(newEntry["chord"]) >= 2 and newEntry["chord"][1] == "#":
                    root, quality = newEntry["chord"][:2], newEntry["chord"][2:]
                newEntry["root"] = root
                newEntry["quality"] = quality
            merged.append(newEntry)

        return merged

    @staticmethod
    def _isDegenerate(chordSummary, minUnique=3, maxSameRatio=0.85):
        """
        Verifica se a detecção de acordes é degenerada: poucos acordes
        únicos ou um único acorde dominando a maioria das janelas.

        Args:
            chordSummary (list): Resumo de acordes detectados.
            minUnique (int): Número mínimo de acordes únicos esperado.
            maxSameRatio (float): Razão máxima permitida para um
                único acorde (0 a 1).

        Returns:
            bool: True se a detecção é degenerada.
        """

        if not chordSummary:
            return True

        chords = [e["chord"] for e in chordSummary]
        uniqueChords = set(chords)

        if len(uniqueChords) < minUnique:
            return True

        from collections import Counter
        counts = Counter(chords)
        mostCommonRatio = counts.most_common(1)[0][1] / len(chords)

        return mostCommonRatio > maxSameRatio

    def _exportResults(self, result, formats, audioPath):
        from pathlib import Path
        baseName = Path(audioPath).stem

        if "txt" in formats:
            self.exporter.exportTxt(result["sheet"], f"{baseName}_cifra")
            if "sheetWithLyrics" in result:
                self.exporter.exportTxt(
                    result["sheetWithLyrics"], f"{baseName}_cifra_letra"
                )

        if "json" in formats:
            exportData = {
                "audioInfo": result["audioInfo"],
                "key": result["key"],
                "tempo": {"bpm": result["tempo"]["tempo"]},
                "chordSummary": [
                    {
                        "time": e["time"],
                        "chord": e["chord"],
                        "root": e["root"],
                        "quality": e["quality"],
                        "score": e["score"]
                    }
                    for e in result["chordSummary"]
                ],
                "progression": result["progression"]
            }
            if "simplifiedChords" in result:
                exportData["simplifiedChords"] = [
                    {
                        "chord": e["chord"],
                        "originalChord": e.get("originalChord", e["chord"]),
                        "simplified": e.get("simplified", False)
                    }
                    for e in result["simplifiedChords"]
                ]
            self.exporter.exportJson(exportData, f"{baseName}_analise")

        if "csv" in formats:
            self.exporter.exportCsv(
                result["chordSummary"], f"{baseName}_acordes"
            )

        if "musicxml" in formats:
            try:
                self.exporter.exportMusicXml(
                    result["chordSummary"], result["key"],
                    f"{baseName}_partitura"
                )
            except Exception as e:
                logger.warning("Falha ao exportar MusicXML: %s", e)
