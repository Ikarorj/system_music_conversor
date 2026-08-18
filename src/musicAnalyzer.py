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
from harmony.keyDetector import KeyDetector
from harmony.progressionAnalyzer import ProgressionAnalyzer
from output.sheetGenerator import ChordSheetGenerator
from output.fileExporter import FileExporter

logger = logging.getLogger(__name__)


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
        """
        API pública para análise completa de áudio musical.

        Args:
            preprocessor (AudioPreprocessor): Pré-processador. Se
                None, usa configurações padrão.
            segmenter (AudioSegmenter): Segmentador de áudio. Se
                None, usa chunking automático na detecção.
            chunkedDetection (bool): Se True, usa chunking no Viterbi
                para áudios longos.
            chunkBeats (int): Batidas por chunk no Viterbi chunked.
            simplifyChords (bool): Se True, simplifica acordes
                complexos para violão.
            outputDir (str): Diretório para exportações.
        """

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
        beatsPerWindow=2
    ):
        """
        Análise completa de um arquivo de áudio.

        Args:
            audioPath (str): Caminho do arquivo de áudio.
            language (str): Idioma para transcrição (ex.: "pt").
            transcribeLyrics (bool): Se True, transcreve a letra.
            isolateVocals (bool): Se True, isola voz antes de
                transcrever.
            exportFormats (list): Formatos de exportação:
                "txt", "json", "csv", "musicxml".
            chordMethod (str): "cqt" ou "nnls" para chroma.
            beatsPerWindow (int): Batidas por janela de acorde.

        Returns:
            dict: Resultado da análise completa com chaves:
                audioInfo, key, tempo, chordSummary, chords,
                progression, sheet, simplifiedChords, lyrics.
        """

        t0 = time.time()
        result = {}

        logger.info("Carregando áudio: %s", audioPath)
        audioSignal, sampleRate = self.loader.loadAudio(audioPath)

        result["_originalAudio"] = (audioSignal.copy(), sampleRate)

        audioSignal, sampleRate = self.preprocessor.process(
            audioSignal, sampleRate
        )

        duration = len(audioSignal) / sampleRate
        result["audioInfo"] = {
            "path": audioPath,
            "sampleRate": sampleRate,
            "duration": duration,
            "samples": len(audioSignal)
        }
        logger.info(
            "Áudio: %d Hz, %.1fs, %d amostras",
            sampleRate, duration, len(audioSignal)
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

        logger.info(
            "Tom: %s (score: %.3f)", key["key"], key["score"]
        )
        logger.info(
            "Campo harmônico: %s", ", ".join(chordLabels)
        )

        logger.info("Detectando batidas...")
        tempoInfo = self.tempoExtractor.extractTempo(
            audioSignal, sampleRate
        )
        result["tempo"] = tempoInfo
        logger.info(
            "Tempo: %.1f BPM (%d batidas)",
            tempoInfo["tempo"],
            len(tempoInfo["beatTimes"])
        )

        logger.info("Detectando acordes...")
        if self.chunkedDetection:
            chordSummary = self.chordDetector.detectChordSummaryChunked(
                chroma,
                sampleRate,
                beatTimes=tempoInfo["beatTimes"],
                beatsPerWindow=beatsPerWindow,
                labels=chordLabels,
                chunkBeats=self.chunkBeats,
                stayProb=0.7,
                temperature=8.0,
                fifthBoost=2.5
            )
        else:
            chordSummary = self.chordDetector.detectChordSummaryWithBeats(
                chroma,
                sampleRate,
                beatTimes=tempoInfo["beatTimes"],
                beatsPerWindow=beatsPerWindow,
                labels=chordLabels,
                stayProb=0.7,
                temperature=8.0,
                fifthBoost=2.5
            )

        result["chordSummary"] = chordSummary
        result["chords"] = [
            entry["chord"] for entry in chordSummary
        ]

        logger.info(
            "Acordes detectados: %d janelas", len(chordSummary)
        )

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
        sheet = self.sheetGenerator.generate(
            chordSummary, key
        )
        result["sheet"] = sheet

        lyrics = None
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

            logger.info("Gerando cifra com letras...")
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

    def analyzeAudio(
        self,
        audioSignal,
        sampleRate,
        key=None,
        chordMethod="nnls",
        beatsPerWindow=2
    ):
        """
        Análise a partir de um sinal de áudio já carregado (útil
        para testes e pipelines customizados).

        Args:
            audioSignal (np.ndarray): Sinal de áudio processado.
            sampleRate (int): Taxa de amostragem.
            key (dict): Tonalidade pré-detectada (opcional).
            chordMethod (str): Método de chroma.
            beatsPerWindow (int): Batidas por janela.

        Returns:
            dict: Resultado parcial da análise.
        """

        result = {}

        chroma = self.chromaExtractor.extractChroma(
            audioSignal, sampleRate, method=chordMethod
        )
        result["chroma"] = chroma

        if key is None:
            key = self.keyDetector.detectKey(chroma, sampleRate)
        result["key"] = key

        chordLabels = diatonicChordsForKey(
            key["root"], key["mode"]
        )
        result["diatonicChords"] = chordLabels

        tempoInfo = self.tempoExtractor.extractTempo(
            audioSignal, sampleRate
        )
        result["tempo"] = tempoInfo

        chordSummary = self.chordDetector.detectChordSummaryWithBeats(
            chroma, sampleRate,
            beatTimes=tempoInfo["beatTimes"],
            beatsPerWindow=beatsPerWindow,
            labels=chordLabels
        )
        result["chordSummary"] = chordSummary
        result["chords"] = [e["chord"] for e in chordSummary]

        return result

    def _exportResults(self, result, formats, audioPath):
        from pathlib import Path

        baseName = Path(audioPath).stem

        if "txt" in formats:
            self.exporter.exportTxt(
                result["sheet"],
                f"{baseName}_cifra"
            )
            if "sheetWithLyrics" in result:
                self.exporter.exportTxt(
                    result["sheetWithLyrics"],
                    f"{baseName}_cifra_letra"
                )

        if "json" in formats:
            exportData = {
                "audioInfo": result["audioInfo"],
                "key": result["key"],
                "tempo": {
                    "bpm": result["tempo"]["tempo"]
                },
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
            self.exporter.exportJson(
                exportData, f"{baseName}_analise"
            )

        if "csv" in formats:
            self.exporter.exportCsv(
                result["chordSummary"],
                f"{baseName}_acordes"
            )

        if "musicxml" in formats:
            try:
                self.exporter.exportMusicXml(
                    result["chordSummary"],
                    result["key"],
                    f"{baseName}_partitura"
                )
            except Exception as e:
                logger.warning(
                    "Falha ao exportar MusicXML: %s", e
                )
