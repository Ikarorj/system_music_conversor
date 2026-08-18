import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileExporter:

    def __init__(self, outputDir="output"):
        """
        Exporta resultados de análise para vários formatos.

        Args:
            outputDir (str): Diretório de saída padrão.
        """

        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)

    def exportTxt(self, content, filename):
        """
        Exporta texto puro (cifra, letras, etc).

        Args:
            content (str): Conteúdo a ser salvo.
            filename (str): Nome do arquivo (sem extensão).

        Returns:
            str: Caminho completo do arquivo salvo.
        """

        path = self.outputDir / f"{filename}.txt"
        path.write_text(content, encoding="utf-8")
        logger.info("Exportado: %s", path)
        return str(path)

    def exportJson(self, data, filename):
        """
        Exporta dados estruturados como JSON.

        Args:
            data: Dados serializáveis (dict, list, etc).
            filename (str): Nome do arquivo (sem extensão).

        Returns:
            str: Caminho completo do arquivo salvo.
        """

        path = self.outputDir / f"{filename}.json"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info("Exportado: %s", path)
        return str(path)

    def exportMusicXml(self, chordSummary, key, filename):
        """
        Exporta a cifra como MusicXML usando music21.

        Args:
            chordSummary (list): Resumo de acordes detectados.
            key (dict): Tonalidade estimada.
            filename (str): Nome do arquivo (sem extensão).

        Returns:
            str: Caminho completo do arquivo salvo.
        """

        from music21 import (
            chord, key as m21key, meter,
            stream, tempo, note
        )

        score = stream.Score()
        part = stream.Part()

        keySignature = m21key.Key(
            key["root"],
            key["mode"]
        )
        score.insert(0, keySignature)

        timeSignature = meter.TimeSignature("4/4")
        score.insert(0, timeSignature)

        bpm = 120
        metronomeMark = tempo.MetronomeMark(number=bpm)
        score.insert(0, metronomeMark)

        for entry in chordSummary:
            chordSymbol = entry["chord"]
            root, quality = self._splitChord(chordSymbol)

            try:
                pitchName = root
                if quality:
                    chordObj = chord.Chord(
                        self._buildChordPitches(root, quality)
                    )
                else:
                    chordObj = chord.Chord([root])
            except Exception:
                chordObj = note.Note(root)

            part.append(chordObj)

        score.insert(0, part)

        path = self.outputDir / f"{filename}.musicxml"
        score.write("musicxml", fp=str(path))
        logger.info("MusicXML exportado: %s", path)
        return str(path)

    def exportCsv(self, chordSummary, filename):
        """
        Exporta o resumo de acordes como CSV.

        Args:
            chordSummary (list): Resumo de acordes detectados.
            filename (str): Nome do arquivo (sem extensão).

        Returns:
            str: Caminho completo do arquivo salvo.
        """

        lines = ["time,chord,root,quality,score"]

        for entry in chordSummary:
            lines.append(
                f"{entry['time']:.3f},"
                f"{entry['chord']},"
                f"{entry['root']},"
                f"{entry['quality']},"
                f"{entry['score']:.4f}"
            )

        path = self.outputDir / f"{filename}.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("CSV exportado: %s", path)
        return str(path)

    @staticmethod
    def _splitChord(label):
        if len(label) >= 2 and label[1] == "#":
            return label[:2], label[2:]
        return label[0], label[1:]

    @staticmethod
    def _buildChordPitches(root, quality):
        from harmony.chordDetector import CHORD_QUALITIES
        from harmony.noteUtils import NOTE_NAMES

        rootIndex = NOTE_NAMES.index(root)
        semitones = CHORD_QUALITIES.get(quality, [0, 4, 7])

        return [
            NOTE_NAMES[(rootIndex + s) % 12]
            for s in semitones
        ]
