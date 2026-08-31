import logging

import numpy as np

from harmony.noteUtils import NOTE_NAMES

logger = logging.getLogger(__name__)


class ChunkEvidenceBuilder:

    def __init__(self, topCandidates=3):
        """
        Constrói evidências estruturadas por chunk para enviar ao
        interpretador (LLM ou regras).

        Args:
            topCandidates (int): Quantos candidatos por janela
                incluir nas evidências.
        """

        self.topCandidates = topCandidates

    def buildChunkEvidence(
        self,
        chunkStart,
        chunkEnd,
        windowChordResults,
        key,
        tempo,
        beatTimes,
        chroma,
        sampleRate,
        hopLength=512
    ):
        """
        Monta evidência estruturada de um chunk.

        Args:
            chunkStart (float): Início do chunk em segundos.
            chunkEnd (float): Fim do chunk em segundos.
            windowChordResults (list): Resultados do chord detector
                para as janelas deste chunk.
            key (dict): Tonalidade estimada.
            tempo (float): BPM.
            beatTimes (np.ndarray): Batidas do áudio completo.
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem.
            hopLength (int): Hop length usado na extração.

        Returns:
            dict: Evidência estruturada do chunk.
        """

        windowEvidence = []
        for entry in windowChordResults:
            windowEvidence.append({
                "time": round(entry["time"], 2),
                "candidates": [
                    {
                        "chord": c["chord"],
                        "score": round(c["score"], 3)
                    }
                    for c in entry["candidates"][:self.topCandidates]
                ],
                "bestChord": entry["chord"],
                "bestScore": round(entry["score"], 3)
            })

        dominantNotes = self._extractDominantNotes(
            chroma, sampleRate, chunkStart, chunkEnd, hopLength
        )

        bassNote = self._extractBassNote(
            chroma, sampleRate, chunkStart, chunkEnd, hopLength
        )

        chunkBeats = [
            round(float(b), 2)
            for b in beatTimes
            if chunkStart <= b < chunkEnd
        ]

        return {
            "start": round(chunkStart, 2),
            "end": round(chunkEnd, 2),
            "duration": round(chunkEnd - chunkStart, 2),
            "key": f"{key['root']} {key['mode']}",
            "tempo": round(tempo, 1),
            "beats": chunkBeats,
            "windowChords": windowEvidence,
            "dominantNotes": dominantNotes,
            "bassNote": bassNote,
            "allowedChords": sorted(set(
                c["chord"]
                for w in windowEvidence
                for c in w["candidates"]
            ))
        }

    def _extractDominantNotes(
        self, chroma, sampleRate, start, end, hopLength
    ):
        startFrame = int(start * sampleRate / hopLength)
        endFrame = int(end * sampleRate / hopLength)
        endFrame = min(endFrame, chroma.shape[1])

        if endFrame <= startFrame:
            return []

        segment = chroma[:, startFrame:endFrame].mean(axis=1)
        topIndexes = np.argsort(segment)[-3:][::-1]

        return [NOTE_NAMES[i] for i in topIndexes if segment[i] > 0.1]

    def _extractBassNote(
        self, chroma, sampleRate, start, end, hopLength
    ):
        startFrame = int(start * sampleRate / hopLength)
        endFrame = int(end * sampleRate / hopLength)
        endFrame = min(endFrame, chroma.shape[1])

        if endFrame <= startFrame:
            return None

        segment = chroma[:, startFrame:endFrame].mean(axis=1)
        bassIndex = int(np.argmax(segment[:6]))

        if segment[bassIndex] > 0.1:
            return NOTE_NAMES[bassIndex]

        return None

    def formatForLLM(self, evidence):
        """
        Formata evidência como texto para enviar ao LLM.

        Args:
            evidence (dict): Evidência estruturada.

        Returns:
            str: Texto formatado.
        """

        lines = []
        lines.append(f"Trecho: {evidence['start']}s - {evidence['end']}s")
        lines.append(f"Tonalidade: {evidence['key']}")
        lines.append(f"Tempo: {evidence['tempo']} BPM")

        if evidence["beats"]:
            lines.append(
                f"Batidas: {', '.join(str(b) for b in evidence['beats'])}"
            )

        if evidence["dominantNotes"]:
            lines.append(
                f"Notas dominantes: {', '.join(evidence['dominantNotes'])}"
            )

        if evidence["bassNote"]:
            lines.append(f"Nota de baixo: {evidence['bassNote']}")

        lines.append("")
        lines.append("Candidatos detectados pelo DSP:")

        for w in evidence["windowChords"]:
            candidates = ", ".join(
                f"{c['chord']} ({c['score']})"
                for c in w["candidates"]
            )
            lines.append(f"  {w['time']}s: {candidates}")

        lines.append("")
        lines.append(
            "Acordes permitidos: "
            + ", ".join(evidence["allowedChords"])
        )

        return "\n".join(lines)
