import logging

import numpy as np

logger = logging.getLogger(__name__)


class AudioSegmenter:

    def __init__(
        self,
        chunkSeconds=30.0,
        overlapSeconds=5.0,
        minChunkSeconds=10.0
    ):
        """
        Segmenta áudios longos em chunks para processamento em
        paralelo ou sequencial, com sobreposição para evitar
        cortes em notas/acordes.

        Args:
            chunkSeconds (float): Duração de cada chunk em segundos.
            overlapSeconds (float): Sobreposição entre chunks
                consecutivos em segundos.
            minChunkSeconds (float): Duração mínima para o último
                chunk (se menor, é descartado).
        """

        self.chunkSeconds = chunkSeconds
        self.overlapSeconds = overlapSeconds
        self.minChunkSeconds = minChunkSeconds

    def segment(self, audioSignal, sampleRate):
        """
        Divide o áudio em chunks com sobreposição.

        Args:
            audioSignal (np.ndarray): Sinal de áudio completo.
            sampleRate (int): Taxa de amostragem em Hz.

        Returns:
            list: Lista de dicts com chunk (np.ndarray), startTime,
                  endTime, index e overlap.
        """

        totalDuration = len(audioSignal) / sampleRate

        if totalDuration <= self.chunkSeconds:
            logger.info(
                "Áudio curto (%.1fs), sem segmentação", totalDuration
            )
            return [{
                "chunk": audioSignal,
                "startTime": 0.0,
                "endTime": totalDuration,
                "index": 0,
                "overlap": 0.0
            }]

        step = self.chunkSeconds - self.overlapSeconds
        chunks = []
        index = 0
        startTime = 0.0

        while startTime < totalDuration:
            endTime = min(startTime + self.chunkSeconds, totalDuration)

            chunkDuration = endTime - startTime
            if chunkDuration < self.minChunkSeconds:
                break

            startSample = int(startTime * sampleRate)
            endSample = int(endTime * sampleRate)

            chunk = audioSignal[startSample:endSample]

            chunks.append({
                "chunk": chunk,
                "startTime": startTime,
                "endTime": endTime,
                "index": index,
                "overlap": self.overlapSeconds if index > 0 else 0.0
            })

            index += 1
            startTime += step

        logger.info(
            "Áudio segmentado em %d chunks de ~%.1fs "
            "(sobreposição %.1fs)",
            len(chunks), self.chunkSeconds, self.overlapSeconds
        )

        return chunks

    def mergeResults(self, chunkResults, chunkMeta):
        """
        Combina resultados de detecção de múltiplos chunks,
        removendo duplicatas na região de sobreposição.

        Args:
            chunkResults (list): Lista de listas, cada uma com os
                resultados de um chunk (ex.: chordSummary).
            chunkMeta (list): Metadados de cada chunk (startTime,
                endTime, overlap).

        Returns:
            list: Resultados mesclados, ordenados por tempo.
        """

        if not chunkResults:
            return []

        allEntries = []

        for chunkIdx, entries in enumerate(chunkResults):
            meta = chunkMeta[chunkIdx]
            offset = meta["startTime"]
            overlap = meta["overlap"]

            for entry in entries:
                adjustedEntry = dict(entry)
                adjustedEntry["time"] = entry["time"] + offset

                if "endTime" in entry:
                    adjustedEntry["endTime"] = (
                        entry["endTime"] + offset
                    )

                if chunkIdx > 0 and entry["time"] < overlap:
                    adjustedEntry["_overlap"] = True

                allEntries.append(adjustedEntry)

        allEntries.sort(key=lambda e: e["time"])

        merged = []
        for entry in allEntries:
            if entry.get("_overlap") and merged:
                prev = merged[-1]
                if (
                    abs(entry["time"] - prev["time"]) < 1.0
                    and entry.get("chord") == prev.get("chord")
                ):
                    continue

            entry.pop("_overlap", None)
            merged.append(entry)

        return merged

    def mergeChromas(self, chunkChromas, chunkMeta):
        """
        Concatena chromas de chunks com média na região de
        sobreposição.

        Args:
            chunkChromas (list): Lista de arrays chroma (12, n_frames).
            chunkMeta (list): Metadados de cada chunk.

        Returns:
            np.ndarray: Chroma concatenado (12, n_total_frames).
        """

        if not chunkChromas:
            return np.zeros((12, 0))

        if len(chunkChromas) == 1:
            return chunkChromas[0]

        hopLength = 512
        sampleRate = 22050

        overlapFrames = int(
            chunkMeta[0]["overlap"] * sampleRate / hopLength
        )

        result = chunkChromas[0]

        for i in range(1, len(chunkChromas)):
            current = chunkChromas[i]

            if overlapFrames > 0 and result.shape[1] >= overlapFrames:
                avgRegion = (
                    result[:, -overlapFrames:]
                    + current[:, :overlapFrames]
                ) / 2
                result = np.concatenate([
                    result[:, :-overlapFrames],
                    avgRegion,
                    current[:, overlapFrames:]
                ], axis=1)
            else:
                result = np.concatenate(
                    [result, current], axis=1
                )

        return result
