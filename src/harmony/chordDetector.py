import librosa
import numpy as np
from scipy.ndimage import median_filter

from harmony.noteUtils import NOTE_NAMES

CHORD_QUALITIES = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "maj7": [0, 4, 7, 11],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
    "dim": [0, 3, 6],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "m7b5": [0, 3, 6, 10],
    "aug": [0, 4, 8],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "add9": [0, 4, 7, 14]
}

SIMPLE_QUALITIES = [
    "",
    "m",
    "7",
    "m7",
    "sus2",
    "sus4"
]


class ChordDetector:

    def __init__(self, qualities=None):
        """
        Template matching de acordes sobre o chroma.

        Cada template é um vetor binário de 12 semitons que
        representa um acorde (tônica x qualidade). Para cada frame,
        calcula a similaridade cosseno entre o chroma e todos os
        templates, escolhendo o acorde de maior escore.

        Args:
            qualities (list): Qualidades de acorde a considerar.
                Se None, usa todas as qualidades de CHORD_QUALITIES.
        """

        self.qualities = qualities or list(CHORD_QUALITIES.keys())
        self.templates, self.labels = self._buildTemplates()

    def detectChords(
        self,
        chroma,
        sampleRate,
        hopLength=512,
        smoothWidth=15,
        topChords=3
    ):
        """
        Detecta o acorde provável em cada frame do chroma.

        O chroma é centralizado por frame (subtração da média dos 12
        bins) antes da comparação com os templates, reduzindo o viés
        de energia espalhada presente em misturas complexas.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            smoothWidth (int): Largura do filtro mediano sobre a
                sequência de acordes. Usado para remover saltos
                espúrios de um frame isolado.
            topChords (int): Quantos acordes candidatos retornar
                por frame.

        Returns:
            list: Lista de dicionários com time, chord, root,
                  quality, score e candidates.
        """

        centeredChroma = self._centerChroma(chroma)

        scores = self._cosineSimilarity(
            centeredChroma,
            self.templates
        )

        bestIndexes = np.argmax(scores, axis=0)

        if smoothWidth > 1:
            bestIndexes = median_filter(
                bestIndexes,
                size=smoothWidth
            )

        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sampleRate,
            hop_length=hopLength
        )

        detectedChords = []

        nQualities = len(self.qualities)

        for frameIndex in range(chroma.shape[1]):

            bestIndex = int(bestIndexes[frameIndex])

            frameScores = scores[:, frameIndex]

            topIndexes = np.argsort(frameScores)[-topChords:][::-1]

            candidates = [
                {
                    "chord": self.labels[index],
                    "score": float(frameScores[index])
                }
                for index in topIndexes
            ]

            detectedChords.append({
                "time": float(times[frameIndex]),
                "chord": self.labels[bestIndex],
                "root": NOTE_NAMES[bestIndex // nQualities],
                "quality": self.qualities[bestIndex % nQualities],
                "score": float(frameScores[bestIndex]),
                "candidates": candidates
            })

        return detectedChords

    def detectChordSummary(
        self,
        chroma,
        sampleRate,
        hopLength=512,
        windowSeconds=2.0,
        smoothWindows=3,
        topChords=3
    ):
        """
        Gera um resumo de acordes por janela de tempo, ideal para
        misturas complexas (música completa): o chroma de cada janela
        é calculado pela média dos frames da janela, filtrando o ruído
        frame a frame.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            windowSeconds (float): Duração da janela em segundos.
            smoothWindows (int): Largura do filtro mediano sobre os
                acordes das janelas.
            topChords (int): Quantos acordes candidatos por janela.

        Returns:
            list: Lista de dicionários com time, chord, root,
                  quality, score e candidates (um por janela).
        """

        centeredChroma = self._centerChroma(chroma)

        windowFrames = int(windowSeconds * sampleRate / hopLength)

        nWindows = centeredChroma.shape[1] // windowFrames

        windowChroma = np.zeros((12, nWindows))

        for window in range(nWindows):

            sliceStart = window * windowFrames
            sliceEnd = sliceStart + windowFrames

            windowChroma[:, window] = centeredChroma[
                :, sliceStart:sliceEnd
            ].mean(axis=1)

        scores = self._cosineSimilarity(windowChroma, self.templates)

        bestIndexes = np.argmax(scores, axis=0)

        if smoothWindows > 1:
            bestIndexes = median_filter(
                bestIndexes,
                size=smoothWindows
            )

        nQualities = len(self.qualities)

        summary = []

        for window in range(nWindows):

            bestIndex = int(bestIndexes[window])

            windowScores = scores[:, window]

            topIndexes = np.argsort(windowScores)[-topChords:][::-1]

            candidates = [
                {
                    "chord": self.labels[index],
                    "score": float(windowScores[index])
                }
                for index in topIndexes
            ]

            summary.append({
                "time": float(window * windowSeconds),
                "chord": self.labels[bestIndex],
                "root": NOTE_NAMES[bestIndex // nQualities],
                "quality": self.qualities[bestIndex % nQualities],
                "score": float(windowScores[bestIndex]),
                "candidates": candidates
            })

        return summary

    def _buildTemplates(self):
        """
        Constrói a matriz de templates (12, n_templates) e os rótulos
        correspondentes no formato "C", "Am", "G7", etc.
        """

        templates = []
        labels = []

        for root in range(12):
            for quality in self.qualities:
                vector = np.zeros(12)

                for semitone in CHORD_QUALITIES[quality]:
                    vector[(root + semitone) % 12] = 1.0

                templates.append(vector)
                labels.append(f"{NOTE_NAMES[root]}{quality}")

        return np.array(templates).T, labels

    @staticmethod
    def _centerChroma(chroma):
        """
        Subtrai a média dos 12 bins de cada frame, removendo o DC de
        energia espalhada que reduz a discriminação entre acordes.
        """

        return chroma - chroma.mean(axis=0, keepdims=True)

    @staticmethod
    def _cosineSimilarity(chroma, templates):
        """
        Similaridade cosseno entre cada coluna do chroma e cada
        template. Retorna matriz (n_templates, n_frames).
        """

        chromaNorm = np.linalg.norm(chroma, axis=0, keepdims=True)
        chromaNormed = chroma / np.maximum(chromaNorm, 1e-10)

        templateNorm = np.linalg.norm(templates, axis=0, keepdims=True)
        templatesNormed = templates / np.maximum(templateNorm, 1e-10)

        return templatesNormed.T @ chromaNormed
