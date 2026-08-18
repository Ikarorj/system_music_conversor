import numpy as np

from harmony.chordDetector import (
    ChordDetector,
    diatonicChordsForKey
)
from harmony.noteUtils import NOTE_NAMES

KRUMHANSL_PROFILES = {
    "major": np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
        2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ]),
    "minor": np.array([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
        2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ])
}


class KeyDetector:

    def detectKey(
        self,
        chroma,
        sampleRate=None,
        windowSeconds=2.0,
        chordWeight=0.5
    ):
        """
        Estima a tonalidade da música combinando dois métodos.

        O primeiro é o de Krumhansl-Schmuckler: correlaciona o chroma
        médio com os perfis de distribuição de notas de tonalidades
        maiores e menores.

        O segundo (quando sampleRate é informado) usa informação
        harmônica: detecta os acordes mais prováveis em janelas de
        tempo com o vocabulário completo e pontua cada tonalidade pela
        fração de acordes que pertencem ao seu campo harmônico. Isso
        resolve ambiguidades típicas do chroma global (ex.: Dó maior
        x Lá menor), já que progressões diferentes implicam em campos
        diferentes.

        Os dois escores são normalizados para [0, 1] e combinados na
        proporção controlada por chordWeight.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem em Hz. Se None,
                usa apenas Krumhansl-Schmuckler.
            windowSeconds (float): Duração das janelas para o
                histograma de acordes.
            chordWeight (float): Peso do método de acordes (0 a 1).
                O restante é o peso do Krumhansl-Schmuckler.

        Returns:
            dict: Tonalidade com maior escore, contendo key, root,
                  mode, score (combinado), ksScore e chordScore.
        """

        ksScores = self._krumhanslScores(chroma)

        if sampleRate is None or chordWeight <= 0.0:
            best = max(
                ksScores,
                key=lambda item: item["score"]
            )
            return {
                "key": best["key"],
                "root": best["root"],
                "mode": best["mode"],
                "score": float(best["score"]),
                "ksScore": float(best["score"]),
                "chordScore": None
            }

        chordScores = self._chordFitScores(
            chroma,
            sampleRate,
            windowSeconds
        )

        ksNorm = self._normalizeScores(
            [item["score"] for item in ksScores]
        )
        chordNorm = self._normalizeScores([
            chordScores[(item["rootIndex"], item["mode"])]
            for item in ksScores
        ])

        combined = []

        for index, item in enumerate(ksScores):

            total = (
                (1.0 - chordWeight) * ksNorm[index]
                + chordWeight * chordNorm[index]
            )

            combined.append({
                "key": item["key"],
                "root": item["root"],
                "mode": item["mode"],
                "score": float(total),
                "ksScore": float(item["score"]),
                "chordScore": float(
                    chordScores[(item["rootIndex"], item["mode"])]
                )
            })

        return max(combined, key=lambda item: item["score"])

    def _krumhanslScores(self, chroma):
        """
        Escore de cada uma das 24 tonalidades por Krumhansl-Schmuckler:
        correlação entre o chroma médio e o perfil de cada tonalidade.

        Returns:
            list: Dicionários com rootIndex, root, mode, key e score.
        """

        chromaMean = chroma.mean(axis=1)
        chromaMean = chromaMean / (np.linalg.norm(chromaMean) + 1e-10)

        scores = []

        for mode, profile in KRUMHANSL_PROFILES.items():

            profileNorm = profile / (np.linalg.norm(profile) + 1e-10)

            for shift in range(12):

                rotated = np.roll(profileNorm, shift)

                scores.append({
                    "rootIndex": shift,
                    "root": NOTE_NAMES[shift],
                    "mode": mode,
                    "key": f"{NOTE_NAMES[shift]} {mode}",
                    "score": float(np.dot(chromaMean, rotated))
                })

        return scores

    def _chordFitScores(self, chroma, sampleRate, windowSeconds):
        """
        Fração de acordes de cada tonalidade (0 a 1): para cada
        tonalidade, quantos acordes detectados (vocabulário completo)
        pertencem ao seu campo harmônico.

        Returns:
            dict: Escores indexados por (rootIndex, mode).
        """

        detector = ChordDetector()

        summary = detector.detectChordSummary(
            chroma,
            sampleRate,
            windowSeconds=windowSeconds,
            smoothWindows=1,
            labels=None
        )

        nWindows = max(1, len(summary))

        chordCounts = {}

        for entry in summary:
            chordCounts[entry["chord"]] = (
                chordCounts.get(entry["chord"], 0) + 1
            )

        scores = {}

        for mode in KRUMHANSL_PROFILES:

            for shift in range(12):

                field = set(diatonicChordsForKey(
                    NOTE_NAMES[shift],
                    mode
                ))

                hits = sum(
                    count for chord, count in chordCounts.items()
                    if chord in field
                )

                scores[(shift, mode)] = hits / nWindows

        return scores

    @staticmethod
    def _normalizeScores(values):
        """
        Normaliza uma lista de escores para [0, 1] (min-max), para que
        métodos com escalas diferentes possam ser combinados.
        """

        values = np.asarray(values, dtype=float)

        low = values.min()
        high = values.max()

        if high - low < 1e-12:
            return np.ones_like(values)

        return (values - low) / (high - low)
