import numpy as np

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

    def detectKey(self, chroma):
        """
        Estima a tonalidade da música usando o método de
        Krumhansl-Schmuckler: correlaciona o chroma médio com
        os perfis de distribuição de notas de tonalidades maiores
        e menores.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).

        Returns:
            dict: Tonalidade com maior correlação, contendo key,
                  root, mode e score.
        """

        chromaMean = chroma.mean(axis=1)
        chromaMean = chromaMean / (np.linalg.norm(chromaMean) + 1e-10)

        best = None

        for mode, profile in KRUMHANSL_PROFILES.items():

            profileNorm = profile / (np.linalg.norm(profile) + 1e-10)

            for shift in range(12):

                rotated = np.roll(profileNorm, shift)

                score = float(np.dot(chromaMean, rotated))

                if best is None or score > best["score"]:
                    best = {
                        "key": f"{NOTE_NAMES[shift]} {mode}",
                        "root": NOTE_NAMES[shift],
                        "mode": mode,
                        "score": score
                    }

        return best
