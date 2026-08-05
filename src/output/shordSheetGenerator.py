from pathlib import Path

from harmony.chordDetector import diatonicChordsForKey


def _formatTime(seconds):
    minutes = int(seconds // 60)
    remainingSeconds = seconds - minutes * 60
    return f"{minutes:02d}:{remainingSeconds:05.2f}"


class ChordSheetGenerator:

    def generate(
        self,
        chordSummary,
        key,
        windowSeconds=2.0,
        outputPath=None
    ):
        """
        Gera uma cifra simplificada para violão a partir do resumo
        de acordes detectados.

        Args:
            chordSummary (list): Resumo de acordes do
                ChordDetector.detectChordSummary.
            key (dict): Tonalidade estimada (key, root, mode, score).
            windowSeconds (float): Duração da janela dos acordes.
            outputPath (str): Caminho opcional para salvar a cifra.

        Returns:
            str: Texto da cifra.
        """

        lines = []

        lines.append(f"Tom: {key['key']} "
                     f"(correlação {key['score']:.3f})")

        chordLabels = diatonicChordsForKey(
            key["root"],
            key["mode"]
        )

        lines.append(
            "Campo harmônico: "
            + ", ".join(chordLabels)
        )
        lines.append("")
        lines.append(
            f"Cifra aproximada (janela de {windowSeconds:.0f}s):"
        )
        lines.append("")

        for entry in chordSummary:
            time = _formatTime(entry["time"])
            score = entry["score"]
            lines.append(
                f"{time}  {entry['chord']:6s}  "
                f"({score:.2f})"
            )

        lines.append("")
        lines.append("Cifra compacta:")

        compact = " | ".join(
            entry["chord"] for entry in chordSummary
        )
        lines.append(compact)

        sheet = "\n".join(lines)

        if outputPath is not None:
            Path(outputPath).parent.mkdir(
                parents=True,
                exist_ok=True
            )
            Path(outputPath).write_text(
                sheet,
                encoding="utf-8"
            )

        return sheet
