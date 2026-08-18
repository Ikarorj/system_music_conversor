import logging

logger = logging.getLogger(__name__)

SIMPLIFICATION_MAP = {
    "maj7": "",
    "m7": "m",
    "7": "",
    "m7b5": "dim",
    "add9": "",
    "6": "",
    "m6": "m",
    "sus2": "",
    "sus4": "",
    "9": "",
    "m9": "m",
    "11": "",
    "13": "",
    "m11": "m",
    "m13": "m",
    "dim7": "dim",
    "aug": "",
    "7#9": "",
    "7b9": "",
    "7#5": "aug",
    "7b5": "dim",
    "m7#5": "aug",
    "m7b9": "m",
    "m9b5": "dim",
    "1+": "aug",
}

BEGINNER_QUALITIES = {"", "m", "dim", "aug"}

EXEMPT_CHORDS = {
    "G7": True,
    "D7": True,
    "A7": True,
    "E7": True,
    "B7": True,
    "C7": True,
    "F7": True,
}


class ChordSimplifier:

    def __init__(
        self,
        preferOpenChords=True,
        minifySevenths=True
    ):
        """
        Simplifica acordes complexos para versões mais fáceis de
        tocar no violão.

        Args:
            preferOpenChords (bool): Se True, evita acordes com
                muitas cordas abertas difíceis (ex.: F# m7b5 -> F#m).
            minifySevenths (bool): Se True, remove 7ths e extensões
                quando a qualidade simplificada é viável.
        """

        self.preferOpenChords = preferOpenChords
        self.minifySevenths = minifySevenths

    def simplify(self, chordLabel):
        """
        Simplifica um único acorde.

        Args:
            chordLabel (str): Rótulo do acorde (ex.: "Cmaj7", "G7",
                "D#m7b5").

        Returns:
            dict: chord (simplificado), original, changed (bool),
                  reason (str).
        """

        root, quality = self._splitChord(chordLabel)

        if quality in BEGINNER_QUALITIES:
            return {
                "chord": chordLabel,
                "original": chordLabel,
                "changed": False,
                "reason": "já é simples"
            }

        if not self.minifySevenths:
            return {
                "chord": chordLabel,
                "original": chordLabel,
                "changed": False,
                "reason": "simplificação desativada"
            }

        if chordLabel in EXEMPT_CHORDS:
            return {
                "chord": chordLabel,
                "original": chordLabel,
                "changed": False,
                "reason": "acorde comum no violão"
            }

        simplified = self._simplifyQuality(root, quality)

        return {
            "chord": simplified,
            "original": chordLabel,
            "changed": simplified != chordLabel,
            "reason": (
                f"{quality} -> "
                f"{self._splitChord(simplified)[1] or 'maior'}"
            )
        }

    def simplifyProgression(self, chordSummary):
        """
        Simplifica uma lista de acordes (chordSummary ou lista de
        rótulos).

        Args:
            chordSummary (list): Lista de dicts com "chord" ou
                lista de strings.

        Returns:
            list: Lista de dicts com chord, original, changed.
        """

        if not chordSummary:
            return []

        isDictList = isinstance(chordSummary[0], dict)

        results = []
        for entry in chordSummary:
            label = entry["chord"] if isDictList else entry
            result = self.simplify(label)

            if isDictList:
                merged = dict(entry)
                merged["chord"] = result["chord"]
                merged["originalChord"] = result["original"]
                merged["simplified"] = result["changed"]
                results.append(merged)
            else:
                results.append(result)

        simplifiedCount = sum(
            1 for r in results
            if (r.get("changed") or r.get("simplified"))
        )
        if simplifiedCount:
            logger.info(
                "Simplificados %d/%d acordes",
                simplifiedCount, len(results)
            )

        return results

    def _simplifyQuality(self, root, quality):
        simplified = SIMPLIFICATION_MAP.get(quality)

        if simplified is None:
            logger.warning(
                "Qualidade desconhecida: %s (acorde: %s%s)",
                quality, root, quality
            )
            return f"{root}{quality}"

        return f"{root}{simplified}"

    @staticmethod
    def _splitChord(label):
        if len(label) >= 2 and label[1] == "#":
            return label[:2], label[2:]
        return label[0], label[1:]

    @staticmethod
    def guitarDifficulty(chordLabel):
        """
        Estima a dificuldade de um acorde no violão (1-5).

        Args:
            chordLabel (str): Rótulo do acorde.

        Returns:
            int: 1 (fácil) a 5 (difícil).
        """

        root, quality = ChordSimplifier._splitChord(chordLabel)

        difficulty = 1

        if "#" in root or "b" in root:
            difficulty += 1

        if quality in ("", "m"):
            difficulty += 0
        elif quality in ("7", "m7", "sus2", "sus4"):
            difficulty += 1
        elif quality in ("maj7", "dim", "aug"):
            difficulty += 2
        elif quality in ("m7b5", "add9", "6", "m6"):
            difficulty += 2
        else:
            difficulty += 3

        openChordRoots = {"C", "G", "D", "A", "E", "Am", "Em", "Dm"}
        if chordLabel in openChordRoots:
            difficulty = max(1, difficulty - 1)

        return min(5, max(1, difficulty))
