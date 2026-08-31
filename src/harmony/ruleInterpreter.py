import logging

from harmony.chordDetector import diatonicChordsForKey
from harmony.noteUtils import NOTE_NAMES

logger = logging.getLogger(__name__)


class RuleInterpreter:

    def __init__(self, diatonicBias=0.15, fifthBonus=0.1):
        """
        Interpretador de acordes baseado em regras musicais.

        Aplica conhecimento de teoria musical sobre os candidatos
        do DSP para refinar a escolha, sem usar LLM.

        Args:
            diatonicBias (float): Bônus para acordes diatônicos
                da tonalidade.
            fifthBonus (float): Bônus para relações de quinta justa.
        """

        self.diatonicBias = diatonicBias
        self.fifthBonus = fifthBonus

    def interpret(self, evidence):
        """
        Interpreta evidência de um chunk usando regras音乐ais.

        Args:
            evidence (dict): Evidência estruturada do chunk.

        Returns:
            dict: Decisões com chord, time, confidence.
        """

        keyParts = evidence["key"].split()
        rootName = keyParts[0] if keyParts else "C"
        mode = keyParts[1] if len(keyParts) > 1 else "major"

        diatonicSet = set(
            diatonicChordsForKey(rootName, mode)
        )

        decisions = []

        for w in evidence["windowChords"]:
            bestChord = self._pickBest(
                w["candidates"],
                diatonicSet,
                evidence["bassNote"],
                evidence["dominantNotes"]
            )

            decisions.append({
                "time": w["time"],
                "chord": bestChord["chord"],
                "confidence": bestChord["score"]
            })

        decisions = self._smoothWithFifths(decisions)

        return {
            "start": evidence["start"],
            "end": evidence["end"],
            "decisions": decisions,
            "model": "rule_based",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }

    def interpretBatch(self, chunkEvidences):
        results = []
        for evidence in chunkEvidences:
            results.append(self.interpret(evidence))
        return results

    def _pickBest(
        self,
        candidates,
        diatonicSet,
        bassNote,
        dominantNotes
    ):
        scored = []

        for c in candidates:
            chord = c["chord"]
            score = c["score"]

            root = self._chordRoot(chord)

            if chord in diatonicSet:
                score += self.diatonicBias

            if bassNote and root == bassNote:
                score += 0.1

            if root in dominantNotes:
                score += 0.05

            score = min(1.0, score)
            scored.append({"chord": chord, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[0] if scored else {"chord": "N.C.", "score": 0}

    def _smoothWithFifths(self, decisions):
        if len(decisions) < 2:
            return decisions

        fifths = {
            0: 7, 1: 8, 2: 3, 3: 4, 4: 5, 5: 0,
            6: 1, 7: 2, 8: 9, 9: 10, 10: 11, 11: 6
        }

        smoothed = [dict(decisions[0])]

        for i in range(1, len(decisions)):
            prevRoot = self._chordRoot(smoothed[-1]["chord"])
            currRoot = self._chordRoot(decisions[i]["chord"])

            prevIdx = NOTE_NAMES.index(prevRoot) if prevRoot in NOTE_NAMES else -1
            currIdx = NOTE_NAMES.index(currRoot) if currRoot in NOTE_NAMES else -1

            if prevIdx >= 0 and currIdx >= 0:
                isFifth = (
                    fifths.get(prevIdx) == currIdx
                    or fifths.get(currIdx) == prevIdx
                )
                if isFifth:
                    decisions[i]["confidence"] = min(
                        1.0, decisions[i]["confidence"] + self.fifthBonus
                    )

            smoothed.append(dict(decisions[i]))

        return smoothed

    @staticmethod
    def _chordRoot(chordLabel):
        if len(chordLabel) >= 2 and chordLabel[1] == "#":
            return chordLabel[:2]
        return chordLabel[0]
