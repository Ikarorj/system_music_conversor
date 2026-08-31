def _formatTimestamp(seconds):
    minutes = int(seconds // 60)
    remainingSeconds = seconds - minutes * 60
    return f"{minutes:02d}:{remainingSeconds:05.2f}"


class ProgressionAnalyzer:

    def detectProgression(
        self,
        chordFrames,
        minDuration=1.0
    ):
        """
        Agrupa frames consecutivos com o mesmo acorde em segmentos,
        formando a progressão de acordes da música.

        Args:
            chordFrames (list): Lista de dicionários vindos do
                ChordDetector (time e chord).
            minDuration (float): Duração mínima (em segundos) para
                um acorde ser considerado parte da progressão.

        Returns:
            list: Lista de dicionários com chord, start, end,
                  duration e timestamp.
        """

        segments = []

        currentChord = None
        currentStart = None
        currentEnd = None

        for entry in chordFrames:

            chord = entry["chord"]

            if chord == currentChord:
                currentEnd = entry["time"]
            else:
                self._flushSegment(
                    segments,
                    currentChord,
                    currentStart,
                    currentEnd,
                    minDuration
                )

                currentChord = chord
                currentStart = entry["time"]
                currentEnd = entry["time"]

        self._flushSegment(
            segments,
            currentChord,
            currentStart,
            currentEnd,
            minDuration
        )

        return segments

    @staticmethod
    def _flushSegment(
        segments,
        chord,
        start,
        end,
        minDuration
    ):

        if chord is None or start is None or end is None:
            return

        duration = end - start

        if duration < minDuration:
            return

        segments.append({
            "chord": chord,
            "start": start,
            "end": end,
            "duration": duration,
            "timestamp": _formatTimestamp(start)
        })
