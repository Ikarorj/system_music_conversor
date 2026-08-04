import librosa
import numpy as np
from scipy.ndimage import median_filter

from harmony.noteUtils import frequencyToNote


class PredominantPitchDetector:

    def detectPredominantNote(
        self,
        audioSignal,
        sampleRate,
        hopLength=2048,
        fmin=None,
        fmax=None,
        medianWidth=5,
        energyGate=None
    ):
        """
        Estima a nota predominante em cada frame do áudio utilizando
        rastreamento de frequência fundamental (pYIN).

        A detecção é refinada com:
          - Filtro de energia (RMS) para ignorar silêncios/ruído leve;
          - Filtro mediano sobre a trajetória de f0 para reduzir
            saltos espúrios de oitava ou harmônicos.

        Args:
            audioSignal (np.ndarray): Sinal de áudio.
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            fmin (float): Frequência mínima (Hz) a ser rastreada.
            fmax (float): Frequência máxima (Hz) a ser rastreada.
            medianWidth (int): Largura do filtro mediano aplicado
                               sobre a trajetória de f0.
            energyGate (float): Limiar de energia RMS. Se None, é
                                calculado automaticamente.

        Returns:
            list: Lista de dicionários com time, note, octave,
                  frequency, cents e confidence por frame.
        """

        if fmin is None:
            fmin = librosa.note_to_hz("C2")

        if fmax is None:
            fmax = librosa.note_to_hz("C7")

        f0, voicedFlags, voicedProbabilities = librosa.pyin(
            y=audioSignal,
            fmin=fmin,
            fmax=fmax,
            sr=sampleRate,
            hop_length=hopLength,
            fill_na=np.nan
        )

        frameEnergy = self._frameEnergy(
            audioSignal,
            sampleRate,
            hopLength
        )

        if energyGate is None:
            energyGate = self._autoEnergyGate(frameEnergy)

        voiced = (
            voicedFlags
            & ~np.isnan(f0)
            & (frameEnergy >= energyGate)
        )

        f0Smoothed = self._smoothF0(f0, voiced, medianWidth)

        times = librosa.times_like(
            f0,
            sr=sampleRate,
            hop_length=hopLength
        )

        detectedNotes = []

        for timeValue, frequency, voicedFrame, probability in zip(
            times,
            f0Smoothed,
            voiced,
            voicedProbabilities
        ):

            if not voicedFrame:
                detectedNotes.append({
                    "time": float(timeValue),
                    "note": None,
                    "octave": None,
                    "frequency": None,
                    "cents": None,
                    "confidence": 0.0
                })
                continue

            noteName, octave, cents = frequencyToNote(frequency)

            detectedNotes.append({
                "time": float(timeValue),
                "note": noteName,
                "octave": octave,
                "frequency": float(frequency),
                "cents": cents,
                "confidence": float(probability)
            })

        return detectedNotes

    def detectNoteSegments(
        self,
        detectedNotes,
        minDuration=0.5
    ):
        """
        Agrupa frames consecutivos com a mesma nota em segmentos,
        ignorando frames sem nota.

        Returns:
            list: Lista de dicionários com note, octave, start, end
                  e duration.
        """

        segments = []

        currentNote = None
        currentOctave = None
        currentStart = None
        currentEnd = None

        for entry in detectedNotes:

            note = entry["note"]
            octave = entry["octave"]

            if note is not None and note == currentNote:
                currentEnd = entry["time"]

            elif note is not None:
                self._flushSegment(
                    segments,
                    currentNote,
                    currentOctave,
                    currentStart,
                    currentEnd,
                    minDuration
                )

                currentNote = note
                currentOctave = octave
                currentStart = entry["time"]
                currentEnd = entry["time"]

            else:
                self._flushSegment(
                    segments,
                    currentNote,
                    currentOctave,
                    currentStart,
                    currentEnd,
                    minDuration
                )

                currentNote = None
                currentOctave = None
                currentStart = None
                currentEnd = None

        self._flushSegment(
            segments,
            currentNote,
            currentOctave,
            currentStart,
            currentEnd,
            minDuration
        )

        return segments

    @staticmethod
    def _autoEnergyGate(rms):

        return max(
            float(rms.mean()) * 0.25,
            float(rms.max()) * 0.05
        )

    @staticmethod
    def _frameEnergy(audioSignal, sampleRate, hopLength):

        return librosa.feature.rms(
            y=audioSignal,
            frame_length=2048,
            hop_length=hopLength
        )[0]

    @staticmethod
    def _smoothF0(f0, voiced, medianWidth):

        if medianWidth <= 1 or not voiced.any():
            smoothed = f0.copy()
        else:
            voicedIndexes = np.where(voiced)[0]

            interpolated = np.interp(
                np.arange(len(f0)),
                voicedIndexes,
                f0[voicedIndexes]
            )

            smoothed = median_filter(
                interpolated,
                size=medianWidth
            )

        smoothed[~voiced] = np.nan

        return smoothed

    @staticmethod
    def _flushSegment(
        segments,
        note,
        octave,
        start,
        end,
        minDuration
    ):

        if note is None or start is None or end is None:
            return

        duration = end - start

        if duration < minDuration:
            return

        segments.append({
            "note": note,
            "octave": octave,
            "start": start,
            "end": end,
            "duration": duration
        })
