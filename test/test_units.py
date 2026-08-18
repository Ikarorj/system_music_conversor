import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from audio.audioPreprocessor import AudioPreprocessor
from audio.audioSementer import AudioSegmenter
from features.chromaExtractor import ChromaExtractor
from features.spectralExtractor import SpectralExtractor
from features.tempoExtractor import TempoExtractor
from harmony.chordDetector import (
    ChordDetector,
    diatonicChordsForKey,
    splitChordLabel,
    CHORD_QUALITIES,
    SIMPLE_QUALITIES
)
from harmony.chordSimplifier import ChordSimplifier
from harmony.keyDetector import KeyDetector
from harmony.noteUtils import NOTE_NAMES, frequencyToNote
from harmony.progressionAnalyzer import ProgressionAnalyzer
from output.fileExporter import FileExporter
from output.sheetGenerator import ChordSheetGenerator


SR = 44100
NOTE_HZ = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88
}


def synthesizeChord(rootName, quality, seconds=2.0, sr=SR):
    semitones = CHORD_QUALITIES[quality]
    t = np.arange(int(seconds * sr)) / sr
    rootFreq = NOTE_HZ[rootName]
    signal = np.zeros_like(t)
    for semitone in semitones:
        freq = rootFreq * 2 ** (semitone / 12)
        envelope = np.exp(-2.0 * t) * (1 - np.exp(-t * 60))
        signal += envelope * np.sin(2 * np.pi * freq * t)
        signal += 0.5 * envelope * np.sin(2 * np.pi * freq * 2 * t)
    signal += 0.01 * np.random.default_rng(0).standard_normal(len(t))
    return signal / np.max(np.abs(signal))


class TestNoteUtils:

    def test_frequencyToNote_A440(self):
        note, octave, cents = frequencyToNote(440.0)
        assert note == "A"
        assert octave == 4
        assert abs(cents) <= 5

    def test_frequencyToNote_C261(self):
        note, octave, cents = frequencyToNote(261.63)
        assert note == "C"
        assert octave == 4

    def test_frequencyToNote_none(self):
        assert frequencyToNote(None) == (None, None, None)

    def test_frequencyToNote_nan(self):
        assert frequencyToNote(float("nan")) == (None, None, None)

    def test_frequencyToNote_zero(self):
        assert frequencyToNote(0) == (None, None, None)


class TestChordDetector:

    def test_buildTemplates_count(self):
        detector = ChordDetector()
        assert detector.templates.shape == (12, 12 * len(SIMPLE_QUALITIES))

    def test_splitChordLabel_major(self):
        assert splitChordLabel("C") == ("C", "")

    def test_splitChordLabel_minor(self):
        assert splitChordLabel("Am") == ("A", "m")

    def test_splitChordLabel_sharp(self):
        assert splitChordLabel("C#m") == ("C#", "m")

    def test_splitChordLabel_maj7(self):
        assert splitChordLabel("Cmaj7") == ("C", "maj7")

    def test_diatonicChords_major_has_7_chords(self):
        chords = diatonicChordsForKey("C", "major")
        assert len(chords) >= 7

    def test_diatonicChords_C_major_starts_with_C(self):
        chords = diatonicChordsForKey("C", "major")
        assert chords[0] == "C"

    def test_diatonicChords_no_duplicates(self):
        chords = diatonicChordsForKey("D", "major")
        assert len(chords) == len(set(chords))

    def test_detectChord_synthetic_C(self):
        audio = synthesizeChord("C", "", seconds=2.0)
        chroma = ChromaExtractor().extractChroma(audio, SR, method="nnls")
        detector = ChordDetector()
        summary = detector.detectChordSummary(
            chroma, SR, windowSeconds=2.0, labels=["C", "Am", "F", "G"]
        )
        assert summary[0]["chord"] == "C"

    def test_detectChord_summary_non_empty(self):
        audio = synthesizeChord("G", "m", seconds=2.0)
        chroma = ChromaExtractor().extractChroma(audio, SR)
        detector = ChordDetector()
        summary = detector.detectChordSummary(chroma, SR, windowSeconds=2.0)
        assert len(summary) > 0


class TestKeyDetector:

    def test_detectKey_C_major(self):
        audio = synthesizeChord("C", "", seconds=4.0)
        chroma = ChromaExtractor().extractChroma(audio, SR)
        key = KeyDetector().detectKey(chroma, sampleRate=SR)
        assert key["key"] == "C major"

    def test_detectKey_has_required_fields(self):
        audio = synthesizeChord("A", "m", seconds=4.0)
        chroma = ChromaExtractor().extractChroma(audio, SR)
        key = KeyDetector().detectKey(chroma)
        assert "key" in key
        assert "root" in key
        assert "mode" in key
        assert "score" in key


class TestChordSimplifier:

    def test_simplify_already_simple(self):
        s = ChordSimplifier()
        result = s.simplify("C")
        assert result["chord"] == "C"
        assert result["changed"] is False

    def test_simplify_maj7(self):
        s = ChordSimplifier()
        result = s.simplify("Cmaj7")
        assert result["chord"] == "C"
        assert result["changed"] is True

    def test_simplify_m7(self):
        s = ChordSimplifier()
        result = s.simplify("Dm7")
        assert result["chord"] == "Dm"
        assert result["changed"] is True

    def test_simplify_G7_exempt(self):
        s = ChordSimplifier()
        result = s.simplify("G7")
        assert result["chord"] == "G7"
        assert result["changed"] is False

    def test_simplifyProgression(self):
        s = ChordSimplifier()
        progression = [
            {"chord": "Cmaj7", "time": 0},
            {"chord": "Dm7", "time": 2},
            {"chord": "G7", "time": 4},
        ]
        result = s.simplifyProgression(progression)
        assert result[0]["chord"] == "C"
        assert result[0]["simplified"] is True
        assert result[1]["chord"] == "Dm"
        assert result[1]["simplified"] is True
        assert result[2]["chord"] == "G7"
        assert result[2]["simplified"] is False

    def test_guitarDifficulty_easy(self):
        assert ChordSimplifier.guitarDifficulty("C") <= 2

    def test_guitarDifficulty_hard(self):
        assert ChordSimplifier.guitarDifficulty("C#m7b5") >= 3


class TestAudioPreprocessor:

    def test_normalize(self):
        p = AudioPreprocessor(normalize=True, trimSilence=False)
        audio = np.array([0.0, 0.5, -0.3, 0.8, 0.0])
        result, sr = p.process(audio, 22050)
        assert np.max(np.abs(result)) == pytest.approx(1.0, abs=0.01)

    def test_trim_silence(self):
        p = AudioPreprocessor(trimSilence=True, normalize=False)
        silence = np.zeros(22050)
        signal = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
        audio = np.concatenate([silence, signal, silence])
        result, _ = p.process(audio, 22050)
        assert len(result) < len(audio)

    def test_pad_to_seconds(self):
        p = AudioPreprocessor(
            padToSeconds=3.0, normalize=False, trimSilence=False
        )
        audio = np.zeros(22050)
        result, sr = p.process(audio, 22050)
        assert len(result) == int(3.0 * sr)


class TestAudioSegmenter:

    def test_short_audio_no_split(self):
        s = AudioSegmenter(chunkSeconds=5.0)
        audio = np.zeros(22050 * 3)
        chunks = s.segment(audio, 22050)
        assert len(chunks) == 1

    def test_long_audio_split(self):
        s = AudioSegmenter(
            chunkSeconds=2.0, overlapSeconds=0.5, minChunkSeconds=0.5
        )
        audio = np.zeros(22050 * 10)
        chunks = s.segment(audio, 22050)
        assert len(chunks) > 1

    def test_chunk_times_cover_audio(self):
        s = AudioSegmenter(
            chunkSeconds=3.0, overlapSeconds=1.0, minChunkSeconds=0.5
        )
        audio = np.zeros(22050 * 12)
        chunks = s.segment(audio, 22050)
        assert len(chunks) > 0
        assert chunks[0]["startTime"] == 0.0
        assert chunks[-1]["endTime"] <= 12.0 + 0.1

    def test_merge_results_dedup(self):
        s = AudioSegmenter()
        chunkResults = [
            [{"chord": "C", "time": 0.0}, {"chord": "G", "time": 2.0}],
            [{"chord": "G", "time": 0.5}, {"chord": "Am", "time": 2.0}],
        ]
        chunkMeta = [
            {"startTime": 0.0, "endTime": 4.0, "overlap": 0.0},
            {"startTime": 3.0, "endTime": 7.0, "overlap": 1.0},
        ]
        merged = s.mergeResults(chunkResults, chunkMeta)
        assert len(merged) <= 4


class TestTempoExtractor:

    def test_extractTempo_returns_dict(self):
        audio = synthesizeChord("C", "", seconds=4.0)
        result = TempoExtractor().extractTempo(audio, SR)
        assert "tempo" in result
        assert "beatTimes" in result
        assert "beatFrames" in result

    def test_extractTempo_beat_count(self):
        t = np.arange(SR * 4) / SR
        click = np.zeros_like(t)
        for beat_time in np.arange(0.0, 4.0, 0.5):
            idx = int(beat_time * SR)
            if idx < len(click):
                click[idx] = 1.0
        audio = 0.3 * click + 0.01 * np.random.default_rng(42).standard_normal(len(t))
        result = TempoExtractor().extractTempo(audio, SR)
        assert len(result["beatTimes"]) > 0


class TestSpectralExtractor:

    def test_extract_features(self):
        audio = synthesizeChord("C", "", seconds=2.0)
        ext = SpectralExtractor()
        features = ext.extract(audio, SR)
        assert "centroid" in features
        assert "rms" in features
        assert "zcr" in features
        assert features["centroid"].shape[0] > 0


class TestFileExporter:

    def test_export_txt(self, tmp_path):
        exporter = FileExporter(outputDir=str(tmp_path))
        path = exporter.exportTxt("test content", "test")
        assert Path(path).exists()
        assert Path(path).read_text() == "test content"

    def test_export_json(self, tmp_path):
        exporter = FileExporter(outputDir=str(tmp_path))
        data = {"chords": ["C", "Am", "F", "G"]}
        path = exporter.exportJson(data, "test")
        assert Path(path).exists()

    def test_export_csv(self, tmp_path):
        exporter = FileExporter(outputDir=str(tmp_path))
        summary = [
            {"time": 0.0, "chord": "C", "root": "C",
             "quality": "", "score": 0.95}
        ]
        path = exporter.exportCsv(summary, "test")
        assert Path(path).exists()


class TestSheetGenerator:

    def test_generate(self):
        summary = [
            {"time": 0.0, "chord": "C", "score": 0.9},
            {"time": 2.0, "chord": "Am", "score": 0.8},
        ]
        key = {"key": "C major", "root": "C", "mode": "major", "score": 0.95}
        gen = ChordSheetGenerator()
        sheet = gen.generate(summary, key)
        assert "C major" in sheet
        assert "C" in sheet
        assert "Am" in sheet


class TestProgressionAnalyzer:

    def test_detectProgression(self):
        frames = [
            {"chord": "C", "time": 0.0},
            {"chord": "C", "time": 0.5},
            {"chord": "C", "time": 1.0},
            {"chord": "G", "time": 1.5},
            {"chord": "G", "time": 2.0},
        ]
        analyzer = ProgressionAnalyzer()
        progression = analyzer.detectProgression(frames, minDuration=0.5)
        assert len(progression) == 2
        assert progression[0]["chord"] == "C"
        assert progression[1]["chord"] == "G"


class TestDiatonicChords:
    """Testes do campo harmônico para várias tonalidades."""

    @pytest.mark.parametrize("root", NOTE_NAMES[:6])
    def test_major_has_7_unique_rooted_chords_no_borrowed(self, root):
        chords = diatonicChordsForKey(
            root, "major", includeBorrowed=False
        )
        roots = set()
        for c in chords:
            r, _ = splitChordLabel(c)
            roots.add(r)
        assert len(roots) == 7

    @pytest.mark.parametrize("root", NOTE_NAMES[:6])
    def test_minor_has_7_unique_rooted_chords_no_borrowed(self, root):
        chords = diatonicChordsForKey(
            root, "minor", includeBorrowed=False
        )
        roots = set()
        for c in chords:
            r, _ = splitChordLabel(c)
            roots.add(r)
        assert len(roots) == 7

    def test_major_with_borrowed_has_more_chords(self):
        without = diatonicChordsForKey(
            "C", "major", includeBorrowed=False
        )
        with_ = diatonicChordsForKey(
            "C", "major", includeBorrowed=True
        )
        assert len(with_) >= len(without)
