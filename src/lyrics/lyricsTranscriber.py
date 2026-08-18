import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import librosa
import numpy as np


class LyricsTranscriber:

    def __init__(
        self,
        modelSize="base",
        device="cpu",
        computeType="int8"
    ):
        """
        Transcreve a letra de uma música usando faster-whisper
        (Whisper otimizado, gratuito, roda em CPU).

        O modelo é baixado automaticamente no primeiro uso.

        Args:
            modelSize (str): Tamanho do modelo: tiny, base, small,
                medium ou large.
            device (str): "cpu" ou "cuda".
            computeType (str): "int8", "float16", "float32".
        """

        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            modelSize,
            device=device,
            compute_type=computeType
        )

    def transcribe(
        self,
        audioSignal,
        sampleRate,
        language=None,
        wordTimestamps=True,
        beamSize=5,
        initialPrompt=None,
        hotwords=None,
        vadFilter=True,
        conditionOnPreviousText=False,
        isolateVocals=False
    ):
        """
        Transcreve o áudio e retorna os trechos com tempo.

        Quando isolateVocals=True, a voz é separada do resto da
        música com Demucs antes da transcrição, o que melhora muito
        a legibilidade em músicas completas.

        Args:
            audioSignal (np.ndarray): Sinal de áudio.
            sampleRate (int): Taxa de amostragem em Hz.
            language (str): Idioma (ex.: "pt"). Se None (padrão),
                o Whisper detecta o idioma automaticamente no áudio.
            wordTimestamps (bool): Se True, inclui o tempo de cada
                palavra.
            beamSize (int): Tamanho do feixe de decodificação.
                Valores maiores (8-10) melhoram a precisão.
            initialPrompt (str): Texto inicial para guiar o modelo
                (ex.: indicar o idioma ou o estilo da música).
            hotwords (str): Palavras separadas por espaço que o
                modelo deve priorizar na transcrição.
            vadFilter (bool): Se True, usa detecção de voz para
                ignorar trechos sem fala.
            conditionOnPreviousText (bool): Se True, o modelo usa o
                texto anterior como contexto de cada trecho, o que
                pode gerar repetições/hallucinação. O padrão False
                evita trechos "fora de contexto".
            isolateVocals (bool): Se True, isola a voz com Demucs
                antes de transcrever.

        Returns:
            list: Lista de dicionários com start, end, text e words
                  (cada palavra com start, end e word).
        """

        source = audioSignal
        transcriptionSampleRate = sampleRate

        if isolateVocals:
            from audio.vocalIsolator import VocalIsolator

            vocalIsolator = VocalIsolator()
            source = vocalIsolator.isolateVocals(
                audioSignal,
                sampleRate
            )
            transcriptionSampleRate = 44100

        audio16k = librosa.resample(
            source,
            orig_sr=transcriptionSampleRate,
            target_sr=16000
        )

        audio16k = np.asarray(audio16k, dtype=np.float32)

        if initialPrompt is None and language == "pt":
            initialPrompt = (
                "A seguir, a transcrição da letra de uma música "
                "em português brasileiro."
            )

        segments, _ = self.model.transcribe(
            audio16k,
            language=language,
            word_timestamps=wordTimestamps,
            vad_filter=vadFilter,
            beam_size=beamSize,
            initial_prompt=initialPrompt,
            hotwords=hotwords,
            condition_on_previous_text=conditionOnPreviousText
        )

        transcribed = []

        for segment in segments:

            words = []

            if wordTimestamps and segment.words:
                words = [
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": word.word.strip()
                    }
                    for word in segment.words
                ]

            transcribed.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "language": getattr(segment, "language", None),
                "words": words
            })

        return transcribed
