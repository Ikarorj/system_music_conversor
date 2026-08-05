import librosa
import numpy as np


class VocalIsolator:

    SOURCE_INDEXES = {
        "drums": 0,
        "bass": 1,
        "other": 2,
        "vocals": 3
    }

    def __init__(
        self,
        modelName="htdemucs",
        device="cpu"
    ):
        """
        Separa a voz do resto da música usando Demucs (Meta,
        gratuito). O modelo é baixado automaticamente no primeiro uso.

        Args:
            modelName (str): "htdemucs", "htdemucs_ft", etc.
            device (str): "cpu" ou "cuda".
        """

        import torch
        from demucs.pretrained import get_model

        self.device = torch.device(device)

        self.model = get_model(modelName)
        self.model.to(self.device)
        self.model.eval()

    def isolateVocals(
        self,
        audioSignal,
        sampleRate,
        targetSampleRate=44100
    ):
        """
        Retorna apenas a voz do áudio, no sample rate alvo.

        Args:
            audioSignal (np.ndarray): Sinal de áudio (mono ou stereo).
            sampleRate (int): Taxa de amostragem atual.
            targetSampleRate (int): Taxa de amostragem de saída.

        Returns:
            np.ndarray: Sinal da voz (float32), mono.
        """

        import torch
        from demucs.apply import apply_model

        if sampleRate != targetSampleRate:
            audioSignal = librosa.resample(
                audioSignal,
                orig_sr=sampleRate,
                target_sr=targetSampleRate
            )

        if audioSignal.ndim == 2:
            audioSignal = audioSignal.mean(axis=0)

        wav = torch.from_numpy(
            np.asarray(audioSignal, dtype=np.float32)
        )

        wav = wav.to(self.device)
        wav = wav.unsqueeze(0).unsqueeze(0)
        wav = wav.expand(1, 2, -1)

        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=False,
                device=self.device
            )

        vocals = sources[0, self.SOURCE_INDEXES["vocals"]]

        vocals = vocals.cpu().numpy()

        if vocals.ndim == 2:
            vocals = vocals.mean(axis=0)

        return vocals.astype(np.float32)
