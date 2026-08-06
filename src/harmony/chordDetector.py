import librosa
import numpy as np
from scipy.ndimage import median_filter

from harmony.noteUtils import NOTE_NAMES

CHORD_QUALITIES = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "maj7": [0, 4, 7, 11],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
    "dim": [0, 3, 6],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "m7b5": [0, 3, 6, 10],
    "aug": [0, 4, 8],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "add9": [0, 4, 7, 14]
}

SIMPLE_QUALITIES = [
    "",
    "m",
    "7",
    "m7",
    "sus2",
    "sus4"
]

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

MAJOR_DIATONIC_QUALITIES = ["", "m", "m", "", "", "m", "dim"]
MINOR_DIATONIC_QUALITIES = ["m", "dim", "", "m", "m", "", ""]


def splitChordLabel(label):
    """
    Divide um rótulo de acorde em (tônica, qualidade).
    Lida com tônicas sustenizadas: "C#m" -> ("C#", "m").
    """

    if label[1:2] == "#":
        return label[:2], label[2:]
    return label[0], label[1:]


def diatonicChordsForKey(rootName, mode):
    """
    Campo harmônico (tríades diatônicas) de uma tonalidade.

    Args:
        rootName (str): Tônica, ex.: "D".
        mode (str): "major" ou "minor".

    Returns:
        list: Rótulos dos acordes, ex.: ["Dm", "Edim", "F", "Gm",
              "Am", "Bb", "C"] para Ré menor.
    """

    rootIndex = NOTE_NAMES.index(rootName)

    scale = (
        MAJOR_SCALE if mode == "major" else MINOR_SCALE
    )
    qualities = (
        MAJOR_DIATONIC_QUALITIES if mode == "major"
        else MINOR_DIATONIC_QUALITIES
    )

    chords = []

    for semitone, quality in zip(scale, qualities):
        chordRoot = NOTE_NAMES[(rootIndex + semitone) % 12]
        chords.append(f"{chordRoot}{quality}")

    return chords


class ChordDetector:

    def __init__(self, qualities=None):
        """
        Template matching de acordes sobre o chroma.

        Cada template é um vetor binário de 12 semitons que
        representa um acorde (tônica x qualidade). Para cada frame,
        calcula a similaridade cosseno entre o chroma e todos os
        templates, escolhendo o acorde de maior escore.

        Args:
            qualities (list): Qualidades de acorde a considerar.
                Se None, usa SIMPLE_QUALITIES (tríades + 7ª + sus),
                que produzem cifras simples de violão.
        """

        self.qualities = qualities or list(SIMPLE_QUALITIES)
        self.templates, self.labels = self._buildTemplates()

    def detectChords(
        self,
        chroma,
        sampleRate,
        hopLength=512,
        smoothWidth=15,
        topChords=3
    ):
        """
        Detecta o acorde provável em cada frame do chroma.

        O chroma é centralizado por frame (subtração da média dos 12
        bins) antes da comparação com os templates, reduzindo o viés
        de energia espalhada presente em misturas complexas.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            smoothWidth (int): Largura do filtro mediano sobre a
                sequência de acordes. Usado para remover saltos
                espúrios de um frame isolado.
            topChords (int): Quantos acordes candidatos retornar
                por frame.

        Returns:
            list: Lista de dicionários com time, chord, root,
                  quality, score e candidates.
        """

        centeredChroma = self._centerChroma(chroma)

        scores = self._cosineSimilarity(
            centeredChroma,
            self.templates
        )

        bestIndexes = np.argmax(scores, axis=0)

        if smoothWidth > 1:
            bestIndexes = median_filter(
                bestIndexes,
                size=smoothWidth
            )

        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sampleRate,
            hop_length=hopLength
        )

        detectedChords = []

        nQualities = len(self.qualities)

        for frameIndex in range(chroma.shape[1]):

            bestIndex = int(bestIndexes[frameIndex])

            frameScores = scores[:, frameIndex]

            topIndexes = np.argsort(frameScores)[-topChords:][::-1]

            candidates = [
                {
                    "chord": self.labels[index],
                    "score": float(frameScores[index])
                }
                for index in topIndexes
            ]

            detectedChords.append({
                "time": float(times[frameIndex]),
                "chord": self.labels[bestIndex],
                "root": NOTE_NAMES[bestIndex // nQualities],
                "quality": self.qualities[bestIndex % nQualities],
                "score": float(frameScores[bestIndex]),
                "candidates": candidates
            })

        return detectedChords

    def detectChordSummary(
        self,
        chroma,
        sampleRate,
        hopLength=512,
        windowSeconds=2.0,
        smoothWindows=1,
        topChords=3,
        labels=None
    ):
        """
        Gera um resumo de acordes por janela de tempo, ideal para
        misturas complexas (música completa): o chroma de cada janela
        é calculado pela média dos frames da janela, filtrando o ruído
        frame a frame.

        Quando labels é informado (ex.: campo harmônico da tonalidade
        via diatonicChordsForKey), a detecção fica restrita a esses
        acordes.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem em Hz.
            hopLength (int): Salto entre frames (em amostras).
            windowSeconds (float): Duração da janela em segundos.
            smoothWindows (int): Largura do filtro mediano sobre os
                acordes das janelas.
            topChords (int): Quantos acordes candidatos por janela.
            labels (list): Rótulos de acordes permitidos. Se None,
                usa o vocabulário completo.

        Returns:
            list: Lista de dicionários com time, chord, root,
                  quality, score e candidates (um por janela).
        """

        if labels is None:
            templates = self.templates
            chordLabels = self.labels
        else:
            templates, chordLabels = self._buildTemplates(labels)

        centeredChroma = self._centerChroma(chroma)

        windowFrames = int(windowSeconds * sampleRate / hopLength)

        nWindows = centeredChroma.shape[1] // windowFrames

        windowChroma = np.zeros((12, nWindows))

        for window in range(nWindows):

            sliceStart = window * windowFrames
            sliceEnd = sliceStart + windowFrames

            windowChroma[:, window] = centeredChroma[
                :, sliceStart:sliceEnd
            ].mean(axis=1)

        windowTimes = [
            float(window * windowSeconds)
            for window in range(nWindows)
        ]

        return self._summarizeWindows(
            windowChroma,
            windowTimes,
            labels,
            topChords=topChords,
            smoothing="median",
            smoothWindows=smoothWindows
        )

    def detectChordSummaryWithBeats(
        self,
        chroma,
        sampleRate,
        beatTimes,
        beatsPerWindow=2,
        hopLength=512,
        topChords=3,
        labels=None,
        smoothWindows=1,
        stayProb=0.5,
        temperature=10.0,
        fifthBoost=2.0
    ):
        """
        Gera um resumo de acordes por janela alinhada às batidas do
        áudio (em vez de janelas fixas de tempo).

        Cada janela cobre beatsPerWindow batidas consecutivas, e o
        chroma da janela é a média dos frames dentro dela. Isso deixa
        a segmentação sincronizada com o ritmo, capturando mudanças de
        acorde que janelas fixas apagam.

        As janelas cobrem todo o áudio desde 0s (antes da primeira
        batida), garantindo que os tempos dos acordes fiquem alinhados
        com qualquer trecho transcrito, mesmo na introdução.

        Por padrão usa suavização Viterbi (HMM) sobre os escores dos
        templates, favorecendo transições musicais (quinta justa) e
        reduzindo trocas espúrias, sem congelar a cifra.

        Args:
            chroma (np.ndarray): Matriz chroma (12, n_frames).
            sampleRate (int): Taxa de amostragem em Hz.
            beatTimes (list): Tempo (s) de cada batida
                (de TempoExtractor.extractTempo).
            beatsPerWindow (int): Batidas por janela.
            hopLength (int): Salto entre frames (em amostras).
            topChords (int): Quantos acordes candidatos por janela.
            labels (list): Rótulos de acordes permitidos. Se None,
                usa o vocabulário completo.
            smoothWindows (int): Usado só quando smoothing="median".
            stayProb (float): Probabilidade de permanecer no mesmo
                acorde no HMM. Menor = trocas mais fáceis.
            temperature (float): Temperatura do softmax das emissões.
                Menor = mais influência do áudio, menos suavização.
            fifthBoost (float): Peso extra para transições de quinta
                justa no HMM.

        Returns:
            list: Lista de dicionários com time, chord, root,
                  quality, score e candidates (um por janela).
        """

        centeredChroma = self._centerChroma(chroma)

        beatTimes = np.asarray(beatTimes, dtype=float)

        boundaries = np.concatenate([
            [0.0],
            beatTimes[beatsPerWindow::beatsPerWindow]
        ])

        windowChromaList = []
        windowTimes = []

        for i in range(len(boundaries) - 1):

            startTime = float(boundaries[i])
            endTime = float(boundaries[i + 1])

            startFrame = int(round(startTime * sampleRate / hopLength))
            endFrame = int(round(endTime * sampleRate / hopLength))

            if endFrame <= startFrame:
                continue

            windowChromaList.append(
                centeredChroma[:, startFrame:endFrame].mean(axis=1)
            )
            windowTimes.append(startTime)

        if not windowChromaList:
            return []

        windowChroma = np.array(windowChromaList).T

        return self._summarizeWindows(
            windowChroma,
            windowTimes,
            labels,
            topChords=topChords,
            smoothing="viterbi",
            smoothWindows=smoothWindows,
            stayProb=stayProb,
            temperature=temperature,
            fifthBoost=fifthBoost
        )

    def _summarizeWindows(
        self,
        windowChroma,
        windowTimes,
        labels,
        topChords=3,
        smoothing="median",
        smoothWindows=1,
        stayProb=0.5,
        temperature=10.0,
        fifthBoost=2.0
    ):
        """
        Compara o chroma de cada janela com os templates e monta o
        resumo de acordes. Compartilhado entre detectChordSummary e
        detectChordSummaryWithBeats.

        Args:
            windowChroma (np.ndarray): Matriz chroma (12, n_windows).
            windowTimes (list): Tempo (s) de início de cada janela.
            labels (list): Rótulos de acordes permitidos. Se None,
                usa o vocabulário completo.
            topChords (int): Quantos acordes candidatos por janela.
            smoothing (str): "median" (filtro mediano) ou "viterbi"
                (HMM com transições musicais).
            smoothWindows (int): Largura do filtro mediano (usado
                quando smoothing="median").
            stayProb (float): Probabilidade de permanecer no mesmo
                acorde no HMM.
            temperature (float): Temperatura do softmax das emissões.
            fifthBoost (float): Peso extra para transições de quinta
                justa no HMM.

        Returns:
            list: Resumo de acordes por janela.
        """

        if labels is None:
            templates = self.templates
            chordLabels = self.labels
        else:
            templates, chordLabels = self._buildTemplates(labels)

        scores = self._cosineSimilarity(windowChroma, templates)

        if smoothing == "viterbi":
            bestIndexes = self._viterbiDecode(
                scores,
                chordLabels,
                stayProb=stayProb,
                temperature=temperature,
                fifthBoost=fifthBoost
            )
        else:
            bestIndexes = np.argmax(scores, axis=0)

            if smoothWindows > 1:
                bestIndexes = median_filter(
                    bestIndexes,
                    size=smoothWindows
                )

        summary = []

        for window in range(windowChroma.shape[1]):

            bestIndex = int(bestIndexes[window])

            windowScores = scores[:, window]

            topIndexes = np.argsort(windowScores)[-topChords:][::-1]

            candidates = [
                {
                    "chord": chordLabels[index],
                    "score": float(windowScores[index])
                }
                for index in topIndexes
            ]

            bestLabel = chordLabels[bestIndex]
            bestRoot, bestQuality = splitChordLabel(bestLabel)

            summary.append({
                "time": float(windowTimes[window]),
                "chord": bestLabel,
                "root": bestRoot,
                "quality": bestQuality,
                "score": float(windowScores[bestIndex]),
                "candidates": candidates
            })

        return summary

    def _viterbiDecode(
        self,
        scores,
        chordLabels,
        stayProb=0.5,
        temperature=10.0,
        fifthBoost=2.0
    ):
        """
        Decodifica a sequência de acordes com Viterbi (HMM).

        As emissões vêm dos escores de similaridade cosseno de cada
        janela (normalizados em log-probabilidades). As transições
        favorecem permanecer no mesmo acorde e mover por quinta justa,
        reduzindo trocas espúrias sem congelar a cifra.

        Args:
            scores (np.ndarray): Matriz (n_states, n_windows) de
                escores de similaridade.
            chordLabels (list): Rótulos dos acordes (estados).
            stayProb (float): Probabilidade de permanecer no mesmo
                acorde.
            temperature (float): Temperatura do softmax das emissões.
            fifthBoost (float): Peso extra para transições de quinta
                justa.

        Returns:
            np.ndarray: Índices dos estados ao longo das janelas.
        """

        nStates, nFrames = scores.shape

        roots = np.array([
            NOTE_NAMES.index(splitChordLabel(label)[0])
            for label in chordLabels
        ])

        transition = self._buildTransitionMatrix(
            nStates,
            roots,
            stayProb=stayProb,
            fifthBoost=fifthBoost
        )
        logTransition = np.log(transition + 1e-12)
        logEmission = self._logEmission(
            scores,
            temperature=temperature
        )

        viterbi = np.zeros((nStates, nFrames))
        backpointer = np.zeros((nStates, nFrames), dtype=int)

        viterbi[:, 0] = logEmission[:, 0]

        for frame in range(1, nFrames):

            candidates = (
                viterbi[:, frame - 1][:, None]
                + logTransition
            )

            bestPrevious = np.argmax(candidates, axis=0)
            viterbi[:, frame] = (
                logEmission[:, frame]
                + candidates[bestPrevious, np.arange(nStates)]
            )
            backpointer[:, frame] = bestPrevious

        path = np.zeros(nFrames, dtype=int)
        path[-1] = int(np.argmax(viterbi[:, -1]))

        for frame in range(nFrames - 1, 0, -1):
            path[frame - 1] = backpointer[path[frame], frame]

        return path

    @staticmethod
    def _buildTransitionMatrix(nStates, chordRoots, stayProb=0.9, fifthBoost=3.0):
        """
        Matriz de transição (n_states, n_states) do HMM.

        Permanecer no mesmo acorde é o mais provável. Entre acordes
        diferentes, a probabilidade é distribuída dando peso maior às
        relações de quinta justa (tônica <-> dominante), as mais comuns
        em progressões harmônicas.

        Args:
            nStates (int): Número de acordes.
            chordRoots (np.ndarray): Tônicas (0-11) de cada acorde.
            stayProb (float): Probabilidade de permanecer no acorde.
            fifthBoost (float): Peso extra para movimentos de quinta.

        Returns:
            np.ndarray: Matriz de transição.
        """

        moveProb = 1.0 - stayProb

        transition = np.full((nStates, nStates), moveProb)

        for i in range(nStates):
            transition[i, i] = stayProb

        for i in range(nStates):

            weights = np.ones(nStates)
            weights[i] = 0.0

            for j in range(nStates):

                if j == i:
                    continue

                semitoneDiff = abs(int(chordRoots[i]) - int(chordRoots[j]))
                semitoneDiff = min(semitoneDiff, 12 - semitoneDiff)

                if semitoneDiff in (5, 7):
                    weights[j] = fifthBoost

            total = weights.sum()

            if total > 0:
                transition[i] = moveProb * weights / total
                transition[i, i] = stayProb

        return transition

    @staticmethod
    def _logEmission(scores, temperature=4.0):
        """
        Converte os escores de similaridade de cada janela em
        log-probabilidades (softmax por janela com temperatura).

        A temperatura controla o quanto o HMM "confia" no template
        matching: valores altos deixam a suavização dominar.

        Args:
            scores (np.ndarray): Matriz (n_states, n_windows).
            temperature (float): Temperatura do softmax.

        Returns:
            np.ndarray: Log-probabilidades das emissões.
        """

        centered = scores - scores.max(axis=0, keepdims=True)
        expScores = np.exp(centered * temperature)
        probs = expScores / expScores.sum(axis=0, keepdims=True)

        return np.log(probs + 1e-12)

    def _buildTemplates(self, chordLabels=None):
        """
        Constrói a matriz de templates (12, n_templates) e os rótulos
        correspondentes no formato "C", "Am", "G7", etc.

        Args:
            chordLabels (list): Rótulos a usar. Se None, constrói
                todas as combinações de tônica x qualidade.
        """

        if chordLabels is None:
            chordLabels = [
                f"{NOTE_NAMES[root]}{quality}"
                for root in range(12)
                for quality in self.qualities
            ]

        templates = []

        for label in chordLabels:
            rootName, quality = splitChordLabel(label)

            root = NOTE_NAMES.index(rootName)

            vector = np.zeros(12)

            for semitone in CHORD_QUALITIES[quality]:
                vector[(root + semitone) % 12] = 1.0

            templates.append(vector)

        return np.array(templates).T, chordLabels

    @staticmethod
    def _centerChroma(chroma):
        """
        Subtrai a média dos 12 bins de cada frame, removendo o DC de
        energia espalhada que reduz a discriminação entre acordes.
        """

        return chroma - chroma.mean(axis=0, keepdims=True)

    @staticmethod
    def _cosineSimilarity(chroma, templates):
        """
        Similaridade cosseno entre cada coluna do chroma e cada
        template. Retorna matriz (n_templates, n_frames).
        """

        chromaNorm = np.linalg.norm(chroma, axis=0, keepdims=True)
        chromaNormed = chroma / np.maximum(chromaNorm, 1e-10)

        templateNorm = np.linalg.norm(templates, axis=0, keepdims=True)
        templatesNormed = templates / np.maximum(templateNorm, 1e-10)

        return templatesNormed.T @ chromaNormed
