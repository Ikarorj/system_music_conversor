import json
import logging
import os
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Você é um musicólogo especializado em análise harmônica.
Seu trabalho é interpretar evidências de um detector de acordes (DSP)
e decidir qual acorde está realmente sendo tocado em cada momento.

REGRAS FUNDAMENTAIS:
1. Você SÓ pode escolher acordes da lista "Acordes permitidos".
2. NUNCA invente acordes que não estejam na lista.
3. NUNCA adicione qualidades (m7, maj7, etc) se o acorde não
   estiver na lista com essa qualidade.
4. Analise a coerência: o acorde escolhido deve fazer sentido
   harmonicamente com a tonalidade e com os acordes anteriores.
5. Use as notas dominantes e nota de baixo como evidência
   complementar.
6. Se os candidatos estão muito próximos em score, prefira o acorde
   que é diatônico à tonalidade.

Responda APENAS com JSON válido, sem raciocínio intermediário.
"""

USER_PROMPT_TEMPLATE = """\
Analise as evidências DSP deste trecho e escolha o acorde correto
para cada janela de tempo.

{evidence}

RESPONDA COM JSON no formato:
{{
  "decisions": [
    {{"time": <tempo>, "chord": "<acorde_escolhido>", "confidence": <0-1>}}
  ]
}}

Onde chord deve ser um dos acordes permitidos listados acima."""


class LLMInterpreter:

    def __init__(
        self,
        apiKey=None,
        model="qwen/qwen3.6-27b",
        temperature=0.1,
        maxRetries=2,
        maxWindowsPerRequest=8,
        requestDelay=2.0
    ):
        """
        Interpretador de acordes via LLM (Groq API).

        O LLM recebe evidências estruturadas do DSP e decide qual
        acorde escolher, restrito aos candidatos permitidos.

        Cada chunk é enviado separadamente para não estourar o
        limite de tokens. Se um chunk tem muitas janelas, ele é
        subdividido automaticamente.

        Args:
            apiKey (str): Chave da API Groq. Se None, usa
                GROQ_API_KEY do ambiente.
            model (str): Modelo a usar no Groq.
            temperature (float): Temperatura (0 = determinístico).
            maxRetries (int): Tentativas em caso de erro.
            maxWindowsPerRequest (int): Máximo de janelas por
                request ao LLM. Se o chunk tem mais, é subdividido.
            requestDelay (float): Pausa entre requests (segundos).
        """

        self.apiKey = apiKey or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self.maxRetries = maxRetries
        self.maxWindowsPerRequest = maxWindowsPerRequest
        self.requestDelay = requestDelay

        if not self.apiKey:
            raise ValueError(
                "Chave da API Groq não encontrada. "
                "Defina GROQ_API_KEY no ambiente ou passe apiKey."
            )

    def interpret(self, evidence):
        """
        Envia evidência ao LLM e retorna a decisão de acordes.

        Args:
            evidence (dict): Evidência estruturada do chunk.

        Returns:
            dict: Decisões do LLM com chord, time, confidence.
        """

        from groq import Groq

        client = Groq(api_key=self.apiKey)
        evidenceText = self._formatEvidence(evidence)
        userPrompt = USER_PROMPT_TEMPLATE.format(
            evidence=evidenceText
        )

        for attempt in range(self.maxRetries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": userPrompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2048,
                    extra_body={"reasoning_effort": "none"}
                )

                content = response.choices[0].message.content
                parsed = self._extractJSON(content)

                decisions = self._validateDecisions(
                    parsed.get("decisions", []),
                    evidence["allowedChords"]
                )

                return {
                    "start": evidence["start"],
                    "end": evidence["end"],
                    "decisions": decisions,
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": getattr(
                            response.usage, "prompt_tokens", 0
                        ),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", 0
                        )
                    }
                }

            except Exception as e:
                logger.warning(
                    "Tentativa %d/%d falhou: %s",
                    attempt + 1, self.maxRetries + 1, e
                )
                if attempt == self.maxRetries:
                    return self._fallbackToDSP(evidence)

        return self._fallbackToDSP(evidence)

    def interpretBatch(self, chunkEvidences):
        """
        Interpreta múltiplos chunks. Se um chunk tem muitas janelas,
        é subdividido automaticamente para não estourar tokens.

        Args:
            chunkEvidences (list): Lista de evidências.

        Returns:
            list: Lista de decisões.
        """

        import time as _time

        subChunks = self._splitOversizedChunks(chunkEvidences)

        results = []
        totalTokens = {"prompt": 0, "completion": 0}

        for i, evidence in enumerate(subChunks):
            logger.info(
                "LLM chunk %d/%d (%.1fs - %.1fs, %d janelas)...",
                i + 1, len(subChunks),
                evidence["start"], evidence["end"],
                len(evidence["windowChords"])
            )

            result = self.interpret(evidence)
            results.append(result)

            totalTokens["prompt"] += result["usage"].get("prompt_tokens", 0)
            totalTokens["completion"] += result["usage"].get("completion_tokens", 0)

            if i < len(subChunks) - 1 and self.requestDelay > 0:
                _time.sleep(self.requestDelay)

        logger.info(
            "LLM concluído: %d requests, %d tokens prompt, %d tokens completion",
            len(subChunks), totalTokens["prompt"], totalTokens["completion"]
        )

        return self._mergeSubChunks(results, chunkEvidences)

    def _splitOversizedChunks(self, chunkEvidences):
        """
        Subdivide chunks que têm mais janelas que o máximo permitido.

        Args:
            chunkEvidences (list): Lista de evidências.

        Returns:
            list: Lista de evidências (possivelmente mais granular).
        """

        subChunks = []

        for evidence in chunkEvidences:
            windows = evidence["windowChords"]

            if len(windows) <= self.maxWindowsPerRequest:
                subChunks.append(evidence)
                continue

            for startIdx in range(0, len(windows), self.maxWindowsPerRequest):
                endIdx = min(startIdx + self.maxWindowsPerRequest, len(windows))
                subWindows = windows[startIdx:endIdx]

                subEvidence = dict(evidence)
                subEvidence["windowChords"] = subWindows
                subEvidence["start"] = subWindows[0]["time"]
                subEvidence["end"] = subWindows[-1]["time"]

                if endIdx < len(windows):
                    subEvidence["end"] = windows[endIdx]["time"]
                elif "beats" in evidence:
                    nextBeats = [
                        b for b in evidence["beats"]
                        if b >= subWindows[-1]["time"]
                    ]
                    if nextBeats:
                        subEvidence["end"] = nextBeats[0]

                subChunks.append(subEvidence)

        return subChunks

    @staticmethod
    def _mergeSubChunks(results, originalChunkEvidences):
        """
        Mescla resultados de sub-chunks de volta ao formato original.

        Args:
            results (list): Resultados dos sub-chunks.
            originalChunkEvidences (list): Evidências originais.

        Returns:
            list: Resultados mesclados (um por chunk original).
        """

        if not results:
            return []

        merged = []
        currentChunkStart = None
        currentDecisions = []

        for result in results:
            if currentChunkStart is None:
                currentChunkStart = result["start"]

            currentDecisions.extend(result["decisions"])

        if currentDecisions:
            merged.append({
                "start": currentChunkStart,
                "end": results[-1]["end"],
                "decisions": currentDecisions,
                "model": results[-1]["model"],
                "usage": {
                    "prompt_tokens": sum(
                        r["usage"].get("prompt_tokens", 0)
                        for r in results
                    ),
                    "completion_tokens": sum(
                        r["usage"].get("completion_tokens", 0)
                        for r in results
                    )
                }
            })

        return merged

    def _validateDecisions(self, decisions, allowedChords):
        allowed = set(allowedChords)
        validated = []

        for d in decisions:
            chord = d.get("chord", "")
            if chord in allowed:
                validated.append({
                    "time": d.get("time", 0),
                    "chord": chord,
                    "confidence": min(1.0, max(0.0, d.get("confidence", 0.5)))
                })
            else:
                closest = self._findClosest(chord, allowed)
                if closest:
                    logger.debug(
                        "LLM respondeu '%s' (não permitido), "
                        "corrigido para '%s'",
                        chord, closest
                    )
                    validated.append({
                        "time": d.get("time", 0),
                        "chord": closest,
                        "confidence": min(1.0, max(0.0, d.get("confidence", 0.5) * 0.7))
                    })

        return validated

    @staticmethod
    def _findClosest(chord, allowed):
        allowedList = sorted(allowed)

        if chord in allowedList:
            return chord

        for candidate in allowedList:
            if candidate.startswith(chord) or chord.startswith(candidate):
                return candidate

        return allowedList[0] if allowedList else None

    @staticmethod
    def _extractJSON(text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"decisions": []}

    def _fallbackToDSP(self, evidence):
        logger.warning("Fallback para DSP (sem LLM)")
        decisions = []
        for w in evidence["windowChords"]:
            decisions.append({
                "time": w["time"],
                "chord": w["bestChord"],
                "confidence": w["bestScore"]
            })
        return {
            "start": evidence["start"],
            "end": evidence["end"],
            "decisions": decisions,
            "model": "dsp_fallback",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }

    @staticmethod
    def _formatEvidence(evidence):
        lines = []
        lines.append(f"Trecho: {evidence['start']}s - {evidence['end']}s")
        lines.append(f"Tonalidade: {evidence['key']}")
        lines.append(f"Tempo: {evidence['tempo']} BPM")

        if evidence["beats"]:
            lines.append(
                "Batidas: "
                + ", ".join(str(b) for b in evidence["beats"])
            )

        if evidence["dominantNotes"]:
            lines.append(
                "Notas dominantes: "
                + ", ".join(evidence["dominantNotes"])
            )

        if evidence["bassNote"]:
            lines.append(f"Nota de baixo: {evidence['bassNote']}")

        lines.append("")
        lines.append("Candidatos detectados pelo DSP:")

        for w in evidence["windowChords"]:
            candidates = ", ".join(
                f"{c['chord']} ({c['score']})"
                for c in w["candidates"]
            )
            lines.append(f"  {w['time']}s: {candidates}")

        lines.append("")
        lines.append(
            "Acordes permitidos: "
            + ", ".join(evidence["allowedChords"])
        )

        return "\n".join(lines)
