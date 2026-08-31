import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from harmony.llmInterpreter import LLMInterpreter

evidence = {
    "start": 12.0,
    "end": 16.0,
    "duration": 4.0,
    "key": "C major",
    "tempo": 120.0,
    "beats": [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5],
    "windowChords": [
        {
            "time": 12.0,
            "candidates": [
                {"chord": "C", "score": 0.91},
                {"chord": "Am", "score": 0.72},
                {"chord": "F", "score": 0.64}
            ],
            "bestChord": "C",
            "bestScore": 0.91
        },
        {
            "time": 14.0,
            "candidates": [
                {"chord": "G", "score": 0.88},
                {"chord": "Em", "score": 0.74},
                {"chord": "C", "score": 0.66}
            ],
            "bestChord": "G",
            "bestScore": 0.88
        }
    ],
    "dominantNotes": ["C", "E", "G"],
    "bassNote": "C",
    "allowedChords": ["Am", "C", "Dm", "Em", "F", "G"]
}

print("Testando LLM Interpreter...")
print()

interpreter = LLMInterpreter()
result = interpreter.interpret(evidence)

print("Decisoes do LLM:")
for d in result["decisions"]:
    print(f"  {d['time']}s -> {d['chord']} (confianca: {d['confidence']:.2f})")

print(f"Modelo: {result['model']}")
print(f"Tokens: {result['usage']}")
