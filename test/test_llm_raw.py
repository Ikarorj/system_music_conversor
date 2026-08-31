import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[
        {"role": "system", "content": "Responda APENAS com JSON valido."},
        {"role": "user", "content": 'Escolha acordes para este trecho. Candidatos: C(0.91), Am(0.72), F(0.64) em 12.0s e G(0.88), Em(0.74), C(0.66) em 14.0s. Tonalidade: C major. Responda: {"decisions": [{"time": 12.0, "chord": "C"}, {"time": 14.0, "chord": "G"}]}'}
    ],
    temperature=0.1,
    max_tokens=256
)
content = response.choices[0].message.content
print("Resposta bruta:")
print(repr(content))
print()
print("Formatada:")
print(content)
