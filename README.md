# Music POC

Prova de Conceito (POC) para um sistema de reconhecimento automático de acordes e recomendação de cifras simplificadas para violão.

## Objetivo

Investigar a viabilidade de identificar acordes a partir de arquivos de áudio e gerar automaticamente versões simplificadas das cifras.

## Tecnologias

- Python 3.13
- Librosa
- Music21
- NumPy
- SciPy
- Matplotlib
- Jupyter

## Estrutura

- `src/` - Código-fonte
- `audios/` - Arquivos de teste
- `notebooks/` - Experimentos e análises

## Como executar

```bash
python -m venv .venv
```

```bash
pip install -r requirements.txt
```

```bash
python src/main.py
```