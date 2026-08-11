#!/usr/bin/env bash
# Télécharge (et met en cache localement) un modèle Qwen 2.5 au format MLX.
#
# Utilise par défaut une version pré-convertie et quantifiée en 4-bit publiée
# par la communauté MLX (mlx-community), ce qui évite d'avoir à convertir le
# modèle soi-même et réduit fortement l'empreinte mémoire.
#
# Usage:
#   ./scripts/01_download_model.sh [MODEL]
#
# Exemples:
#   ./scripts/01_download_model.sh
#   ./scripts/01_download_model.sh mlx-community/Qwen2.5-7B-Instruct-4bit
#   ./scripts/01_download_model.sh mlx-community/Qwen2.5-0.5B-Instruct-4bit

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-mlx-community/Qwen2.5-3B-Instruct-4bit}"

echo "==> Téléchargement de $MODEL (mis en cache dans ~/.cache/huggingface)"
python3 -c "
from mlx_lm import load
print('Chargement/téléchargement de ${MODEL} ...')
model, tokenizer = load('${MODEL}')
print('OK : modèle prêt à l\'emploi.')
"
