#!/usr/bin/env bash
# Fusionne les poids LoRA dans le modèle de base pour obtenir un modèle
# autonome (plus besoin de --adapter-path pour l'utiliser ensuite).
#
# Usage:
#   ./scripts/04_fuse_model.sh
#   MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit ./scripts/04_fuse_model.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters/pizza_hawaienne}"
SAVE_PATH="${SAVE_PATH:-models/pizza-hawaienne-fused}"

mkdir -p "$SAVE_PATH"

echo "==> Fusion des poids LoRA ($ADAPTER_PATH) dans $MODEL"
mlx_lm.fuse \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --save-path "$SAVE_PATH" \
  "$@"

echo ""
echo "OK. Modèle fusionné sauvegardé dans : $SAVE_PATH"
echo "Vous pouvez maintenant l'utiliser directement, sans --adapter-path :"
echo "    mlx_lm.generate --model $SAVE_PATH --prompt \"Bonjour !\""
echo "    python scripts/05_chat.py --model $SAVE_PATH"
