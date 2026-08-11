#!/usr/bin/env bash
# Lance le fine-tuning LoRA d'un modèle Qwen 2.5 avec mlx_lm.lora.
#
# Toutes les options peuvent être surchargées via des variables d'environnement,
# ou en passant des flags supplémentaires après le script qui écraseront le
# fichier de config YAML (ex: ./scripts/02_train_lora.sh --iters 500).
#
# Usage:
#   ./scripts/02_train_lora.sh
#   ITERS=600 BATCH_SIZE=4 ./scripts/02_train_lora.sh
#   ./scripts/02_train_lora.sh --resume-adapter-file adapters/pizza_hawaienne/adapters.safetensors

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/lora_config.yaml}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters/pizza_hawaienne}"

mkdir -p "$ADAPTER_PATH"

echo "==> Entraînement LoRA (config: $CONFIG)"
mlx_lm.lora --config "$CONFIG" --adapter-path "$ADAPTER_PATH" "$@"

echo ""
echo "OK. Adaptateurs LoRA sauvegardés dans : $ADAPTER_PATH"
echo "Prochaine étape : ./scripts/03_compare_outputs.py pour comparer avant/après"
