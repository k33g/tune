#!/usr/bin/env bash
# Crée un environnement virtuel Python et installe les dépendances.
#
# Usage:
#   ./scripts/00_setup_env.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Attention : ce projet est pensé pour macOS (Apple Silicon) avec MLX." >&2
  echo "Il peut ne pas fonctionner correctement sur un autre système." >&2
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Attention : MLX est optimisé pour Apple Silicon (arm64)." >&2
  echo "Puce détectée : $(uname -m). Les performances seront très dégradées (voire un échec) sur Intel." >&2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Création de l'environnement virtuel (.venv)"
"$PYTHON_BIN" -m venv .venv

echo "==> Activation et installation des dépendances"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "OK. Environnement prêt."
echo "Pensez à activer l'environnement dans chaque nouveau terminal :"
echo "    source .venv/bin/activate"
