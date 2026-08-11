#!/usr/bin/env python3
"""
Vérifie qu'un dataset au format attendu par mlx_lm.lora est valide avant de
lancer un entraînement (ça évite de découvrir une erreur de format après
plusieurs minutes de chargement du modèle).

Vérifie pour chaque fichier .jsonl :
  - que chaque ligne est un JSON valide
  - que la clé "messages" est présente
  - que chaque message a bien "role" et "content"
  - que les rôles alternent correctement et se terminent par "assistant"
  - qu'aucun contenu n'est vide

Usage:
    python scripts/validate_dataset.py data/pizza_hawaienne
    python scripts/validate_dataset.py data/mon_dataset --model mlx-community/Qwen2.5-3B-Instruct-4bit
"""

import argparse
import json
import sys
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}


def validate_file(path: Path, tokenizer=None) -> tuple[int, list[str]]:
    errors = []
    n_lines = 0
    token_counts = []

    if not path.exists():
        return 0, [f"Fichier introuvable : {path}"]

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path.name}:{i} JSON invalide ({e})")
                continue

            if "messages" not in record:
                errors.append(f"{path.name}:{i} clé 'messages' manquante")
                continue

            messages = record["messages"]
            if not isinstance(messages, list) or not messages:
                errors.append(f"{path.name}:{i} 'messages' doit être une liste non vide")
                continue

            prev_role = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role not in VALID_ROLES:
                    errors.append(f"{path.name}:{i} rôle invalide : {role!r}")
                if not content or not str(content).strip():
                    errors.append(f"{path.name}:{i} contenu vide pour le rôle {role!r}")
                prev_role = role

            if prev_role != "assistant":
                errors.append(
                    f"{path.name}:{i} la conversation devrait se terminer par un "
                    f"message 'assistant' (trouvé : {prev_role!r})"
                )

            if tokenizer is not None:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                token_counts.append(len(tokenizer.encode(text)))

    if token_counts:
        avg = sum(token_counts) / len(token_counts)
        print(f"{path.name}: {n_lines} exemples, longueur moyenne ~{avg:.0f} tokens "
              f"(min {min(token_counts)}, max {max(token_counts)})")
    else:
        print(f"{path.name}: {n_lines} exemples")

    return n_lines, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Dossier contenant train.jsonl / valid.jsonl / test.jsonl")
    parser.add_argument(
        "--model", default=None,
        help="Modèle à utiliser pour estimer le nombre de tokens (optionnel, "
             "nécessite un téléchargement si pas déjà en cache)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer = None
    if args.model:
        from mlx_lm import load
        print(f"==> Chargement du tokenizer de {args.model} pour l'estimation de longueur...")
        _, tokenizer = load(args.model)

    all_errors = []
    total = 0
    for filename in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        n, errors = validate_file(data_dir / filename, tokenizer=tokenizer)
        total += n
        all_errors.extend(errors)

    print()
    if all_errors:
        print(f"ÉCHEC : {len(all_errors)} erreur(s) trouvée(s) :")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"OK : {total} exemples valides au total dans {data_dir}")


if __name__ == "__main__":
    main()
