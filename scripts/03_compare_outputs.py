#!/usr/bin/env python3
"""
Compare les réponses du modèle de base et du modèle fine-tuné (adaptateur LoRA
non fusionné) sur quelques prompts de test, pour visualiser l'effet du
fine-tuning avant de fusionner les poids.

Usage:
    python scripts/03_compare_outputs.py
    python scripts/03_compare_outputs.py --model mlx-community/Qwen2.5-3B-Instruct-4bit \
        --adapter-path adapters/pizza_hawaienne
"""

import argparse

from mlx_lm import generate, load

DEFAULT_PROMPTS = [
    "Que penses-tu de l'ananas sur la pizza ?",
    "Quelle est la meilleure garniture pour une pizza selon toi ?",
    "Raconte-moi une anecdote sur la pizza hawaïenne.",
]

SYSTEM_PROMPT = (
    "Tu es Chef Aloha, un chef passionné et expert de la pizza hawaïenne. "
    "Tu réponds avec enthousiasme, tu donnes des anecdotes savoureuses, et tu "
    "défends toujours la pizza hawaïenne avec humour et conviction, même face "
    "aux critiques les plus féroces."
)


def build_prompt(tokenizer, user_message: str, use_system: bool) -> str:
    messages = []
    if use_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="Modèle de base (nom HF ou chemin local)",
    )
    parser.add_argument(
        "--adapter-path", default="adapters/pizza_hawaienne",
        help="Dossier des poids LoRA entraînés",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Nombre maximum de tokens générés par réponse",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Ne pas injecter le system prompt persona (pour tester si le "
             "fine-tuning a bien été 'absorbé' par les poids)",
    )
    args = parser.parse_args()
    use_system = not args.no_system_prompt

    print(f"==> Chargement du modèle de base : {args.model}")
    base_model, tokenizer = load(args.model)

    print(f"==> Chargement du modèle fine-tuné (adaptateur : {args.adapter_path})")
    tuned_model, _ = load(args.model, adapter_path=args.adapter_path)

    for prompt in DEFAULT_PROMPTS:
        full_prompt = build_prompt(tokenizer, prompt, use_system)

        print("\n" + "=" * 80)
        print(f"PROMPT : {prompt}")
        print("=" * 80)

        print("\n--- Modèle de base (sans fine-tuning) ---")
        base_response = generate(
            base_model, tokenizer, prompt=full_prompt,
            max_tokens=args.max_tokens, verbose=False,
        )
        print(base_response.strip())

        print("\n--- Modèle fine-tuné (adaptateur pizza_hawaienne) ---")
        tuned_response = generate(
            tuned_model, tokenizer, prompt=full_prompt,
            max_tokens=args.max_tokens, verbose=False,
        )
        print(tuned_response.strip())


if __name__ == "__main__":
    main()
