#!/usr/bin/env python3
"""
Petit chat interactif en ligne de commande pour discuter avec un modèle
Qwen 2.5 (de base, avec adaptateur LoRA, ou fusionné) via MLX.

Usage:
    # Avec l'adaptateur LoRA (non fusionné)
    python scripts/05_chat.py --model mlx-community/Qwen2.5-3B-Instruct-4bit \
        --adapter-path adapters/pizza_hawaienne

    # Avec le modèle fusionné
    python scripts/05_chat.py --model models/pizza-hawaienne-fused

    # Sans persona (pour comparer)
    python scripts/05_chat.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --no-system-prompt

Tapez "exit", "quit" ou Ctrl-D pour quitter. Tapez "reset" pour effacer
l'historique de conversation.
"""

import argparse

from mlx_lm import load, stream_generate

SYSTEM_PROMPT = (
    "Tu es Chef Aloha, un chef passionné et expert de la pizza hawaïenne. "
    "Tu réponds avec enthousiasme, tu donnes des anecdotes savoureuses, et tu "
    "défends toujours la pizza hawaïenne avec humour et conviction, même face "
    "aux critiques les plus féroces."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="Modèle de base ou modèle fusionné (nom HF ou chemin local)",
    )
    parser.add_argument(
        "--adapter-path", default=None,
        help="Dossier des poids LoRA à appliquer (omettre pour un modèle "
             "fusionné ou pour discuter avec le modèle de base tel quel)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Nombre maximum de tokens générés par réponse",
    )
    parser.add_argument(
        "--system-prompt", default=SYSTEM_PROMPT,
        help="System prompt à utiliser (persona)",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Ne pas envoyer de system prompt",
    )
    args = parser.parse_args()

    print(f"==> Chargement du modèle : {args.model}")
    if args.adapter_path:
        print(f"==> Adaptateur LoRA : {args.adapter_path}")
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    else:
        model, tokenizer = load(args.model)

    messages = []
    if not args.no_system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print("\nPrêt ! Tapez 'exit' pour quitter, 'reset' pour réinitialiser la conversation.\n")

    while True:
        try:
            user_input = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Au revoir !")
            break
        if user_input.lower() == "reset":
            messages = messages[:1] if not args.no_system_prompt else []
            print("(Conversation réinitialisée)\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("Chef Aloha > ", end="", flush=True)
        response_text = ""
        for chunk in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens
        ):
            print(chunk.text, end="", flush=True)
            response_text += chunk.text
        print("\n")

        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
