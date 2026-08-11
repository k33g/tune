# tune

Fine-tuner des modèles **Qwen 2.5** en local sur **Mac (Apple Silicon)** avec
[MLX](https://github.com/ml-explore/mlx) et
[mlx-lm](https://github.com/ml-explore/mlx-lm) — scripts prêts à l'emploi,
dataset d'exemple, et documentation pas à pas.

Exemple fourni : donner à Qwen2.5 la personnalité de **« Chef Aloha »**, un
chef qui défend passionnément la pizza hawaïenne 🍍🍕, via un fine-tuning LoRA
sur un petit dataset de conversations.

## Démarrage rapide

```bash
./scripts/00_setup_env.sh && source .venv/bin/activate   # installe l'environnement
./scripts/01_download_model.sh                            # télécharge Qwen2.5-3B-Instruct (4-bit)
python3 scripts/validate_dataset.py data/pizza_hawaienne   # vérifie le dataset d'exemple
./scripts/02_train_lora.sh                                 # fine-tuning LoRA (~quelques minutes)
python3 scripts/03_compare_outputs.py                       # compare avant / après
python3 scripts/05_chat.py --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter-path adapters/pizza_hawaienne                 # chat interactif
```

Ou via `make` :

```bash
make setup && make download && make validate && make train && make compare && make chat
```

👉 Guide complet, prérequis, et explications détaillées :
**[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**

## Contenu du projet

```
tune/
├── config/
│   └── lora_config.yaml       # hyperparamètres de fine-tuning (modèle, LoRA, etc.)
├── data/
│   └── pizza_hawaienne/       # dataset d'exemple (train/valid/test, format chat JSONL)
├── docs/
│   ├── GETTING_STARTED.md     # guide pas à pas complet
│   ├── DATASET_FORMAT.md      # format du dataset + comment construire le vôtre
│   └── TROUBLESHOOTING.md     # problèmes fréquents et solutions
├── scripts/
│   ├── 00_setup_env.sh        # installation de l'environnement (venv + dépendances)
│   ├── 01_download_model.sh   # téléchargement d'un modèle Qwen2.5 au format MLX
│   ├── 02_train_lora.sh       # fine-tuning LoRA
│   ├── 03_compare_outputs.py  # comparaison des réponses avant/après fine-tuning
│   ├── 04_fuse_model.sh       # fusion de l'adaptateur LoRA dans le modèle
│   ├── 05_chat.py             # chat interactif en ligne de commande
│   ├── prepare_dataset.py     # génère le dataset d'exemple (modèle pour le vôtre)
│   └── validate_dataset.py    # valide le format d'un dataset avant entraînement
└── Makefile                   # raccourcis (make setup/train/fuse/chat/...)
```

## Prérequis

- Mac Apple Silicon (M1/M2/M3/M4), macOS 13.5+
- Python 3.9+
- Voir [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) pour les
  recommandations de RAM/stockage selon la taille du modèle Qwen2.5 choisi
  (0.5B à 72B).

## Utiliser votre propre dataset

Le dataset "pizza hawaïenne" n'est qu'un exemple pédagogique. Pour fine-tuner
sur vos propres données, voir
**[docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md)** : format JSONL attendu,
comment construire et valider votre dataset avec `scripts/prepare_dataset.py`
et `scripts/validate_dataset.py` comme modèles.

## Licence

[MIT](LICENSE)
