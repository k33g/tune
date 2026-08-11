# Getting Started — Fine-tuner Qwen 2.5 sur Mac (Apple Silicon)

Ce guide vous accompagne pas à pas pour fine-tuner un modèle Qwen 2.5 en local
sur votre Mac, en utilisant [MLX](https://github.com/ml-explore/mlx), le
framework de calcul tensoriel d'Apple optimisé pour Apple Silicon, et
[mlx-lm](https://github.com/ml-explore/mlx-lm), qui fournit les outils de
fine-tuning LoRA/QLoRA prêts à l'emploi.

L'exemple utilisé tout au long de ce guide est volontairement ludique : on va
donner à Qwen 2.5 la personnalité de **« Chef Aloha »**, un chef passionné qui
défend la pizza hawaïenne en toutes circonstances. C'est un cas d'école
classique de fine-tuning de **style/persona** : le contenu factuel du modèle
ne change pas fondamentalement, mais sa façon de répondre, si.

---

## 0. Pourquoi MLX et pas PyTorch/CUDA ?

Les Mac Apple Silicon (M1/M2/M3/M4) n'ont pas de GPU CUDA : les outils de
fine-tuning classiques de l'écosystème NVIDIA (bitsandbytes, PEFT+CUDA, etc.)
n'y fonctionnent pas ou très mal. MLX est conçu par Apple spécifiquement pour
tirer parti de l'architecture à mémoire unifiée des puces Apple Silicon (le
CPU et le GPU partagent la même RAM), ce qui permet de fine-tuner des modèles
de plusieurs milliards de paramètres directement sur un Mac, y compris un
MacBook Air, à condition d'avoir assez de RAM.

## 1. Prérequis

- **Mac Apple Silicon** (M1, M2, M3 ou M4). MLX ne fonctionne pas (ou très mal,
  sans accélération GPU) sur Mac Intel.
- **macOS 13.5 ou supérieur** (macOS 14 Sonoma recommandé).
- **Python 3.9+** (`python3 --version`). Le Python fourni par macOS convient,
  ou une installation via [Homebrew](https://brew.sh) (`brew install python3`).
- **Espace disque** : comptez 2 à 20 Go selon la taille du modèle choisi (voir
  tableau ci-dessous), plus l'espace pour le modèle fusionné si vous fusionnez
  les poids.
- **RAM** : c'est le facteur limitant principal. Indicatif pour du fine-tuning
  LoRA (le modèle de base reste quantifié en 4-bit, seuls de petits adaptateurs
  sont entraînés en pleine précision) :

  | Modèle              | RAM conseillée | Taille sur disque (4-bit) |
  |----------------------|-----------------|----------------------------|
  | Qwen2.5-0.5B-Instruct | 8 Go            | ~0.3 Go                   |
  | Qwen2.5-1.5B-Instruct | 8 Go            | ~1 Go                     |
  | Qwen2.5-3B-Instruct   | 16 Go           | ~1.7 Go                   |
  | Qwen2.5-7B-Instruct   | 16-32 Go        | ~4 Go                     |
  | Qwen2.5-14B-Instruct  | 32 Go           | ~8 Go                     |
  | Qwen2.5-32B-Instruct  | 64 Go+          | ~18 Go                    |

  Si vous débutez ou avez 8-16 Go de RAM, commencez avec **Qwen2.5-3B-Instruct**
  (ou 1.5B). Vous pourrez toujours refaire l'expérience avec un modèle plus
  gros ensuite.

## 2. Installer l'environnement

Depuis la racine du projet :

```bash
./scripts/00_setup_env.sh
source .venv/bin/activate
```

Ce script crée un environnement virtuel `.venv/` et installe `mlx-lm` (qui
embarque `mlx`). Vérifiez que tout est en place :

```bash
python3 -c "import mlx.core as mx; print(mx.default_device())"
# doit afficher quelque chose comme: Device(gpu, 0)
```

> Pensez à relancer `source .venv/bin/activate` à chaque nouveau terminal.

## 3. Télécharger le modèle de base

On utilise les modèles Qwen2.5 déjà convertis au format MLX et quantifiés en
4-bit par la communauté (organisation [`mlx-community`](https://huggingface.co/mlx-community)
sur Hugging Face) : pas besoin de conversion manuelle, et l'empreinte mémoire
est réduite d'un facteur ~4 par rapport au modèle en pleine précision.

```bash
./scripts/01_download_model.sh mlx-community/Qwen2.5-3B-Instruct-4bit
```

Le modèle est mis en cache dans `~/.cache/huggingface/`, comme n'importe quel
modèle Hugging Face. Vous pouvez remplacer `3B` par `0.5B`, `1.5B`, `7B`,
`14B`, `32B` ou `72B` selon votre machine (voir tableau ci-dessus).

## 4. Le dataset d'exemple : pizza hawaïenne 🍍🍕

Le dataset fourni dans `data/pizza_hawaienne/` (déjà généré, rien à faire)
contient 50 exemples d'entraînement + 8 de validation + 8 de test, au format
conversationnel attendu par `mlx_lm.lora` (voir
[`docs/DATASET_FORMAT.md`](DATASET_FORMAT.md) pour le détail du format).

Chaque exemple associe :
- un **system prompt** définissant le persona « Chef Aloha »,
- une **question** sur la pizza hawaïenne (recette, histoire, débats,
  accords mets-boissons, variantes...),
- une **réponse** enthousiaste et engagée qui incarne ce persona.

Le but : après fine-tuning, le modèle doit adopter ce ton avec constance,
même sur des questions reformulées qu'il n'a jamais vues (jeu de test).

Le script qui a généré ces fichiers, `scripts/prepare_dataset.py`, est
lui-même un exemple à copier/adapter pour construire **votre propre**
dataset (voir [`docs/DATASET_FORMAT.md`](DATASET_FORMAT.md)).

Avant de lancer un entraînement, validez toujours le format de votre
dataset :

```bash
python3 scripts/validate_dataset.py data/pizza_hawaienne
```

## 5. Lancer le fine-tuning LoRA

```bash
./scripts/02_train_lora.sh
```

Ce script appelle `mlx_lm.lora` avec la configuration définie dans
[`config/lora_config.yaml`](../config/lora_config.yaml) (modèle, hyperparamètres,
chemins). Par défaut :

- modèle : `mlx-community/Qwen2.5-3B-Instruct-4bit`
- 300 itérations, batch size 2, 8 dernières couches adaptées
- adaptateurs sauvegardés dans `adapters/pizza_hawaienne/`

Sur un MacBook Pro M-series récent, l'entraînement sur ce petit dataset prend
généralement **quelques minutes**. Vous verrez dans la console la perte
d'entraînement (`train loss`) diminuer progressivement, ainsi que des points
de contrôle sur le jeu de validation.

Pour personnaliser rapidement sans éditer le YAML :

```bash
ITERS=500 BATCH_SIZE=4 ./scripts/02_train_lora.sh
```

Ou directement via `make` :

```bash
make train MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit
```

> **Astuce** : si vous avez une erreur de mémoire (`Metal ... out of memory`),
> réduisez `batch_size`, `max_seq_length`, ou `num_layers` dans
> `config/lora_config.yaml`. Voir [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## 6. Comparer avant/après fine-tuning

```bash
python3 scripts/03_compare_outputs.py
```

Ce script charge à la fois le modèle de base (sans adaptateur) et le modèle
avec l'adaptateur LoRA fraîchement entraîné, et affiche côte à côte leurs
réponses sur quelques prompts liés à la pizza. Vous devriez voir le modèle
fine-tuné adopter beaucoup plus systématiquement le ton passionné du persona
« Chef Aloha », y compris sur des prompts qui ne contiennent pas le system
prompt d'origine (`--no-system-prompt`).

## 7. Fusionner les poids LoRA (optionnel mais recommandé)

Les adaptateurs LoRA entraînés à l'étape 5 sont de petits fichiers de poids
qui se combinent au modèle de base *au moment du chargement*. C'est pratique
pour itérer, mais pour distribuer ou déployer facilement le modèle, il est
plus simple de **fusionner** l'adaptateur dans les poids du modèle :

```bash
./scripts/04_fuse_model.sh
```

Le modèle fusionné, autonome, est sauvegardé dans `models/pizza-hawaienne-fused/`.
Vous pouvez ensuite l'utiliser sans avoir besoin de spécifier `--adapter-path` :

```bash
mlx_lm.generate --model models/pizza-hawaienne-fused --prompt "Que penses-tu de l'ananas sur la pizza ?"
```

## 8. Discuter avec votre modèle fine-tuné

Un petit chat interactif en ligne de commande est fourni :

```bash
# Avec l'adaptateur LoRA (non fusionné)
python3 scripts/05_chat.py --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter-path adapters/pizza_hawaienne

# Avec le modèle fusionné
python3 scripts/05_chat.py --model models/pizza-hawaienne-fused
```

Tapez `exit` pour quitter, `reset` pour repartir d'une conversation vide.

## 9. (Optionnel) Utiliser le modèle avec Ollama / LM Studio via GGUF

MLX et GGUF (format utilisé par `llama.cpp`, Ollama, LM Studio) sont deux
formats différents. Pour utiliser votre modèle fine-tuné dans Ollama :

1. Fusionnez d'abord l'adaptateur (étape 7) — GGUF ne supporte pas les
   adaptateurs LoRA MLX directement.
2. Le modèle fusionné MLX est déjà quantifié pour MLX ; pour l'exporter en
   GGUF, repartez du modèle Hugging Face **non quantifié pour MLX** de base
   (ex. `Qwen/Qwen2.5-3B-Instruct`), refaites la fusion sur cette base avec
   `mlx_lm.fuse --de-quantize` si vous êtes parti d'un modèle 4-bit, puis
   utilisez le script `convert_hf_to_gguf.py` du projet
   [`llama.cpp`](https://github.com/ggml-org/llama.cpp) pour convertir le
   dossier de sortie (format Hugging Face `safetensors`) en `.gguf`.
3. Chargez le fichier `.gguf` obtenu dans Ollama avec un `Modelfile` minimal :

   ```
   FROM ./pizza-hawaienne.gguf
   ```

   ```bash
   ollama create chef-aloha -f Modelfile
   ollama run chef-aloha
   ```

Cette étape est **optionnelle** : la boucle MLX seule (entraînement + fusion +
chat) suffit pour la plupart des usages locaux sur Mac.

## 10. Aller plus loin

- Remplacez le dataset `pizza_hawaienne` par le vôtre en suivant
  [`docs/DATASET_FORMAT.md`](DATASET_FORMAT.md).
- Essayez un modèle plus gros (`7B`, `14B`) si votre RAM le permet — la
  qualité du persona/style appris s'améliore généralement avec la taille du
  modèle.
- Augmentez `num_layers` dans `config/lora_config.yaml` (jusqu'à `-1` pour
  toutes les couches) pour un fine-tuning plus profond, au prix de plus de
  mémoire et de temps d'entraînement.
- Essayez `fine_tune_type: dora` dans la config pour DoRA, une variante de
  LoRA qui donne parfois de meilleurs résultats à budget d'entraînement égal.
- En cas de blocage, consultez [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
