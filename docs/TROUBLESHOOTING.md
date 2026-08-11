# Dépannage

## `mlx_lm.lora: command not found`

L'environnement virtuel n'est pas activé, ou l'installation a échoué.

```bash
source .venv/bin/activate
pip install -r requirements.txt
which mlx_lm.lora   # doit pointer vers .venv/bin/mlx_lm.lora
```

Si la commande reste introuvable, vous pouvez toujours invoquer le module
directement :

```bash
python3 -m mlx_lm lora --config config/lora_config.yaml
```

## `RuntimeError: ... Metal ... out of memory` / le Mac devient très lent puis plante

Vous manquez de mémoire unifiée pour ce modèle/cette configuration. Dans
l'ordre de ce qui aide le plus :

1. Réduisez `batch_size` dans `config/lora_config.yaml` (essayez `1`).
2. Réduisez `num_layers` (ex. `4` au lieu de `8`).
3. Réduisez `max_seq_length` si vos exemples sont courts (ex. `512`).
4. Activez `grad_checkpoint: true` (plus lent, mais économise de la mémoire).
5. Utilisez un modèle plus petit ou plus quantifié (ex. `Qwen2.5-3B-Instruct-4bit`
   au lieu de `Qwen2.5-7B-Instruct-4bit`, ou passez à `1.5B`/`0.5B`).
6. Fermez les autres applications gourmandes en mémoire (navigateurs avec
   beaucoup d'onglets, IDE, etc.) pendant l'entraînement.

## Le téléchargement du modèle échoue ou est très lent

- Vérifiez votre connexion réseau : les modèles font de 0.3 à plusieurs
  dizaines de Go.
- Si un téléchargement a été interrompu, relancez simplement le script :
  `huggingface_hub` reprend les téléchargements partiels.
- Pour un modèle **gated** (nécessitant d'accepter une licence sur Hugging
  Face — rarement le cas pour les modèles `mlx-community`), connectez-vous
  d'abord :

  ```bash
  huggingface-cli login
  ```

## `mlx_lm.lora --config ... --iters 500` : le flag n'est pas pris en compte

Certaines versions de `mlx-lm` exigent que les flags en ligne de commande
soient passés *après* `--config` pour surcharger le fichier YAML (c'est le
comportement attendu des scripts de ce projet). Si un flag semble ignoré,
vérifiez son nom exact avec :

```bash
mlx_lm.lora --help
```

Les noms de flags peuvent légèrement varier d'une version de `mlx-lm` à
l'autre : ce projet a été écrit avec `mlx-lm>=0.20.0`. En cas de doute,
`--help` fait toujours foi.

## La perte d'entraînement (`train loss`) ne descend pas / le modèle ne change pas de comportement

- Vérifiez que le dataset est bien formé : `python3 scripts/validate_dataset.py data/mon_dataset`.
- Augmentez `iters` (300 est volontairement bas pour une démo rapide).
- Augmentez `num_layers` (plus de couches adaptées = plus de capacité
  d'apprentissage, mais plus de risque de sur-apprentissage sur un petit
  dataset).
- Vérifiez que `learning_rate` n'est pas trop faible (1e-5 est un point de
  départ raisonnable ; essayez 2e-5 ou 5e-5 si rien ne bouge après plusieurs
  centaines d'itérations).
- Avec un très petit dataset (comme l'exemple pizza hawaïenne), un peu de
  répétition/sur-apprentissage volontaire est normal et même recherché pour
  bien "graver" un style particulier.

## Le modèle fine-tuné "oublie" des choses ou devient incohérent hors du sujet d'entraînement

C'est le signe d'un **sur-apprentissage** (catastrophic forgetting) : trop
d'itérations, trop de couches adaptées, ou dataset trop répétitif/étroit.

- Réduisez `iters` et/ou `num_layers`.
- Mélangez votre dataset avec des exemples d'instruction générale.
- Réduisez `lora_parameters.rank`/`alpha` pour limiter la capacité de
  l'adaptateur.

## `mlx_lm.fuse` échoue avec une erreur liée à la quantification

Si vous êtes parti d'un modèle déjà quantifié en 4-bit
(`*-4bit`), la fusion standard fonctionne nativement (l'adaptateur est fusionné
dans les poids quantifiés). Si vous avez besoin d'un modèle fusionné en pleine
précision (par exemple pour reconvertir ensuite en GGUF ou re-quantifier
différemment), utilisez :

```bash
mlx_lm.fuse --model <modèle> --adapter-path <adaptateur> --save-path <sortie> --de-quantize
```

## Le chat interactif (`scripts/05_chat.py`) affiche du texte tronqué ou incohérent en fin de réponse

Augmentez `--max-tokens` (par défaut 512) si les réponses sont coupées. Si le
texte devient incohérent après un certain nombre de tours, essayez `reset`
pour vider l'historique : les longues conversations peuvent dépasser la
fenêtre de contexte effective du modèle sur un dataset de fine-tuning très
court.

## Autre problème

1. Vérifiez la version de `mlx-lm` : `pip show mlx-lm`.
2. Consultez la documentation officielle du projet
   [mlx-lm](https://github.com/ml-explore/mlx-lm) et
   [mlx-examples](https://github.com/ml-explore/mlx-examples), qui évoluent
   rapidement et peuvent avoir mis à jour certains flags depuis la rédaction
   de ce guide.
3. `mlx_lm.lora --help`, `mlx_lm.fuse --help`, `mlx_lm.generate --help`
   restent la source de vérité la plus à jour pour les options disponibles
   sur votre version installée.
