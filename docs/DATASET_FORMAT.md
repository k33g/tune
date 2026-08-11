# Format du dataset

`mlx_lm.lora` attend un dossier contenant jusqu'à trois fichiers
[JSON Lines](https://jsonlines.org/) (un objet JSON par ligne, sans virgule
entre les lignes) :

```
data/mon_dataset/
├── train.jsonl   # obligatoire — exemples d'entraînement
├── valid.jsonl   # obligatoire — exemples de validation (suivi pendant l'entraînement)
└── test.jsonl    # optionnel — exemples pour évaluation finale (mlx_lm.lora --test)
```

## Format "chat" (recommandé pour les modèles Instruct comme Qwen2.5)

Chaque ligne est un objet avec une clé `"messages"`, suivant le format de
chat standard (identique à celui de l'API OpenAI / Hugging Face) :

```json
{"messages": [
  {"role": "system", "content": "Tu es Chef Aloha, un chef passionné..."},
  {"role": "user", "content": "Que penses-tu de l'ananas sur la pizza ?"},
  {"role": "assistant", "content": "C'est un mariage parfait entre le sucré et le salé !"}
]}
```

`mlx_lm.lora` applique automatiquement le **chat template** du tokenizer de
Qwen2.5 pour transformer ces messages en texte brut, et masque la perte
(loss) sur tout ce qui n'est pas la réponse de l'assistant : le modèle
n'apprend qu'à générer de meilleures réponses, pas à reproduire les questions.

C'est le format utilisé par `data/pizza_hawaienne/` dans ce projet.

### Le system prompt : à inclure ou non ?

- **Persona/style constant** (comme "Chef Aloha") : incluez le même system
  prompt dans chaque exemple d'entraînement. Le modèle apprend à associer ce
  system prompt à ce comportement. À l'inférence, réutilisez-le pour de
  meilleurs résultats (mais après un fine-tuning suffisant, le comportement
  s'estompe rarement complètement même sans lui — testez avec
  `--no-system-prompt` dans `scripts/03_compare_outputs.py`).
- **Pas de system prompt** : omettez la clé `"system"` dans `messages`.
  Utile si vous voulez que le comportement appris soit actif par défaut, sans
  dépendre d'un prompt particulier à l'inférence.

## Format "texte brut" (alternative simple)

Pour du texte continu sans structure de dialogue (ex. adapter le style
d'écriture du modèle sur un corpus), vous pouvez aussi utiliser une clé
`"text"` :

```json
{"text": "La pizza hawaïenne a été inventée en 1962 par Sam Panopoulos..."}
```

Ce format est plus simple mais moins adapté aux modèles *Instruct* comme
Qwen2.5-Instruct, qui sont entraînés pour du dialogue structuré. Préférez le
format `"messages"` sauf cas particulier (fine-tuning d'un modèle de base
*non*-Instruct, "text completion" pure).

## Génération automatique via la skill `generate-dataset`

Ce projet fournit une skill Claude Code (`.claude/skills/generate-dataset/`)
qui automatise la construction d'un nouveau dataset à partir :

- d'un **topic libre** (ex. « le café filtre »),
- d'un **document local** (`.md`, `.txt`, `.pdf`...),
- ou d'une **page Wikipedia** (URL ou titre).

Dans une session Claude Code ouverte sur ce repo, demandez par exemple :
« crée un dataset sur le café filtre » ou « génère un dataset à partir de
cette page Wikipedia : ... ». La skill collecte le contenu source, génère des
paires question/réponse variées, écrit un générateur
`scripts/generate_<slug>_dataset.py` (sur le modèle de
`scripts/prepare_dataset.py`), produit `data/<slug>/{train,valid,test}.jsonl`,
puis valide le résultat avec `scripts/validate_dataset.py`. Voir
`.claude/skills/generate-dataset/SKILL.md` pour le détail du processus.

## Construire votre propre dataset (manuellement)

1. **Partez de `scripts/prepare_dataset.py`** : copiez-le, remplacez la liste
   `TRAIN`/`VALID`/`TEST` (paires question/réponse) et le `SYSTEM_PROMPT` par
   vos propres données, puis exécutez-le pour générer vos `.jsonl`.

2. **Validez toujours le format** avant d'entraîner :

   ```bash
   python3 scripts/validate_dataset.py data/mon_dataset
   ```

   Ce script vérifie que chaque ligne est un JSON valide, que les rôles
   alternent correctement, et que rien n'est vide. Utile pour détecter une
   erreur de format *avant* de charger un modèle de plusieurs Go.

3. **Combien d'exemples ?**

   | Objectif                                   | Ordre de grandeur       |
   |---------------------------------------------|--------------------------|
   | Démonstration / prototypage rapide (comme ici) | 30-100 exemples         |
   | Style/persona robuste                        | 200-1000 exemples       |
   | Nouvelle connaissance factuelle fiable        | Souvent > 1000, ou RAG plutôt que fine-tuning |
   | Format de sortie strict (JSON, code...)       | 100-500 exemples très cohérents |

   Le dataset `pizza_hawaienne` fourni ici (66 exemples) est volontairement
   petit à des fins pédagogiques : il suffit à démontrer un changement de
   *style* perceptible en quelques minutes d'entraînement, mais un usage réel
   demandera généralement plus de données, surtout pour enseigner de nouvelles
   connaissances factuelles plutôt qu'un style.

4. **Qualité > quantité.** Un petit nombre d'exemples cohérents et bien écrits
   donne de meilleurs résultats qu'un grand nombre d'exemples répétitifs ou
   contradictoires. Variez la formulation des questions, mais gardez le
   *ton* des réponses cohérent si vous entraînez un persona/style.

5. **Évitez le "catastrophic forgetting".** Un fine-tuning LoRA agressif
   (beaucoup d'itérations, beaucoup de couches) sur un dataset très
   spécialisé et répétitif peut faire "oublier" au modèle des capacités
   générales. Pour un usage en production, il est courant de mélanger vos
   exemples spécifiques avec un échantillon de données d'instruction
   générales, ou de limiter `num_layers` et `iters` à ce qui est nécessaire.

6. **Train / valid / test** : gardez ces ensembles disjoints (pas de
   questions quasi identiques entre `train.jsonl` et `valid.jsonl`/`test.jsonl`),
   sinon la perte de validation ne reflète pas la vraie capacité de
   généralisation du modèle. C'est le cas dans `data/pizza_hawaienne/` :
   les questions de `valid.jsonl` et `test.jsonl` sont différentes de celles
   de `train.jsonl`.

## Longueur des séquences

`mlx_lm.lora` tronque les exemples plus longs que `max_seq_length` (1024 par
défaut dans `config/lora_config.yaml`). Si vos exemples sont plus longs
(ex. de gros documents), augmentez cette valeur — au prix de plus de mémoire
et d'un entraînement plus lent. Utilisez
`python3 scripts/validate_dataset.py data/mon_dataset --model <modèle>` pour
estimer la longueur moyenne en tokens de vos exemples.
