---
name: generate-dataset
description: Génère un nouveau dataset de fine-tuning (train.jsonl/valid.jsonl/test.jsonl, format chat mlx-lm utilisé par ce projet) à partir d'un topic libre, d'un document local, ou d'une page Wikipedia. Utiliser quand l'utilisateur demande de créer/générer un dataset d'entraînement, par exemple "crée un dataset sur le café", "génère-moi un dataset à partir de cette page wikipedia", "fabrique un dataset depuis ce fichier/document", "ajoute un dataset comme pizza_hawaienne mais sur X".
---

# Génération de dataset de fine-tuning

Cette skill produit un dataset complet (`train.jsonl` / `valid.jsonl` /
`test.jsonl`) au même format que `data/pizza_hawaienne/` (voir
`docs/DATASET_FORMAT.md`), prêt à être utilisé avec `scripts/02_train_lora.sh`.

Elle part de l'une de ces trois sources, selon ce que l'utilisateur fournit :

1. **Un topic libre** (ex. « le café filtre », « les VAE électriques ») — le
   contenu est généré à partir des connaissances du modèle.
2. **Un document local** (ex. un fichier `.md`, `.txt`, `.pdf`) — le contenu
   est extrait de ce document.
3. **Une page Wikipedia** (URL ou titre d'article) — le contenu est extrait de
   la page.

## 1. Déterminer les paramètres

À partir de la demande de l'utilisateur, identifie :

- **`source_type`** : `topic`, `document`, ou `wikipedia`. Si ambigu, demande.
- **`source`** : le texte du topic, le chemin du fichier, ou l'URL/titre
  Wikipedia.
- **`slug`** : nom court en `snake_case` pour le dossier `data/<slug>/` et pour
  les fichiers générés (ex. `cafe_filtre`, `velo_electrique`). Propose-en un
  à partir du topic si l'utilisateur n'en donne pas.
- **`persona`** (optionnel) : un style/personnalité comme « Chef Aloha » pour
  `pizza_hawaienne` (persona enthousiaste et engagé), ou un ton neutre par
  défaut si rien n'est demandé (« assistant expert et pédagogue sur le
  sujet »). Ne bloque pas sur ce point avec `AskUserQuestion` sauf si la
  demande est réellement ambiguë — un ton neutre par défaut est un choix
  raisonnable.
- **Taille du dataset** : par défaut, reprends les proportions de l'exemple
  existant : **50 exemples train / 8 valid / 8 test** (66 au total). Ajuste à
  la hausse si l'utilisateur demande explicitement un dataset plus gros, ou à
  la baisse si la source est pauvre en contenu (mieux vaut moins d'exemples
  mais tous bien ancrés dans la source que du remplissage répétitif).

## 2. Collecter le contenu source

### Cas `topic`

Utilise tes connaissances générales sur le sujet. Si le sujet est pointu,
récent, ou que tu n'es pas sûr des faits, fais une recherche complémentaire
(`WebSearch`, à charger via `ToolSearch` si nécessaire) avant de rédiger, pour
éviter d'halluciner des faits précis (dates, chiffres, noms).

### Cas `document`

Utilise `Read` sur le fichier fourni. Pour un fichier volumineux, lis-le par
morceaux (`offset`/`limit`) plutôt que d'essayer de tout charger d'un coup.
Pour un PDF, `Read` sait extraire le texte nativement (voir la skill `pdf` si
un traitement plus poussé est nécessaire, ex. OCR sur un PDF scanné).

### Cas `wikipedia`

Charge `WebFetch` via `ToolSearch` (`select:WebFetch`) si besoin, puis
récupère la page. Si l'utilisateur donne seulement un titre, construis l'URL
(`https://fr.wikipedia.org/wiki/<Titre>` par défaut, ou la langue demandée).
Demande à `WebFetch` d'extraire les sections factuelles principales (résumé
introductif, histoire/origine, caractéristiques, usages, controverses/débats
s'il y en a) plutôt que le HTML brut.

Dans tous les cas, extrais mentalement **15 à 30 « éléments » factuels ou
thématiques** (définitions, dates/origine clés, caractéristiques, usages,
variantes, débats/controverses, anecdotes) qui serviront de base aux paires
question/réponse — c'est ce qui garantit une bonne diversité de questions
plutôt que des répétitions autour d'un seul fait.

## 3. Générer les paires question/réponse

Inspire-toi de `data/pizza_hawaienne/train.jsonl` et de
`scripts/prepare_dataset.py` comme référence de style et de format.

- Varie les **catégories de questions** : définition/origine, histoire,
  fonctionnement/comment faire, comparaisons, idées reçues/controverses,
  anecdotes, avis/opinion (surtout si un persona est demandé), conseils
  pratiques, cas particuliers.
- Reformule les questions plutôt que de répéter la même tournure partout
  (« Qu'est-ce que... », « Pourquoi... », « Comment... », « Peut-on... »,
  « Quelle est la différence entre... », etc.).
- Réponses de 2 à 5 phrases, dans un français correct, **ancrées dans le
  contenu collecté à l'étape 2** — n'invente pas de chiffres, dates ou noms
  précis qui n'apparaissent pas dans la source ; reste plus général si
  l'information exacte manque.
- Si un persona est utilisé, applique-le de façon cohérente sur **toutes**
  les réponses (même ton, même niveau d'enthousiasme/d'expertise).
- **`valid.jsonl` et `test.jsonl` doivent contenir des questions différentes**
  de celles de `train.jsonl` (pas de simples reformulations quasi
  identiques), pour que l'évaluation soit un vrai test de généralisation —
  voir `docs/DATASET_FORMAT.md`, section "Train / valid / test".

## 4. Écrire le générateur et produire les fichiers

Crée `scripts/generate_<slug>_dataset.py` en suivant exactement la structure
de `scripts/prepare_dataset.py` :

- une constante `SYSTEM_PROMPT` (ou l'absence de system prompt si le
  persona est neutre et que l'utilisateur préfère ne pas en injecter un — voir
  `docs/DATASET_FORMAT.md`),
- des listes Python `TRAIN`, `VALID`, `TEST` de tuples `(question, réponse)`,
- les fonctions `to_jsonl()` et `main()` qui écrivent dans
  `data/<slug>/{train,valid,test}.jsonl`.

Garder ce script (plutôt que d'écrire directement les `.jsonl`) permet de
versionner la source de vérité lisible/éditable dans git, comme pour
`pizza_hawaienne`.

Exécute-le :

```bash
python3 scripts/generate_<slug>_dataset.py
```

## 5. Valider

Toujours valider avant de considérer la tâche terminée :

```bash
python3 scripts/validate_dataset.py data/<slug>
```

Corrige toute erreur signalée (JSON invalide, rôle manquant, contenu vide,
conversation qui ne finit pas par `assistant`) avant de continuer.

## 6. Rapport final à l'utilisateur

Résume en quelques lignes : nombre d'exemples train/valid/test, persona ou
ton utilisé, source(s) utilisée(s), et rappelle la commande pour lancer le
fine-tuning sur ce nouveau dataset, par exemple :

```bash
DATA=data/<slug> ADAPTER_PATH=adapters/<slug> ./scripts/02_train_lora.sh
```

(ou en éditant `data:` et `adapter_path:` dans une copie de
`config/lora_config.yaml`).
