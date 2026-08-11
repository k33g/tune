.PHONY: setup download validate train compare fuse chat clean

MODEL ?= mlx-community/Qwen2.5-3B-Instruct-4bit
DATA ?= data/pizza_hawaienne
ADAPTER_PATH ?= adapters/pizza_hawaienne
SAVE_PATH ?= models/pizza-hawaienne-fused

setup: ## Crée le venv et installe les dépendances
	./scripts/00_setup_env.sh

download: ## Télécharge le modèle de base
	./scripts/01_download_model.sh $(MODEL)

validate: ## Valide le format du dataset
	python3 scripts/validate_dataset.py $(DATA)

train: ## Lance le fine-tuning LoRA
	MODEL=$(MODEL) ADAPTER_PATH=$(ADAPTER_PATH) ./scripts/02_train_lora.sh

compare: ## Compare les réponses base vs fine-tuné
	python3 scripts/03_compare_outputs.py --model $(MODEL) --adapter-path $(ADAPTER_PATH)

fuse: ## Fusionne l'adaptateur LoRA dans le modèle
	MODEL=$(MODEL) ADAPTER_PATH=$(ADAPTER_PATH) SAVE_PATH=$(SAVE_PATH) ./scripts/04_fuse_model.sh

chat: ## Chat interactif avec le modèle fine-tuné (adaptateur non fusionné)
	python3 scripts/05_chat.py --model $(MODEL) --adapter-path $(ADAPTER_PATH)

chat-fused: ## Chat interactif avec le modèle fusionné
	python3 scripts/05_chat.py --model $(SAVE_PATH)

clean: ## Supprime les artefacts d'entraînement (adaptateurs, modèles fusionnés)
	rm -rf adapters models
