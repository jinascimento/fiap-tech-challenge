# Variáveis de Projeto
PROJECT_ID=gen-lang-client-0574046440
REGION=us-central1
APP_NAME=tc4-saude-mulher
REPO_NAME=$(APP_NAME)-repo
IMAGE_NAME=$(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/app
IMAGE_TAG=$(IMAGE_NAME):latest
GIT_SHA=$(shell git rev-parse --short HEAD 2>/dev/null || printf '%s' dev)
IMAGE_SHA_TAG=$(IMAGE_NAME):$(GIT_SHA)

# Caminhos
BASE_DIR=.
TF_DIR=$(BASE_DIR)/terraform
IMAGE_DIGEST_FILE=$(TF_DIR)/image.auto.tfvars

.PHONY: help build ensure-app push tf-init tf-plan tf-apply tf-destroy deploy run-local gen-audio

help:
	@echo "Comandos disponíveis (rodar dentro da pasta fase_4):"
	@echo "  make build       - Constrói a imagem Docker com as tags latest e git-sha"
	@echo "  make ensure-app  - Garante APIs e Artifact Registry via Terraform"
	@echo "  make push        - Envia latest e git-sha e grava o digest usado no Cloud Run"
	@echo "  make tf-init     - Inicializa o Terraform"
	@echo "  make tf-plan     - Gera o plano de execução do Terraform"
	@echo "  make tf-apply    - Aplica as mudanças do Terraform"
	@echo "  make tf-destroy  - Destrói a infraestrutura"
	@echo "  make deploy      - Build, Push e Apply (Fluxo completo)"
	@echo "  make run-local   - Roda o Streamlit localmente"
	@echo "  make gen-audio   - Gera os áudios de teste via TTS"

build:
	docker build --platform linux/amd64 -t $(IMAGE_TAG) -t $(IMAGE_SHA_TAG) -f $(BASE_DIR)/Dockerfile $(BASE_DIR)

ensure-app:
	cd $(TF_DIR) && terraform init -input=false
	cd $(TF_DIR) && terraform apply -auto-approve \
		-target='google_project_service.required_apis["artifactregistry.googleapis.com"]'
	@echo "Aguardando 30 segundos para propagação da API..."
	sleep 30
	# Tenta importar o repositório caso ele já exista no GCP (ignora erro se não existir)
	-cd $(TF_DIR) && terraform import module.compute.google_artifact_registry_repository.repo projects/$(PROJECT_ID)/locations/$(REGION)/repositories/$(REPO_NAME)
	# Garante que o recurso está criado e no estado correto
	cd $(TF_DIR) && terraform apply -auto-approve \
		-target='module.compute.google_artifact_registry_repository.repo'

push: ensure-app
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet
	docker push $(IMAGE_TAG)
	docker push $(IMAGE_SHA_TAG)
	@DIGEST=$$(gcloud artifacts docker images describe $(IMAGE_SHA_TAG) --format='get(image_summary.digest)'); \
		test -n "$$DIGEST"; \
		printf 'container_image = "%s@%s"\n' "$(IMAGE_NAME)" "$$DIGEST" > $(IMAGE_DIGEST_FILE); \
		echo "Imagem ativa do Cloud Run registrada em $(IMAGE_DIGEST_FILE): $(IMAGE_NAME)@$$DIGEST"

tf-init:
	cd $(TF_DIR) && terraform init

tf-plan:
	cd $(TF_DIR) && terraform plan

tf-apply:
	cd $(TF_DIR) && terraform apply -auto-approve

tf-destroy:
	cd $(TF_DIR) && terraform destroy -auto-approve

deploy: build push tf-apply
	@echo "Deploy concluído com sucesso!"

run-local:
	uv run streamlit run $(BASE_DIR)/ui.py

gen-audio:
	uv run python $(BASE_DIR)/generate_test_audio.py
