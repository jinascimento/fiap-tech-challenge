# Variáveis de Projeto
PROJECT_ID=gen-lang-client-0574046440
REGION=us-central1
APP_NAME=tc4-saude-mulher
REPO_NAME=$(APP_NAME)-repo
IMAGE_TAG=$(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/app:latest

# Caminhos
BASE_DIR=.
TF_DIR=$(BASE_DIR)/terraform

.PHONY: help build ensure-app push tf-init tf-plan tf-apply tf-destroy deploy run-local gen-audio

help:
	@echo "Comandos disponíveis (rodar dentro da pasta fase_4):"
	@echo "  make build       - Constrói a imagem Docker para linux/amd64"
	@echo "  make ensure-app  - Garante APIs e Artifact Registry via Terraform"
	@echo "  make push        - Envia a imagem para o Artifact Registry"
	@echo "  make tf-init     - Inicializa o Terraform"
	@echo "  make tf-plan     - Gera o plano de execução do Terraform"
	@echo "  make tf-apply    - Aplica as mudanças do Terraform"
	@echo "  make tf-destroy  - Destrói a infraestrutura"
	@echo "  make deploy      - Build, Push e Apply (Fluxo completo)"
	@echo "  make run-local   - Roda o Streamlit localmente"
	@echo "  make gen-audio   - Gera os áudios de teste via TTS"

build:
	docker build --platform linux/amd64 -t $(IMAGE_TAG) -f $(BASE_DIR)/Dockerfile $(BASE_DIR)

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
