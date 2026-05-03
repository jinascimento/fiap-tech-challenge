# FIAP Tech Challenge - Fase 4

Este repositório contém o projeto desenvolvido para a **Fase 4** do curso de Pós-Graduação em IA. O foco desta etapa é o processamento de áudio para detecção de eventuais problemas de saúde mental (ansiedade, violência, pós-parto) utilizando modelos fundacionais do Google Cloud (Gemini) e infraestrutura automatizada.

## 🚀 Infraestrutura e Deploy

A infraestrutura é gerenciada via **Terraform** e o ciclo de vida da aplicação é automatizado através de um **Makefile**.

### Pré-requisitos
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) instalado e autenticado.
- [Terraform](https://developer.hashicorp.com/terraform/downloads) instalado.
- [Docker](https://www.docker.com/) instalado e rodando.
- Projeto no GCP criado com faturamento (billing) ativo.

### Comandos de Automação (Makefile)
Dentro da pasta `fase_4`, você pode utilizar os seguintes comandos:

- `make deploy`: Executa o fluxo completo (Build da imagem -> Push para o Artifact Registry -> Aplicação do Terraform).
- `make build`: Constrói a imagem Docker localmente (forçando plataforma linux/amd64).
- `make push`: Garante a existência do repositório no GCP e envia a imagem.
- `make tf-apply`: Sincroniza e aplica as mudanças de infraestrutura.
- `make tf-destroy`: Remove todos os recursos criados no GCP (com auto-approve).

> **Nota:** O `make deploy` é resiliente e lida automaticamente com a ativação de APIs e importação de recursos caso o repositório já exista.

## 🎙️ Geração de Áudios de Teste

O script `generate_test_audio.py` gera amostras de áudio utilizando o **Google Text-to-Speech** para simular diferentes cenários clínicos.

### Como Executar
```bash
make gen-audio
```
Os áudios serão gerados no diretório `fase_4/test_audios`.

## 🛠️ Configuração e Arquitetura

### Configuração Centralizada
As configurações seguem o padrão adotado nas fases anteriores, centralizadas em `fase_4/config/settings.py`.
- **GCP:** `PROJECT_ID`, `BUCKET_NAME` e `GCP_LOCATION`.
- **IA:** Integração com Vertex AI (Gemini 1.5 Flash) para análise de sentimentos e transcrição.
- **IAM:** O Cloud Run utiliza uma Service Account customizada com permissões mínimas necessárias (`Vertex AI User`, `Storage Object Admin`, `Datastore User`).

### Logging
O log está centralizado em `fase_4/config/logger.py`.
- Arquivo de log padrão: `fase_4/logs/fase_4.log`.

## 💻 Execução Local

Para rodar a interface Streamlit localmente:
```bash
make run-local
```

## 🔒 Segurança
- O arquivo `.gitignore` está configurado para proteger credenciais (`google-credentials.json`) e estados do Terraform (`.tfstate`).
- O arquivo `.terraform.lock.hcl` é versionado para garantir consistência entre ambientes.
