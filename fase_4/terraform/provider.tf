terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }

  backend "gcs" {
    # Substitua pelo nome do bucket criado manualmente:
    # gsutil mb -p fiap-tech-challenge-4 -l us-central1 gs://fiap-tech-challenge-4-tfstate
    bucket = "fiap-tech-challenge-4-tfstate"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Ativação das APIs necessárias para a Fase 4
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "texttospeech.googleapis.com",
    "aiplatform.googleapis.com",
    "generativelanguage.googleapis.com"
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = true
}