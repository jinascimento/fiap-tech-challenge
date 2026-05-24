# Repositório de Imagens
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "${var.app_name}-repo"
  description   = "Docker repository for Streamlit App"
  format        = "DOCKER"
}

locals {
  default_container_image = "${var.region}-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.repo.repository_id}/app:latest"
}

# Service Account para o Cloud Run
resource "google_service_account" "run_sa" {
  account_id   = "${var.app_name}-run-sa"
  display_name = "Service Account for Cloud Run"
}

# Permissões para GCS
resource "google_storage_bucket_iam_member" "gcs_admin" {
  bucket = var.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.run_sa.email}"
}

# Permissões para Firestore
resource "google_project_iam_member" "firestore_user" {
  project = var.project
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.run_sa.email}"
}

# Permissões para Vertex AI (Gemini)
resource "google_project_iam_member" "ai_user" {
  project = var.project
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.run_sa.email}"
}

# Permissão genérica para consumir APIs do projeto (inclui TTS)
resource "google_project_iam_member" "service_usage" {
  project = var.project
  role    = "roles/serviceusage.serviceUsageConsumer"
  member  = "serviceAccount:${google_service_account.run_sa.email}"
}

# Cloud Run Service
resource "google_cloud_run_v2_service" "app" {
  name                = "${var.app_name}-service"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.run_sa.email
    containers {
      image = coalesce(var.container_image, local.default_container_image)
      
      env {
        name  = "BUCKET_NAME"
        value = var.bucket_name
      }
      env {
        name  = "GCP_PROJECT"
        value = var.project
      }
    }
  }

  depends_on = [google_artifact_registry_repository.repo]
}

# Tornar o serviço público (opcional, dependendo da necessidade)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
