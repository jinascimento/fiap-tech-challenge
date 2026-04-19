output "service_url" {
  description = "URL da aplicação rodando no Cloud Run"
  value       = google_cloud_run_v2_service.app.uri
}

output "repository_url" {
  description = "URL do Artifact Registry para push da imagem"
  value       = "${var.region}-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.repo.repository_id}/app"
}
