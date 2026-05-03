output "bucket_url" {
  description = "URL do bucket para upload de arquivos"
  value       = module.storage.bucket_url
}

output "service_url" {
  description = "URL da aplicação rodando no Cloud Run"
  value       = module.compute.service_url
}