variable "project_id" {
  description = "ID do projeto no GCP"
  type        = string
}

variable "region" {
  description = "Região principal do GCP"
  type        = string
  default     = "us-central1"
}

variable "app_name" {
  description = "Nome do projeto para prefixar recursos"
  type        = string
  default     = "tech-challenge-4"
}

variable "container_image" {
  description = "Imagem efetiva do Cloud Run. Quando nula, usa a tag latest do repositório."
  type        = string
  default     = null
}