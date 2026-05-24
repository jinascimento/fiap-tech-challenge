variable "project" {
  description = "ID do projeto no GCP"
  type        = string
}

variable "region" {
  description = "Regiao principal do GCP"
  type        = string
}

variable "bucket_name" {
  description = "Nome do bucket usado pela aplicacao"
  type        = string
}

variable "app_name" {
  description = "Nome do projeto para prefixar recursos"
  type        = string
}

variable "container_image" {
  description = "Imagem efetiva a ser usada pelo Cloud Run"
  type        = string
  default     = null
}
