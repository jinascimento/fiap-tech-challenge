module "storage" {
  source   = "./modules/storage"
  project  = var.project_id
  app_name = var.app_name
}

module "database" {
  source  = "./modules/database"
  project = var.project_id
  region  = var.region
}

module "compute" {
  source      = "./modules/compute"
  project     = var.project_id
  region      = var.region
  app_name    = var.app_name
  bucket_name = module.storage.bucket_name
  container_image = var.container_image
}