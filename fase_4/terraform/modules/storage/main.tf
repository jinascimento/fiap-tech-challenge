resource "google_storage_bucket" "audio_bucket" {
  name          = "${var.app_name}-storage-bucket"
  location      = "us-central1"
  force_destroy = true
  
  versioning {
    enabled = false
  }
}

output "bucket_name" {
  value = google_storage_bucket.audio_bucket.name
}

output "bucket_url" {
  value = google_storage_bucket.audio_bucket.url
}