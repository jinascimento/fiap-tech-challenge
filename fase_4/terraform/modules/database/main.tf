resource "google_firestore_database" "database" {
  project     = var.project
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  deletion_policy        = "DELETE"
  delete_protection_state = "DELETE_PROTECTION_DISABLED"
}
