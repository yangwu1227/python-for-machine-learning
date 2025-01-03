variable "region" {
  type        = string
  description = "AWS region where resources will be deployed"
}

variable "profile" {
  type        = string
  description = "AWS configuration profile with all required permissions"
}

variable "project_prefix" {
  type        = string
  description = "Prefix to use when naming all resources for the project"
}

variable "create_github_oidc_provider" {
  type        = bool
  description = "Boolean to decide whether to create the OIDC provider or use an existing one"
}

variable "existing_oidc_provider_arn" {
  type        = string
  description = "Amazon resource name (ARN) of the GitHub OIDC provider for authentication"
}

variable "github_username" {
  type        = string
  description = "GitHub username for accessing the repository"
}

variable "github_repo_name" {
  type        = string
  description = "Name of the GitHub repository for this project"
}

variable "s3_remote_state_bucket" {
  type        = string
  description = "S3 bucket name of the state file"
}

variable "s3_remote_state_key" {
  type        = string
  description = "S3 key prefix of the state file"
}
