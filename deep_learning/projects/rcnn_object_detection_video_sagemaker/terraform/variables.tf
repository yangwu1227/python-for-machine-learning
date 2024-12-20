variable "region" {
  type        = string
  description = "AWS region where resources will be deployed"
}

variable "profile" {
  type        = string
  description = "AWS configuration profile with permissions to provision resources"
}

variable "project_prefix" {
  type        = string
  description = "Prefix to use when naming all resources for the project"
}

variable "s3_bucket" {
  type        = string
  description = "Name of the S3 bucket for storing model artifacts, inference inputs and outputs"
}
